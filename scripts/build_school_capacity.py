"""Attach capacity and infrastructure data to the final Kerala school layer.

Sources (one row per school, keyed by the 5-digit KITE school code):
- KITE Sametham school pages (current academic year): students per standard, HSS/VHSS students,
  staff posts, the 51-item infrastructure table. Parsed by fetch_sametham_school_details.py --all.
- UDISE "Basic Details of Schools" (India Data Portal, retrieved 2022-01-12): class rooms, teachers,
  students per class, category, rural/urban, year of establishment. Linked by the UDISE code that
  the Sametham page lists (fallback: the code already in the school layer).

capacity_students = Sametham Std 1-10 + HSS + VHSS students; UDISE students where Sametham has none.

Class teachers per level (teachers_lp/up/hs/hss) are counted from the staff posts listed on the Sametham page,
by Kerala post name: L P School Assistant / Teacher (L P School) -> LP; UP School Assistant / Teacher (U P School)
/ High School Assistant (lower scale) -> UP; High School Assistant <subject> -> HS; HSST <subject> -> HSS.
P D Teachers teach primary classes and are shared between LP and UP by pupils. The headmaster counts as a
teacher only in schools with up to 150 pupils (RTE Act 2009 Schedule: the head teacher is part of the required
teachers below 151 children). Language teachers (Arabic, Hindi, Sanskrit, Urdu), specialist teachers (PE,
drawing, music, sewing, craft) and non-teaching staff are counted separately, not as class teachers.

Output: FINAL COMPLETE DATASET/schools/kerala_schools_capacity.csv
Usage:  python scripts/build_school_capacity.py
"""

from pathlib import Path

import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHOOLS = ROOT / "FINAL COMPLETE DATASET" / "schools" / "kerala_schools_govt_aided_final.csv"
SAMETHAM = ROOT / "data" / "interim" / "schools" / "sametham_school_details.csv"
UDISE = ROOT / "data" / "raw" / "udise_basic_details" / "basic-details-of-schools.csv"
OUT = ROOT / "FINAL COMPLETE DATASET" / "schools" / "kerala_schools_capacity.csv"
HSCAP = ROOT / "data" / "interim" / "schools" / "hscap_hss_courses.csv"  # notebook 22: HSS batches per school, 2026
# G.O.(Rt) No.5293/2026/GEDN (HSCAP 2026 prospectus erratum): "50 merit seats per batch"
HSS_SEATS_PER_BATCH = 50
VHSCAP = ROOT / "data" / "interim" / "schools" / "vhscap_vhse_batches.csv"  # notebook 23: one row per VHSE batch, 2026
VHSE_SEATS_PER_BATCH = 30  # VHSE prospectus 2026 (G.O.(Rt) No.3537/2026/GEDN), section 21: 30 students per batch
# Sametham "Class Wise Strength 2026-27" (welcome/studsC2026), Std 1-10 totals, retrieved 2026-11-06
STATE_STD1_10 = {"Government": 1_033_009, "Aided": 1_997_616}
UDISE_COLS = ["udise_school_code", "school_category", "school_type", "management", "location_type",
              "year_of_establishment", "status", "class_from", "class_to", "class_rooms", "other_rooms",
              "total_teachers", "pre_primary_students", "class_students", "class_with_pre_primary_students"] + \
             [f"{r}_students" for r in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]]

# Kerala staff post name -> group (first match wins). Checked on all 525 school-section and 274 HSS post names.
POST_GROUPS = [
    ("non_teaching", r"clerk|attendant|attender|\bftm\b|men[ie]al|mineal|peon|sweeper|cook|typist|lab assistant|"
                     r"librarian|driver|watchman|matron|steward|accountant|office|store keeper|record|helper|ayah|"
                     r"nurse|security|workshop instructor|instructor gr"),
    ("head", r"head ?master|head ?mistres|principal"),
    ("special", r"physical education|\bpet\b|drawing|music|sewing|needle ?work|craft|work experience|art teacher|"
                r"dance|specialist|counsel"),
    ("hss", r"hsst|higher secondary school teacher"),
    ("hs_lower", r"high school ass(istan)?t.*lower scale"),
    ("hs", r"high school ass|\bhsa\b|\bhst\b"),  # before "lang": High School Assistant Hindi is an HS subject teacher
    ("lang", r"arabic|hindi|sanskrit|urd?u|tamil|kannada|language|lower grade|munshi"),
    ("up", r"u ?p school|upsa|\bupst\b|teacher \(u ?p"),
    ("lp", r"l ?p school|lpsa|\blpst\b|teacher \(l ?p"),
    ("pd", r"p ?d teacher"),
]
RTE_HEAD_COUNTS_UPTO = 150


def post_counts(posts):
    """'High School Assistant Hindi 1; UP School Assistant 2' -> number of posts per group (None: no staff list)."""
    if not isinstance(posts, str):
        return None
    out = dict.fromkeys([g for g, _ in POST_GROUPS] + ["other"], 0)
    for item in posts.split("; "):
        m = re.match(r"(.+) (\d+)$", item)
        if m:
            out[next((g for g, p in POST_GROUPS if re.search(p, m.group(1), re.I)), "other")] += int(m.group(2))
    return out


def teachers_by_level(df):
    """Class teachers per level from the Sametham staff posts (see the module docstring for the rules)."""
    groups = [g for g, _ in POST_GROUPS] + ["other"]
    school = pd.DataFrame([post_counts(p) or {} for p in df["sam_staff_school_posts"]], index=df.index, columns=groups)
    hss = pd.DataFrame([post_counts(p) or {} for p in df["sam_staff_hss_posts"]], index=df.index, columns=groups)
    has_list = df["sam_staff_school_posts"].notna()
    school, hss = school.fillna(0), hss.fillna(0)
    pupils = lambda a, b: sum(pd.to_numeric(df[f"sam_students_std{i:02d}"], errors="coerce").fillna(0) for i in range(a, b + 1))
    lp, up, hs = pupils(1, 4), pupils(5, 7), pupils(8, 10)
    lp_share = (lp / (lp + up)).where(lp + up > 0, 0.5)
    head = school["head"].where((lp + up > 0) & (lp + up + hs <= RTE_HEAD_COUNTS_UPTO), 0)
    out = pd.DataFrame({
        "teachers_lp": school["lp"] + school["pd"] * lp_share + head.where(lp > 0, 0),
        "teachers_up": school["up"] + school["hs_lower"] + school["pd"] * (1 - lp_share) + head.where(lp.eq(0) & up.gt(0), 0),
        "teachers_hs": school["hs"],
        "teachers_hss": hss["hss"],
        "teachers_language": school["lang"] + hss["lang"],
        "teachers_specialist": school["special"] + hss["special"],
        "teachers_head": school["head"] + hss["head"],
        "teachers_unclassified": school["other"] + hss["other"],
    }).round(2)
    out = out.where(has_list)  # no staff list on the page: unknown, not zero
    out["teachers_by_level_source"] = pd.Series("sametham_staff_posts", index=df.index).where(has_list)
    return out


def add_hss_batches(df):
    """HSS batches per school from HSCAP 2026, linked by the HSS code on the Sametham page. Public seats count only
    Government / Aided batches; self-financing (Un-Aided) batches inside a school are kept in a separate column."""
    if not HSCAP.exists():
        print("  HSCAP batches: not available yet (run notebook 22)")
        return df
    h = pd.read_csv(HSCAP, dtype={"hss_code": str})
    h["key"] = h["hss_code"].str.lstrip("0")  # Sametham writes some codes with a leading zero ('01139' = HSCAP '1139')
    unaided = h["batch_funding"].str.lower().str.contains("un-aided")
    agg = pd.DataFrame({"hss_batches": h[~unaided].groupby("key")["batches"].sum(),
                        "hss_courses": h[~unaided].groupby("key")["course_code"].nunique(),
                        "hss_batches_unaided": h[unaided].groupby("key")["batches"].sum()})
    agg[["hss_batches", "hss_courses"]] = agg[["hss_batches", "hss_courses"]].fillna(0)  # only self-financing batches
    n = len(df)
    df = df.merge(agg, left_on=df["sam_hss_code"].str.lstrip("0"), right_index=True, how="left").drop(columns="key_0")
    assert len(df) == n, "an HSS code matched more than one HSCAP school"
    df["hss_seats_per_year"] = df["hss_batches"] * HSS_SEATS_PER_BATCH
    df["hss_capacity_2yr"] = df["hss_seats_per_year"] * 2  # Plus One + Plus Two, for the 16-17 age group
    df["hss_batches_source"] = pd.Series("hscap_2026", index=df.index).where(df["hss_batches"].notna())
    if VHSCAP.exists():  # Vocational Higher Secondary: one row per batch, linked by the Sametham "VHSE Code"
        v = pd.read_csv(VHSCAP, dtype={"vhse_code": str}).groupby("vhse_code").size().rename("vhse_batches")
        df = df.merge(v, left_on="sam_vhse_code", right_index=True, how="left")
        assert len(df) == n, "a VHSE code matched more than one VHSCAP school"
        df["vhse_seats_per_year"] = df["vhse_batches"] * VHSE_SEATS_PER_BATCH
        df["vhse_capacity_2yr"] = df["vhse_seats_per_year"] * 2
    return df


def main():
    s = pd.read_csv(SCHOOLS, dtype={"school_code_num": str, "udise_code": str}, low_memory=False)
    s = s[["school_code_num", "school_name", "district", "finance_type", "level", "lat", "lon", "udise_code"]]
    sam = pd.read_csv(SAMETHAM, dtype=str).drop_duplicates("school_code_num", keep="last")
    sam = sam.drop(columns=[c for c in ["address", "revenue_district", "pin_code"] if c in sam])
    sam = sam.rename(columns={c: f"sam_{c}" for c in sam if c != "school_code_num"})
    df = s.merge(sam, on="school_code_num", how="left")
    page_code = df["sam_udise_code"].where(df["sam_udise_code"].str.fullmatch(r"\d{11}", na=False))
    df["udise_code"] = page_code.fillna(df["udise_code"])

    u = pd.read_csv(UDISE, dtype=str, usecols=UDISE_COLS + ["state_name"])
    u = u[u["state_name"].eq("Kerala")].drop(columns="state_name").drop_duplicates("udise_school_code")
    u = u.rename(columns={c: f"udise_{c}" for c in u if c != "udise_school_code"})
    df = df.merge(u, left_on="udise_code", right_on="udise_school_code", how="left").drop(columns="udise_school_code")

    sam_students = sum(pd.to_numeric(df.get(f"sam_{c}"), errors="coerce").fillna(0)
                       for c in ["students_total", "students_hss_total", "students_vhss_total"] if f"sam_{c}" in df)
    has_sam = pd.to_numeric(df["sam_students_total"], errors="coerce").notna()
    udise_students = pd.to_numeric(df["udise_class_students"], errors="coerce")
    df["capacity_students"] = sam_students.where(has_sam, udise_students)
    df["capacity_source"] = pd.Series("sametham", index=df.index).where(has_sam, "udise").where(
        df["capacity_students"].notna())
    df["classrooms"] = pd.to_numeric(df["sam_infra_13_total_class_room"], errors="coerce").fillna(
        pd.to_numeric(df["udise_class_rooms"], errors="coerce"))
    teach = sum(pd.to_numeric(df.get(f"sam_staff_{p}_teaching"), errors="coerce").fillna(0)
                for p in ["school", "hss", "vhss"] if f"sam_staff_{p}_teaching" in df)
    has_staff = pd.to_numeric(df.get("sam_staff_school_teaching"), errors="coerce").notna()
    df["teachers"] = teach.where(has_staff, pd.to_numeric(df["udise_total_teachers"], errors="coerce"))
    df = pd.concat([df, teachers_by_level(df)], axis=1)
    df = add_hss_batches(df)

    dropped = df[df["capacity_students"].isna()]  # no Sametham page and no UDISE link: no capacity data, left out
    df = df[df["capacity_students"].notna()]
    has_sam = has_sam[df.index]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"dropped (no capacity data): {len(dropped)} ->",
          "; ".join(f"{c} {n}" for c, n in zip(dropped["school_code_num"], dropped["school_name"])))
    n = len(df)
    print(f"schools: {n} | Sametham page parsed: {has_sam.sum()} | UDISE linked: {df['udise_class_rooms'].notna().sum()}")
    for c in ["capacity_students", "classrooms", "teachers"]:
        print(f"  {c:18s} filled {df[c].notna().sum():>6} ({df[c].notna().mean():.1%})")
    print("  capacity_source:", df["capacity_source"].value_counts(dropna=False).to_dict())
    print("  class teachers by level (Sametham staff posts):",
          {c: round(df[c].sum()) for c in ["teachers_lp", "teachers_up", "teachers_hs", "teachers_hss",
                                           "teachers_language", "teachers_specialist", "teachers_head",
                                           "teachers_unclassified"]})
    for lvl, a, b, col in [("LP", 1, 4, "teachers_lp"), ("UP", 5, 7, "teachers_up"), ("HS", 8, 10, "teachers_hs")]:
        st = sum(pd.to_numeric(df[f"sam_students_std{i:02d}"], errors="coerce").fillna(0) for i in range(a, b + 1))
        ok = df[col] > 0
        print(f"  pupils per {lvl} teacher: {st[ok].sum() / df.loc[ok, col].sum():.1f} | "
              f"schools with {lvl} pupils but no {lvl} teacher listed: {int((st.gt(0) & ~ok).sum())}")
    hss_st = pd.to_numeric(df["sam_students_hss_total"], errors="coerce").fillna(0)
    ok = df["teachers_hss"] > 0
    print(f"  pupils per HSS teacher: {hss_st[ok].sum() / df.loc[ok, 'teachers_hss'].sum():.1f} | "
          f"schools with HSS pupils but no HSST listed: {int((hss_st.gt(0) & ~ok).sum())}")
    if "hss_batches" in df:
        p1 = pd.to_numeric(df["sam_students_hss_plus1"], errors="coerce")
        has = df["hss_batches"].gt(0) & p1.gt(0)
        all_seats = (df["hss_batches"] + df["hss_batches_unaided"].fillna(0)) * HSS_SEATS_PER_BATCH
        ratio = (p1 / all_seats)[has]
        print(f"  HSS (HSCAP 2026): {int(df['hss_batches'].notna().sum())} schools linked, "
              f"{int(df['hss_batches'].sum())} public batches (+{int(df['hss_batches_unaided'].sum())} self-financing) | "
              f"schools with Plus One pupils but no HSCAP link: {int((p1.gt(0) & df['hss_batches'].isna()).sum())}")
        print(f"  check Plus One pupils / (batches x {HSS_SEATS_PER_BATCH}): median {ratio.median():.2f}, "
              f"middle 80% {ratio.quantile(.1):.2f}-{ratio.quantile(.9):.2f}")
    if "vhse_batches" in df:
        has_code = df["sam_vhse_code"].str.fullmatch(r"9\d{5}", na=False)
        print(f"  VHSE (VHSCAP 2026): {int(df['vhse_batches'].notna().sum())} of {int(has_code.sum())} schools with a VHSE "
              f"code linked, {int(df['vhse_batches'].sum())} batches x {VHSE_SEATS_PER_BATCH} seats | "
              f"VHSE code but no VHSCAP batches: {int((has_code & df['vhse_batches'].isna()).sum())}")
    std = pd.to_numeric(df["sam_students_total"], errors="coerce")
    for ft, ref in STATE_STD1_10.items():
        got = std[df["finance_type"].eq(ft)].sum()
        print(f"  check Std 1-10 {ft}: ours {got:,.0f} vs Sametham state table {ref:,} ({got / ref:.1%})")
    print("written:", OUT)


if __name__ == "__main__":
    main()
