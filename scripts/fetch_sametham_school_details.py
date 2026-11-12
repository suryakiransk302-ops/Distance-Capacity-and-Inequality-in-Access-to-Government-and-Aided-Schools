"""Fetch school detail pages from KITE Sametham and parse them.

Each page https://sametham.kite.kerala.gov.in/<school code> lists the school's UDISE code,
address, PIN code and education/sub district, the 51-item infrastructure table (classrooms,
building, area, toilets, labs ...), staff posts (HS/LP/UP, HSS, VHSS) and the current year's
students per standard (Std 1-10, HSS +1/+2, VHSS +1/+2). The UDISE code links the school to the
UDISE "Basic Details of Schools" file. Pages are saved once in data/raw/sametham_school_pages/.

Output: data/interim/schools/sametham_school_details.csv
Usage:  python scripts/fetch_sametham_school_details.py            (schools missing coordinates)
        python scripts/fetch_sametham_school_details.py --review   (also schools whose point is under review)
        python scripts/fetch_sametham_school_details.py --all      (every Government + Aided school)
"""

import argparse
import datetime as dt
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SCHOOLS = ROOT / "data" / "processed" / "schools" / "kerala_schools_govt_aided.csv"
LOCATED = ROOT / "data" / "processed" / "schools" / "kerala_schools_govt_aided_located.csv"
RAW_DIR = ROOT / "data" / "raw" / "sametham_school_pages"
OUT_CSV = ROOT / "data" / "interim" / "schools" / "sametham_school_details.csv"
BASE = "https://sametham.kite.kerala.gov.in/"
DELAY_S = 1.2
FIELDS = {"address": "Address", "udise_code": "UDISE Code", "revenue_district": "Revenue District",
          "education_district": "Education District", "sub_district": "Sub District", "pin_code": "PIN Code",
          "is_hilly": "Is Hilly Area", "is_coastal": "Is Coastal Area",
          "hss_code": "HSS Code",    # the Higher Secondary code HSCAP uses (NIL when the school has no HSS)
          "vhse_code": "VHSE Code"}  # the 6-digit code VHSCAP uses (NIL when the school has no VHSE)


def page_text(raw):
    raw = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", " ", raw)
    t = html.unescape(re.sub(r"<[^>]+>", " | ", raw))
    return re.sub(r"(\|\s*)+", "| ", re.sub(r"\s+", " ", t))


SECTIONS = ["infra-lpuphs", "staff-lpuphs", "staff-hss", "staff-vhss",
            "students-lpuphs", "students-hss", "students-vhss"]
NON_TEACHING = re.compile(r"clerk|attendant|ftm|full time menial|part time menial|peon|sweeper|cook|typist|"
                          r"lab assistant|librarian|driver|watchman|matron|steward|accountant|office", re.I)


def sections(raw):
    pos = {s: raw.find(f'id="{s}"') for s in SECTIONS}
    order = sorted((s for s in pos if pos[s] >= 0), key=pos.get)
    return {s: raw[pos[s]:(pos[order[i + 1]] if i + 1 < len(order) else len(raw))] for i, s in enumerate(order)}


def cells(h):
    h = re.sub(r"(?s)<!--.*?-->", "", h)
    out = []
    for c in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", h):
        c = re.sub(r"<br\s*/?>", " ; ", c)
        out.append(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip())
    return out


def num(v):
    m = re.search(r"\d+(?:\.\d+)?", v or "")
    return float(m.group()) if m else None


def parse_infra(c):
    """Infrastructure table: cells come as label, ':', value triplets; keep all 51 items as text."""
    rec = {}
    for i in range(len(c) - 2):
        m = re.match(r"(\d{1,2})\. (.+)", c[i])
        if m and c[i + 1] == ":":
            slug = re.sub(r"[^a-z0-9]+", "_", m.group(2).split(";")[0].lower()).strip("_")
            rec[f"infra_{int(m.group(1)):02d}_{slug}"] = c[i + 2] or None
    return rec


def parse_staff(c, prefix):
    posts = [(p, int(n)) for p, n in (re.match(r"(.+?)\s*-\s*(\d+)$", x).groups()
                                      for x in c if re.match(r".+?\s*-\s*\d+$", x) and not x.startswith("Total Employees"))]
    tot = next((int(x.rsplit("-", 1)[1]) for x in c if x.startswith("Total Employees")), None)
    if tot is None and not posts:
        return {}
    non_t = sum(n for p, n in posts if NON_TEACHING.search(p))
    return {f"{prefix}_employees": tot if tot is not None else sum(n for _, n in posts),
            f"{prefix}_non_teaching": non_t, f"{prefix}_teaching": sum(n for _, n in posts) - non_t,
            f"{prefix}_posts": "; ".join(f"{p} {n}" for p, n in posts)}


def parse_students_std(c):
    """Std 1-10 table: rows of standard + 5 language groups x (boys, girls, total); last 3 = ALL."""
    rec, i = {}, c.index("Standard") + 1 if "Standard" in c else len(c)
    while i < len(c) and not re.fullmatch(r"\d{1,2}", c[i]):  # header row also has 'Total' cells
        i += 1
    while i + 15 < len(c) + 1 and re.fullmatch(r"\d{1,2}|Total", c[i]):
        row = c[i + 1:i + 16]
        key = "students_total" if c[i] == "Total" else f"students_std{int(c[i]):02d}"
        rec[key] = int(num(row[-1]) or 0)
        if c[i] == "Total":
            rec["students_boys"], rec["students_girls"] = int(num(row[-3]) or 0), int(num(row[-2]) or 0)
        i += 16
    if rec and "students_total" not in rec:
        rec["students_total"] = sum(v for k, v in rec.items() if k.startswith("students_std"))
    return rec


def parse_students_plus(c, prefix):
    """HSS/VHSS table: rows start with '+ 1' / '+ 2'; the last number in a row is ALL."""
    rec, cur = {}, None
    for x in c:
        m = re.fullmatch(r"\+\s*([12])", x)
        if m:
            cur = f"{prefix}_plus{m.group(1)}"
        elif x.lower().startswith("total"):
            cur = None
        elif cur and num(x) is not None:
            rec[cur] = int(num(x.split()[-1]))  # merged cells like '45 0': keep last value, overwritten until row ends
    if rec:
        rec[f"{prefix}_total"] = sum(rec.values())
    return rec


def parse(raw):
    t = page_text(raw)
    rec = {}
    for col, label in FIELDS.items():
        m = re.search(re.escape(label) + r"\s*\|\s*:\s*\|\s*([^|]*)\|", t)
        rec[col] = m.group(1).strip() if m else None
    s = {k: cells(v) for k, v in sections(raw).items()}
    rec |= parse_infra(s.get("infra-lpuphs", []))
    rec |= parse_staff(s.get("staff-lpuphs", []), "staff_school")
    rec |= parse_staff(s.get("staff-hss", []), "staff_hss")
    rec |= parse_staff(s.get("staff-vhss", []), "staff_vhss")
    rec |= parse_students_std(s.get("students-lpuphs", []))
    rec |= parse_students_plus(s.get("students-hss", []), "students_hss")
    rec |= parse_students_plus(s.get("students-vhss", []), "students_vhss")
    m = re.search(r"Students Details (\d{4}-\d{2})", t)
    rec["students_year"] = m.group(1) if m else None
    return rec


def main(all_schools, review):
    df = pd.read_csv(SCHOOLS, dtype={"school_code_num": str})
    pick = ~df["has_coords"]
    if review:
        loc = pd.read_csv(LOCATED, dtype={"school_code_num": str}, low_memory=False)
        pick |= df["school_code_num"].isin(loc.loc[loc["needs_review"].astype(str).eq("True"), "school_code_num"])
    codes = df["school_code_num"] if all_schools else df.loc[pick, "school_code_num"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "msc-spatial-inequality-study/0.1 (academic project)"
    rows, fetched = [], 0
    for n, code in enumerate(codes, 1):
        f = RAW_DIR / f"{code}.html"
        if not f.exists():
            for attempt in range(3):
                try:
                    time.sleep(DELAY_S)
                    r = session.get(BASE + code, timeout=60)
                    r.raise_for_status()
                    f.write_text(r.text, encoding="utf-8")
                    fetched += 1
                    break
                except requests.RequestException as e:
                    print(f"  retry {attempt + 1} for {code}: {e}")
                    time.sleep(10)
        if f.exists():
            rows.append({"school_code_num": code} | parse(f.read_text(encoding="utf-8")))
        if n % 100 == 0:
            print(f"{n}/{len(codes)} pages ({fetched} fetched this run)", flush=True)
    out = pd.DataFrame(rows)
    # keep the original retrieval date of pages parsed before; only newly fetched pages get today's date
    prev = pd.read_csv(OUT_CSV, usecols=["school_code_num", "retrieved_on"], dtype=str).set_index("school_code_num")[
        "retrieved_on"] if OUT_CSV.exists() else {}
    out["retrieved_on"] = out["school_code_num"].map(prev).fillna(dt.date.today().isoformat())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("schools:", len(out), "| with UDISE code:", int(out["udise_code"].str.fullmatch(r"\d{11}").fillna(False).sum()))
    print("written:", OUT_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    main(a.all, a.review)
