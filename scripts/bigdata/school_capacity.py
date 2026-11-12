"""Step D5 (schools), part 1: school places (supply) per level and the Government + Aided share of children (demand).

Supply, two versions per school and level:
  cap_main   norm-based places = teachers x pupils allowed per teacher (the main run)
  cap_enrol  pupils enrolled in that level (the sensitivity run)

Kerala (Sametham levels LP Std 1-4, UP 5-7, HS 8-10, HSS +1/+2), rules from the Kerala Education Rules (KER, 5th ed.,
corrected to 31-07-2026):
  LP   teachers_lp x 30                                   KER Rule 23A -> RTE Act 2009 Schedule (Std I-V 30:1)
  UP   teachers_up x (30 for Std 5, 35 for Std 6-7)       KER Rule 23A -> RTE Schedule (Std VI-VIII 35:1)
  HS   teachers_hs x (35 for Std 8, 45 for Std 9-10)      Rule 23A (Std 8); Rule 23: 45 pupils per division in Std IX-X,
                                                          one High School Assistant post per division (KER Ch. XXIII R. 4)
  HSS  (HSCAP public batches x 50 + VHSCAP batches x 30) x 2 years   G.O.(Rt) 5293/2026/GEDN; VHSE prospectus 2026 s.21
       cap_hss60: the same with 60 seats per HSS batch (marginal increase, sensitivity)
  A school with pupils in LP/UP/HS but no class teacher listed for that level (Sametham gives one staff list for
  Std 1-10) shares all its LP+UP+HS class teachers (else total teachers, else classrooms) between its levels by
  pupils ("split" source).
India (national levels primary 1-5, upper primary 6-8, secondary 9-10, higher secondary 11-12), UDISE 2022:
  teachers (else classrooms: RTE, one classroom per teacher) shared between the levels the school teaches by pupils,
  x 30 / 35 / 30 / 30 (RTE Schedule; RMSA PTR 30:1; 11-12: 30 assumed, no separate national norm found).

Demand share: children of school age who attend Government + Aided schools, per state and level, from UDISE 2022
(pupils of all managements and boards; Kerala with its own levels Std 1-4 / 5-7 / 8-10 / 11-12). Check on Kerala:
WorldPop children x share = 0.99 / 0.96 / 1.10 / 0.83 M vs Government + Aided pupils 2026-27 0.96 / 0.93 / 1.14 / 0.79 M.
(The Sametham class-wise table was not used for the share: it counts state-syllabus pupils only, no CBSE/ICSE,
and overstated the share, 84 % vs 61 % for LP.)
"""

import numpy as np
import pandas as pd

KERALA_LEVELS = {"lp": range(1, 5), "up": range(5, 8), "hs": range(8, 11)}
KERALA_CLASSES = {"lp": (1, 4), "up": (5, 7), "hs": (8, 10), "hss": (11, 12)}  # for the demand share
INDIA_LEVELS = {"primary": (1, 5, 30), "upper_primary": (6, 8, 35), "secondary": (9, 10, 30), "higher_secondary": (11, 12, 30)}
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
HSS_SEATS, HSS_SEATS_HIGH, VHSE_SEATS = 50, 60, 30
# catchment (metres) per level: RTE Model Rules 2010 (1 km, 3 km); Samagra Shiksha (5 km, 3 km hilly; 7 km)
CATCHMENT_M = {"lp": 1000, "up": 3000, "hs": 5000, "hs_hilly": 3000, "hss": 7000,
               "primary": 1000, "upper_primary": 3000, "secondary": 5000, "higher_secondary": 7000}
GOVT_AIDED = {"Department of Education", "Local body", "Tribal Welfare Department", "Social welfare Department",
              "Other Govt. managed schools", "Kendriya Vidyalaya", "Jawahar Navodaya Vidyalaya", "Ministry of Labor",
              "Railway School", "Others Central Government School", "Sainik School", "Central Tibetan School",
              "Government Aided", "Madarsa recognized (by Wakf board/Madarsa Board)"}  # = build_india_schools GOVT | AIDED


def _n(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def kerala_supply(k):
    """k: silver schools_kerala (pandas). Returns one row per school and level with cap_main, cap_enrol, source."""
    k = k[k["lat"].notna()].copy()
    p = {s: _n(k[f"sam_students_std{s:02d}"]) for s in range(1, 11)}
    pup = {lv: sum(p[s] for s in st) for lv, st in KERALA_LEVELS.items()}
    tch = {lv: _n(k[f"teachers_{lv}"]) for lv in KERALA_LEVELS}
    norm = {"lp": pd.Series(30.0, index=k.index),
            "up": ((30 * p[5] + 35 * (p[6] + p[7])) / pup["up"]).where(pup["up"] > 0, 35.0),
            "hs": ((35 * p[8] + 45 * (p[9] + p[10])) / pup["hs"]).where(pup["hs"] > 0, 45.0)}
    missing = sum((pup[lv] > 0) & (tch[lv] <= 0) for lv in KERALA_LEVELS) > 0
    total = sum(pup.values())
    pool = sum(tch.values())
    pool = pool.where(pool > 0, _n(k["teachers"])).where(lambda x: x > 0, _n(k["classrooms"]))
    rows = []
    for lv in KERALA_LEVELS:
        split = pool * (pup[lv] / total.where(total > 0))
        cap = np.where(missing, split.fillna(0) * norm[lv], tch[lv] * norm[lv])
        src = np.where(missing, "split_by_pupils", "teachers_by_level")
        rows.append(pd.DataFrame({"school_id": k["school_code_num"], "level": lv, "cap_main": cap,
                                  "cap_enrol": pup[lv], "cap_source": src, "has_level": k[f"has_{lv}"].fillna(False)}))
    hss_b, vhse_b = _n(k["hss_batches"]), _n(k["vhse_batches"])
    rows.append(pd.DataFrame({"school_id": k["school_code_num"], "level": "hss",
                              "cap_main": (hss_b * HSS_SEATS + vhse_b * VHSE_SEATS) * 2,
                              "cap_hss60": (hss_b * HSS_SEATS_HIGH + vhse_b * VHSE_SEATS) * 2,
                              # no VHSE pupil counts are published: VHSE seats stand in for its pupils
                              "cap_enrol": _n(k["sam_students_hss_total"]) + vhse_b * VHSE_SEATS * 2,
                              "cap_source": "hscap_vhscap_batches", "has_level": k["has_hss"].fillna(False) | (vhse_b > 0)}))
    out = pd.concat(rows, ignore_index=True)
    out = out[(out["cap_main"] > 0) | (out["cap_enrol"] > 0)]
    xy = k.set_index("school_code_num")
    out["lat"], out["lon"] = out["school_id"].map(xy["lat"]).to_numpy(), out["school_id"].map(xy["lon"]).to_numpy()
    hilly = out["school_id"].map(xy["sam_is_hilly"]).eq(True)
    out["catchment_m"] = np.where(out["level"].eq("hs") & hilly, CATCHMENT_M["hs_hilly"], out["level"].map(CATCHMENT_M))
    out["district"] = out["school_id"].map(xy["district"]).to_numpy()
    return out.reset_index(drop=True)


def india_supply(s):
    """s: silver schools_india (pandas). One row per school and national level it teaches."""
    s = s[s["lat"].notna()].copy()
    pup = {lv: sum(_n(s[f"{ROMAN[c - 1]}_students"]) for c in range(a, b + 1)) for lv, (a, b, _) in INDIA_LEVELS.items()}
    has = {lv: s[f"has_{lv}"].fillna(False).astype(bool) for lv in INDIA_LEVELS}
    total = sum(pup[lv].where(has[lv], 0) for lv in INDIA_LEVELS)
    n_levels = sum(has[lv].astype(int) for lv in INDIA_LEVELS)
    t, rooms = _n(s["total_teachers"]), _n(s["class_rooms"])
    staff = t.where(t > 0, rooms)
    src = np.select([t > 0, rooms > 0], ["teachers", "classrooms"], "enrolment")
    rows = []
    for lv, (_, _, ptr) in INDIA_LEVELS.items():
        share = (pup[lv] / total.where(total > 0)).fillna(1 / n_levels.where(n_levels > 0))
        cap = np.where(src == "enrolment", pup[lv], staff * share * ptr)
        rows.append(pd.DataFrame({"school_id": s["udise_code"], "level": lv, "cap_main": cap, "cap_enrol": pup[lv],
                                  "cap_source": src, "lat": s["lat"], "lon": s["lon"], "state_name": s["state_name"],
                                  "district": s["district_name"], "location_quality": s["location_quality"]})[has[lv].to_numpy()])
    out = pd.concat(rows, ignore_index=True)
    out["catchment_m"] = out["level"].map(CATCHMENT_M)
    return out[(out["cap_main"] > 0) | (out["cap_enrol"] > 0)].reset_index(drop=True)


def gov_aided_share(udise_csv, levels):
    """Government + Aided share of pupils per state and level (UDISE 2022, all managements). levels: name -> (from, to)."""
    cols = ["state_name", "management"] + [f"{r}_students" for r in ROMAN]
    u = pd.read_csv(udise_csv, usecols=cols, dtype={"state_name": str, "management": str})
    ga = u["management"].isin(GOVT_AIDED)
    rows = []
    for lv, (a, b) in levels.items():
        p = sum(_n(u[f"{ROMAN[c - 1]}_students"]) for c in range(a, b + 1))
        g = pd.DataFrame({"state_name": u["state_name"], "pupils_all": p, "pupils_ga": p.where(ga, 0)}).groupby("state_name").sum()
        rows.append(g.assign(level=lv, share=g["pupils_ga"] / g["pupils_all"]).reset_index())
    return pd.concat(rows, ignore_index=True)


def demand_sql(level):
    """Children of the level's ages from the WorldPop 5-year bands (ages spread evenly inside a band)."""
    return {"lp": "0.8 * pop_5_9",                                   # Kerala Std 1-4: ages 6-9
            "up": "0.6 * pop_10_14",                                 # Std 5-7: ages 10-12
            "hs": "0.4 * pop_10_14 + 0.2 * pop_15_19",               # Std 8-10: ages 13-15
            "hss": "0.4 * pop_15_19",                                # +1/+2: ages 16-17
            "primary": "0.8 * pop_5_9 + 0.2 * pop_10_14",            # India classes 1-5: ages 6-10
            "upper_primary": "0.6 * pop_10_14",                      # 6-8: ages 11-13
            "secondary": "0.2 * pop_10_14 + 0.2 * pop_15_19",        # 9-10: ages 14-15
            "higher_secondary": "0.4 * pop_15_19"}[level]            # 11-12: ages 16-17


def gaussian(d_col, d0_col):
    """Dai (2010) Gaussian decay, SQL: 1 at the school, 0 at the catchment edge d0."""
    return f"(exp(-0.5 * pow({d_col} / {d0_col}, 2)) - exp(-0.5)) / (1 - exp(-0.5))"


if __name__ == "__main__":  # self-check on two made-up schools
    k = pd.DataFrame({"school_code_num": ["1", "2"], "lat": [10.0, 10.1], "lon": [76.0, 76.1], "district": ["X", "X"],
                      "sam_is_hilly": [True, None], "teachers": [9, 4], "classrooms": [9, 4],
                      "teachers_lp": [4, 3], "teachers_up": [3, 0], "teachers_hs": [2, 0],
                      "hss_batches": [2, None], "vhse_batches": [1, None], "sam_students_hss_total": [110, 0],
                      "has_lp": [True, True], "has_up": [True, True], "has_hs": [True, False], "has_hss": [True, False],
                      **{f"sam_students_std{s:02d}": [30, 20 if s <= 5 else 0] for s in range(1, 11)}})
    o = kerala_supply(k).set_index(["school_id", "level"])
    assert o.loc[("1", "lp"), "cap_main"] == 120 and o.loc[("1", "lp"), "cap_source"] == "teachers_by_level"
    assert abs(o.loc[("1", "up"), "cap_main"] - 3 * (30 * 30 + 35 * 60) / 90) < 1e-9
    assert abs(o.loc[("1", "hs"), "cap_main"] - 2 * (35 * 30 + 45 * 60) / 90) < 1e-9 and o.loc[("1", "hs"), "catchment_m"] == 3000
    assert o.loc[("1", "hss"), "cap_main"] == (2 * 50 + 30) * 2 and o.loc[("1", "hss"), "cap_hss60"] == (2 * 60 + 30) * 2
    # school 2: UP pupils (Std 5) but no UP teacher -> its 3 LP teachers are shared 80/20 between LP and UP
    assert o.loc[("2", "lp"), "cap_source"] == "split_by_pupils" and abs(o.loc[("2", "lp"), "cap_main"] - 3 * 0.8 * 30) < 1e-9
    assert abs(o.loc[("2", "up"), "cap_main"] - 3 * 0.2 * 30) < 1e-9 and ("2", "hss") not in o.index
    print("ok")
