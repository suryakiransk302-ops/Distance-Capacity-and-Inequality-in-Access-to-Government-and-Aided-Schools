"""Extract the Kerala DHS "List of Modern Medicine Institutions 2024-25" (PDF) into tables.

Source:  Directorate of Health Services, Kerala, Health Information Cell (Statistics Division)
         data/raw/health/List-of-Modern-Medicine-Institutions-2024-25.pdf
Sections used:
  hospitals and health centres (DH, GH, W&C, THQH, TH, CHC, PHC, FHC, TB, mental health,
  leprosy, health clinics, dispensaries, mobile units, speciality): name, address,
  local body, health block, taluk, latitude, longitude, NIN
  sub centres: name, address, parent CHC/PHC/FHC, health block, NIN, population served (no coordinates)
  PP units (post-partum units inside hospitals) and non-care offices (nursing schools, labs,
  training centres, vector control units, DHS offices) are not facilities people visit for care
  and are skipped.
The PHC and FHC sections list the same institutions twice (PHCs converted to Family Health
Centres appear in both), so institutions are de-duplicated on NIN, else on name + coordinates.

Outputs: data/interim/health/kerala_dhs_institutions_2024_25.csv
         data/interim/health/kerala_dhs_subcentres_2024_25.csv
Usage:   python scripts/extract_kerala_dhs_institutions.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "health" / "List-of-Modern-Medicine-Institutions-2024-25.pdf"
OUT_DIR = ROOT / "data" / "interim" / "health"
SOURCE = "DHS Kerala, List of Modern Medicine Institutions 2024-25"
SKIP = re.compile(r"school of nursing|training|public health lab|vector control|other institutions|pp units|abstract|"
                  r"sl no|district wise|medical institutions and sanctioned|^district$", re.I)
# header text (spaces and line breaks removed, lower case) -> column
INST_COLS = [("sl.no", "sl_no"), ("district", "district_code"), ("category(includingfhc)", "category_incl_fhc"),
             ("centralcategor", "central_category"), ("category", "category"), ("fhcphase", "fhc_phase"),
             ("nameoftheinstitution", "name"), ("completeaddress", "address"), ("phone", "phone"),
             ("functionalbed", "functional_beds"), ("rural/urban", "rural_urban"), ("nameofthecorp", "local_body"),
             ("(c/m/p)", "local_body_type"), ("nameofhealthblock", "health_block"),
             ("nameofblockpanchayath", "block_panchayat"), ("nameoftaluk", "taluk"), ("assembly", "assembly"),
             ("email", "email"), ("lattitude", "lat"), ("longitude", "lon"), ("nin", "nin"),
             ("typeofbuilding", "building")]
SC_COLS = [("districtslno", "sl_no"), ("district", "district"), ("nameofhealthsubcentre", "name"),
           ("completeaddress", "address"), ("nameoffacility", "parent_facility"), ("typeoffacility", "parent_type"),
           ("rural/urban", "rural_urban"), ("healthblock", "health_block"), ("ninnumber", "nin"),
           ("nameoflocalself", "local_body"), ("typeofbuilding", "building"), ("totalpopulation", "population_served")]
DISTRICTS = {"TVM": "Thiruvananthapuram", "KLM": "Kollam", "PTA": "Pathanamthitta", "ALP": "Alappuzha",
             "KTM": "Kottayam", "IDK": "Idukki", "EKM": "Ernakulam", "TSR": "Thrissur", "PKD": "Palakkad",
             "PLKD": "Palakkad", "MLP": "Malappuram", "KKD": "Kozhikode", "WYD": "Wayanad", "KNR": "Kannur",
             "KSD": "Kasaragod"}


def clean(c):
    return re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()


def map_header(header, spec):
    cols = []
    for h in header:
        k = re.sub(r"\s+", "", (h or "").lower())
        cols.append(next((col for key, col in spec if k.startswith(key) or key in k), None))
    return cols


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Jupyter's output stream has no reconfigure()
        sys.stdout.reconfigure(encoding="utf-8")
    doc = pymupdf.open(PDF)
    inst, sc = [], []
    for pn, page in enumerate(doc, 1):
        if pn < 9:  # contents and abstracts
            continue
        for t in page.find_tables().tables:
            rows = t.extract()
            hdr = next((i for i, r in enumerate(rows[:5]) if any(
                c and re.search(r"Name of the Institution|Name of Health Sub Centre|Facility Name", c) for c in r)), None)
            if hdr is None:
                print(f"page {pn}: table without header skipped ({len(rows)} rows)")
                continue
            section = next((clean(r[0]) for r in rows[:hdr] if r[0] and "List of Modern" not in r[0]), "")
            if SKIP.search(section):
                continue
            is_sc = section.lower().startswith("subcentre")
            names = map_header(rows[hdr], SC_COLS if is_sc else INST_COLS)
            for r in rows[hdr + 1:]:
                rec = {c: clean(v) for c, v in zip(names, r) if c}
                if not rec.get("name") or not re.match(r"\d", rec.get("sl_no", "")):
                    continue
                rec |= {"section": section, "page": pn}
                (sc if is_sc else inst).append(rec)

    i = pd.DataFrame(inst)
    i["district"] = i["district_code"].str.upper().map(DISTRICTS)
    i["lat_as_printed"], i["lon_as_printed"] = i["lat"], i["lon"]
    for c in ["lat", "lon"]:
        i[c] = pd.to_numeric(i[c].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    swap = i["lat"].gt(70) & i["lon"].lt(15)
    i.loc[swap, ["lat", "lon"]] = i.loc[swap, ["lon", "lat"]].to_numpy()
    # longitude printed without its decimal point (e.g. 765587900 -> 76.5587900)
    nodot = i["lon"].gt(1000)
    digits = i.loc[nodot, "lon_as_printed"].str.replace(r"\D", "", regex=True)
    i.loc[nodot, "lon"] = pd.to_numeric(digits.str[:2] + "." + digits.str[2:])
    i["coord_note"] = np.where(nodot, "longitude decimal point restored", "")
    bad = i["lat"].notna() & ~(i["lat"].between(8.1, 12.9) & i["lon"].between(74.8, 77.5))
    i.loc[bad, "coord_note"] = "coordinates unusable as printed"
    i.loc[bad, ["lat", "lon"]] = None
    i["nin"] = i["nin"].str.replace(r"\D", "", regex=True).replace("", None)
    i["key"] = i["nin"].fillna(i["name"].str.lower() + "|" + i["lat"].round(4).astype(str) + "|" + i["lon"].round(4).astype(str))
    i["listed_in_sections"] = i.groupby("key")["section"].transform(lambda s: "; ".join(dict.fromkeys(s)))
    # keep the record from the facility's own section (PHC / CHC list), not the FHC list
    i["fhc_list"] = i["section"].str.contains("Family Health", case=False)
    i = i.sort_values(["fhc_list", "page"]).drop_duplicates("key").drop(columns=["key", "fhc_list"])
    i["source"] = SOURCE

    s = pd.DataFrame(sc)
    s["district"] = s["district"].str.title().replace({"Trivandrum": "Thiruvananthapuram", "Kozhikkode": "Kozhikode",
                                                       "Kasargod": "Kasaragod", "Palghat": "Palakkad"})
    s["nin"] = s["nin"].str.replace(r"\D", "", regex=True).replace("", None)
    s["source"] = SOURCE

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    i.to_csv(OUT_DIR / "kerala_dhs_institutions_2024_25.csv", index=False, encoding="utf-8")
    s.to_csv(OUT_DIR / "kerala_dhs_subcentres_2024_25.csv", index=False, encoding="utf-8")
    print("institutions:", len(i), "| with lat/lon:", int(i["lat"].notna().sum()), "| swapped fixed:", int(swap.sum()),
          "| with NIN:", int(i["nin"].notna().sum()))
    print("coordinate notes:", i["coord_note"].value_counts().to_dict())
    print(i.loc[i["coord_note"].ne(""), ["district", "name", "lat_as_printed", "lon_as_printed", "lat", "lon",
                                         "coord_note"]].to_string())
    print(i["section"].value_counts().to_string())
    print("by district:", i["district"].value_counts(dropna=False).to_dict())
    print("\nsub centres:", len(s), "| with NIN:", int(s["nin"].notna().sum()))
    print("by district:", s["district"].value_counts(dropna=False).to_dict())
    print("written:", OUT_DIR)


if __name__ == "__main__":
    main()
