"""Final all-India public health facility layer used in the analysis.

All states except Kerala: the NIN + All India Health Centres Directory merge
  (data/processed/health/india_health_facilities.csv, scripts/build_india_health_facilities.py).
Kerala: replaced by the reviewed DHS Kerala 2024-25 layer
  (FINAL COMPLETE DATASET/health/kerala_health_facilities_final.csv, scripts/finalize_kerala_health_facilities.py).
  Kerala is thin in NIN (2 rows) and its 2016 directory points are ~1 km off (median vs OSM), while the
  reviewed DHS layer is within tens of metres. Kerala rows keep their KL_HF_ ids.

in_access_analysis = False for facilities without a location, mobile units, AYUSH dispensaries
(study covers modern medicine) and Kerala's TB centres / leprosy sanatorium.
location_quality is kept so placeholder ("poor") points can be dropped in a sensitivity run.

Outputs: FINAL COMPLETE DATASET/health/india_health_facilities_final.{csv,gpkg}
Usage:   python scripts/build_india_health_final.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from build_india_health_facilities import OUTSIDE_STATE_KM, TYPE_NAMES, spatial_checks

ROOT = Path(__file__).resolve().parents[1]
INDIA = ROOT / "data" / "processed" / "health" / "india_health_facilities.csv"
KERALA = ROOT / "FINAL COMPLETE DATASET" / "health" / "kerala_health_facilities_final.csv"
OUT_DIR = ROOT / "FINAL COMPLETE DATASET" / "health"
TYPE_NAMES = TYPE_NAMES | {"SPEC": "Speciality centre (TB, mental health, leprosy)", "MOB": "Mobile unit"}
SOURCE_YEAR = {"NIN+HCD": "2021", "NIN": "2021", "HCD": "2016", "DHS_KERALA": "2024-25"}


def kerala_rows():
    k = pd.read_csv(KERALA, low_memory=False, dtype={"nin": str})
    return pd.DataFrame({
        "facility_id": k["facility_id"], "name": k["name"], "std_type": k["std_type"],
        "type_original": k["section"].fillna("Sub Centre"), "state": "Kerala", "state_key": "kerala",
        "district": k["district"], "subdistrict": k["taluk"], "address": k["address"],
        "pin_code": pd.to_numeric(k["address"].str.extract(r"(6\d{5})", expand=False), errors="coerce"),
        "location_type": k["rural_urban"].str.upper().map({"R": "Rural", "U": "Urban"}),
        "nin": k["nin"], "lat": k["lat"], "lon": k["lon"], "location_from": k["location_from"],
        "location_quality": k["location_quality"], "qa_flags": k["qa_flags"].fillna(""), "sources": "DHS_KERALA",
        "in_access_analysis": k["in_access_analysis"], "review_decision": k["review_decision"],
        "health_block": k["health_block"], "parent_facility": k["parent_facility"]})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    india = pd.read_csv(INDIA, low_memory=False, dtype={"nin": str, "pin_code": str})
    rest = india[india["state"].ne("Kerala")].copy()
    ayush = rest["type_original"].fillna("").str.contains("Ayush", case=False)
    rest["in_access_analysis"] = rest["lat"].notna() & ~ayush
    print("India merge:", len(india), "| Kerala rows replaced:", int(india["state"].eq("Kerala").sum()),
          "| AYUSH dispensaries kept but excluded from analysis:", int(ayush.sum()))

    k = spatial_checks(kerala_rows())
    k["qa_flags"] = (k["qa_flags"] + np.where(k["km_outside_state"].gt(OUTSIDE_STATE_KM), " outside_listed_state", "")
                     ).str.strip()
    k = pd.DataFrame(k.drop(columns=["geometry", "state_key"]))

    out = pd.concat([rest, k], ignore_index=True)
    out["pin_code"] = pd.to_numeric(out["pin_code"], errors="coerce").astype("Int64")
    out["type_name"] = out["std_type"].map(TYPE_NAMES)
    out["source_year"] = out["sources"].map(SOURCE_YEAR)
    out["in_access_analysis"] = out["in_access_analysis"].astype(bool)
    out = out.sort_values(["state", "district", "std_type", "name"]).reset_index(drop=True)

    g = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["lon"], out["lat"]), crs=4326)
    g.drop(columns="geometry").to_csv(OUT_DIR / "india_health_facilities_final.csv", index=False, encoding="utf-8")
    g[g["lat"].notna()].to_file(OUT_DIR / "india_health_facilities_final.gpkg", layer="health_facilities",
                                driver="GPKG")

    a = out[out["in_access_analysis"]]
    print("facilities:", len(out), "| in access analysis:", len(a), "| states:", out["state"].nunique())
    print("sources:", out["sources"].value_counts().to_dict())
    print("location quality (all):", out["location_quality"].value_counts().to_dict())
    print("location quality (analysis):", a["location_quality"].value_counts().to_dict())
    print("Kerala:", k["std_type"].value_counts().to_dict(), "| outside state:",
          int(k["km_outside_state"].gt(OUTSIDE_STATE_KM).sum()))
    print("\nby type (analysis set):\n", a["type_name"].value_counts().to_string())
    print("\nwritten:", OUT_DIR / "india_health_facilities_final.csv")


if __name__ == "__main__":
    main()
