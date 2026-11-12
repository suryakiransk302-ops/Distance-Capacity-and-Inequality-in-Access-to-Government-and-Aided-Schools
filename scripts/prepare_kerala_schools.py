"""Prepare Kerala Government + Aided schools (KITE Sametham) for accessibility analysis.

Adds, from the class range in `level` (e.g. "5 - 12"):
  min_class, max_class
  has_lp   teaches all of classes 1-4   (Lower Primary)
  has_up   teaches all of classes 5-7   (Upper Primary)
  has_hs   teaches all of classes 8-10  (High School)
  has_hss  teaches classes 11-12        (Higher Secondary)
  sections e.g. "LP+UP+HS"
and keeps Sametham's own category (school_code prefix LP / UP / HS = highest section).
A section counts only if the school teaches the whole of it: an LP school with class 5
(level "1 - 5") is LP only, because a child in class 6 or 7 cannot attend it.

Input:  data/raw/sametham/sametham_kerala_schools_govt_aided_2026-10-26.csv
Output: data/processed/schools/kerala_schools_govt_aided.{csv,gpkg}  (gpkg: schools with coordinates)
Usage:  python scripts/prepare_kerala_schools.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "sametham" / "sametham_kerala_schools_govt_aided_2026-10-26.csv"
OUT_DIR = ROOT / "data" / "processed" / "schools"
SECTIONS = {"lp": (1, 4), "up": (5, 7), "hs": (8, 10), "hss": (11, 12)}


def main():
    df = pd.read_csv(SRC, dtype={"school_code": str, "phone": str})
    rng = df["level"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    df["min_class"], df["max_class"] = rng[0].astype("Int64"), rng[1].astype("Int64")
    for name, (lo, hi) in SECTIONS.items():
        df[f"has_{name}"] = (df["min_class"] <= lo) & (df["max_class"] >= hi)
    df["sections"] = df[[f"has_{s}" for s in SECTIONS]].apply(
        lambda r: "+".join(s.upper() for s in SECTIONS if r[f"has_{s}"]), axis=1)
    df["sametham_category"] = df["school_code"].str.extract(r"^([A-Z]+)\s*:")[0]
    df["school_code_num"] = df["school_code"].str.extract(r"(\d+)")[0]
    df["has_coords"] = df["coord_check"].eq("ok")
    df["source"] = "KITE Sametham (https://sametham.kite.kerala.gov.in/), retrieved 2026-10-26"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "kerala_schools_govt_aided.csv", index=False, encoding="utf-8")
    pts = df[df["has_coords"]]
    gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326").to_file(
        OUT_DIR / "kerala_schools_govt_aided.gpkg", layer="schools", driver="GPKG")

    print("schools:", len(df), "| with coordinates:", int(df["has_coords"].sum()),
          "| unparsed level:", int(df["min_class"].isna().sum()))
    print("\nsections combination:")
    print(df.groupby("sections").agg(schools=("school_code", "size"), with_coords=("has_coords", "sum"))
          .sort_values("schools", ascending=False).to_string())
    print("\nschools offering each section (any school that teaches it):")
    for s in SECTIONS:
        col = df[f"has_{s}"]
        print(f"  {s.upper():3s}: {int(col.sum()):6d} | with coordinates: {int((col & df['has_coords']).sum())}")
    print("\nwritten:", OUT_DIR)


if __name__ == "__main__":
    main()
