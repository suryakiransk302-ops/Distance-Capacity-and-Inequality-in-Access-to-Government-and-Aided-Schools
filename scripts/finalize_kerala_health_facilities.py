"""Build the final Kerala public health facility layer after the manual review.

Input:  data/processed/health/kerala_health_facilities.csv (build_kerala_health_facilities.py)
Review (data/raw/health_manual/), one decision per reviewed institution:
  manual      point located by the researcher in Google Maps (manual_locations_google_maps_*.csv),
              checked against the DHS 2024-25 point and OSM
  osm         OSM point of the named facility (osm_kerala_health_*.json), chosen by the researcher
              where the manual point was doubtful or missing
  dhs         DHS 2024-25 point kept (no better source)
  unresolved  no trustworthy point -> no location
  exclude     not a modern-medicine facility -> dropped
Institutions not reviewed keep their location from the builder. Sub centres placed at their parent
facility are re-placed at the parent's reviewed point, and a sub centre whose 2016 directory point lies
> 1 km outside its district is moved to its parent facility (location_quality "proxy").

in_access_analysis = False for mobile units, TB centres and the leprosy sanatorium (disease-programme
units, not general care) and for facilities without a location.

Outputs: FINAL COMPLETE DATASET/health/kerala_health_facilities_final.{csv,gpkg}
Usage:   python scripts/finalize_kerala_health_facilities.py
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from build_india_health_facilities import haversine_m
from build_kerala_health_facilities import ADM2, KERALA_DISTRICTS, OSM_JSON, OUTSIDE_KM, district_name, parent_key

ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "data" / "processed" / "health" / "kerala_health_facilities.csv"
MANUAL_DIR = ROOT / "data" / "raw" / "health_manual"
OUT_DIR = ROOT / "FINAL COMPLETE DATASET" / "health"
NOT_GENERAL_CARE = r"(?i)\bTB Centre\b|\bTBC\b|\bDTBC\b|Leprosy Sanitorium"
LOCATION_SOURCES = {"manual": "manual_google_maps", "osm": "OSM_reviewed", "unresolved": "none"}


def osm_points():
    els = json.loads(OSM_JSON.read_text(encoding="utf-8"))["elements"]
    return {f"{e['type']}/{e['id']}": (e.get("lat", e.get("center", {}).get("lat")),
                                       e.get("lon", e.get("center", {}).get("lon")), e.get("tags", {}).get("name"))
            for e in els}


def km_outside(g, districts):
    pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), index=g.index, crs=4326).to_crs(32643)
    return [round(p.distance(districts.geometry[dn]) / 1000, 2) if has else np.nan
            for p, dn, has in zip(pts, g["district"], g["lat"].notna())]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = pd.read_csv(BUILT, low_memory=False, dtype={"nin": str})
    manual = pd.concat([pd.read_csv(f) for f in sorted(MANUAL_DIR.glob("manual_locations_google_maps_*.csv"))])
    manual = manual.drop_duplicates("facility_id", keep="last").set_index("facility_id")
    dec = pd.concat([pd.read_csv(f) for f in sorted(MANUAL_DIR.glob("location_decisions_*.csv"))])
    dec = dec.drop_duplicates("facility_id", keep="last").set_index("facility_id")

    # facility_id is positional in the builder output: make sure it still points at the same facility
    chk = manual.join(g.set_index("facility_id")[["district"]], rsuffix="_built")
    if chk["district"].ne(chk["district_built"]).any() or not dec.index.isin(g["facility_id"]).all():
        raise SystemExit(f"Review files do not match the facility list:\n{chk[chk['district'].ne(chk['district_built'])]}")

    g["qa_flags_before_review"] = g["qa_flags"]
    g["review_decision"] = g["facility_id"].map(dec["decision"]).fillna("")
    g["review_note"] = g["facility_id"].map(dec["reason"]).fillna("")
    g = g[g["review_decision"].ne("exclude")].copy()

    osm = osm_points()
    is_man, is_osm = g["review_decision"].eq("manual"), g["review_decision"].eq("osm")
    oid = g.loc[is_osm, "facility_id"].map(dec["osm_id"])
    missing = oid[~oid.isin(osm.keys())]
    if len(missing):
        raise SystemExit(f"OSM ids not in {OSM_JSON.name}: {missing.tolist()}")
    g["reviewed_osm_name"] = oid.map(lambda i: osm[i][2])
    g["pre_review_lat"], g["pre_review_lon"] = g["lat"], g["lon"]
    g.loc[is_man, "lat"] = g.loc[is_man, "facility_id"].map(manual["lat"])
    g.loc[is_man, "lon"] = g.loc[is_man, "facility_id"].map(manual["lon"])
    g.loc[is_osm, "lat"] = oid.map(lambda i: osm[i][0])
    g.loc[is_osm, "lon"] = oid.map(lambda i: osm[i][1])
    g.loc[g["review_decision"].eq("unresolved"), ["lat", "lon"]] = np.nan
    g["location_from"] = g["review_decision"].map(LOCATION_SOURCES).fillna(g["location_from"])
    d = gpd.read_file(ADM2)[["shapeName", "geometry"]].cx[74.5:77.8, 7.9:13.1]
    d["shapeName"] = d["shapeName"].map(district_name)
    d = d[d["shapeName"].isin(KERALA_DISTRICTS)].dissolve("shapeName").to_crs(32643)

    # Sub centres placed at their parent facility follow the parent's reviewed point; a sub centre whose
    # 2016 directory point falls outside its district is moved to the parent as well.
    inst = g[g["std_type"].ne("SC") & g["lat"].notna()].copy()
    parents = inst.assign(pk=inst["name"].map(parent_key)).drop_duplicates(["district", "pk"]).set_index(["district", "pk"])
    sc = g["std_type"].eq("SC")
    key = pd.MultiIndex.from_arrays([g.loc[sc, "district"], g.loc[sc, "parent_facility"].map(parent_key)])
    g.loc[sc, "parent_lat"] = parents["lat"].reindex(key).to_numpy()
    g.loc[sc, "parent_lon"] = parents["lon"].reindex(key).to_numpy()
    proxy = g["location_from"].eq("parent_facility_proxy")
    g.loc[proxy, ["lat", "lon"]] = g.loc[proxy, ["parent_lat", "parent_lon"]].to_numpy()
    g["km_outside_district"] = km_outside(g, d)
    stray = sc & g["km_outside_district"].gt(OUTSIDE_KM) & g["location_from"].ne("parent_facility_proxy")
    g.loc[stray, ["lat", "lon"]] = g.loc[stray, ["parent_lat", "parent_lon"]].to_numpy()
    g.loc[stray, "location_from"] = np.where(g.loc[stray, "lat"].notna(), "parent_facility_proxy", "none")
    g.loc[stray, "review_note"] = "2016 directory point was outside the district; placed at parent facility"
    g["km_outside_district"] = km_outside(g, d)
    g["review_moved_m"] = haversine_m(g["lat"], g["lon"], g["pre_review_lat"], g["pre_review_lon"]).round(0)
    gg = gpd.GeoDataFrame(g.drop(columns=["parent_lat", "parent_lon"]),
                          geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs=4326)

    reviewed = gg["review_decision"].isin(["manual", "osm", "dhs"])
    gg["location_quality"] = np.select(
        [gg["lat"].isna(), reviewed & gg["review_decision"].ne("dhs"), reviewed,
         gg["location_from"].eq("parent_facility_proxy")], ["none", "verified", "check", "proxy"],
        gg["location_quality"])
    gg["in_access_analysis"] = (gg["lat"].notna() & gg["std_type"].ne("MOB")
                                & ~gg["name"].str.contains(NOT_GENERAL_CARE, regex=True))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gg.drop(columns="geometry").to_csv(OUT_DIR / "kerala_health_facilities_final.csv", index=False, encoding="utf-8")
    gg[gg["lat"].notna()].to_file(OUT_DIR / "kerala_health_facilities_final.gpkg", layer="health_facilities",
                                  driver="GPKG")

    inst = gg[gg["std_type"].ne("SC")]
    print("facilities:", len(gg), "| institutions:", len(inst), "| sub centres:", int(gg["std_type"].eq("SC").sum()))
    print("review decisions:", gg["review_decision"].replace("", "not reviewed").value_counts().to_dict())
    print("institution location quality:", inst["location_quality"].value_counts().to_dict())
    print("institutions without location:", inst.loc[inst["lat"].isna(), "name"].tolist())
    print("in access analysis:", int(gg["in_access_analysis"].sum()), "| excluded:",
          gg.loc[~gg["in_access_analysis"]].groupby("std_type").size().to_dict())
    out = inst[inst["km_outside_district"].gt(OUTSIDE_KM)]
    print(f"institutions > {OUTSIDE_KM} km outside listed district:",
          out[["facility_id", "name", "district", "km_outside_district", "review_decision"]].to_string(index=False))
    print("written:", OUT_DIR / "kerala_health_facilities_final.csv")


if __name__ == "__main__":
    main()
