"""Fill and check coordinates for Kerala Government + Aided schools.

Location per school, in order of trust (measured against OpenStreetMap for the same schools:
Sametham median 19 m, UDISE median 82 m with 27% over 1 km):
  sametham        coordinate published on KITE Sametham
  osm             OSM school with the same name, type (LP/UP/HS/HSS) and management in the same
                  district, and the only such candidate; agrees with UDISE when both exist
  osm_unconfirmed same, but UDISE puts the school more than AGREE_M away   -> needs review
  udise           UDISE "Basic Details of Schools" (retrieved 2022-01-12), linked by the UDISE
                  code from the school's Sametham page                    -> lower accuracy;
                  rejected (school stays missing, see `note`) if > 1 km outside its listed district
  missing         none of the above

Checks: each point lies in its listed district; Sametham points are compared with their
strict OSM match where one exists (gap > REVIEW_M flagged for review).

Inputs:  data/processed/schools/kerala_schools_govt_aided.csv
         data/interim/schools/sametham_school_details.csv   (scripts/fetch_sametham_school_details.py)
         data/raw/udise_basic_details/basic-details-of-schools.csv
         data/raw/osm/osm_kerala_schools_2026-10-28.json
Outputs: data/processed/schools/kerala_schools_govt_aided_located.{csv,gpkg}
         data/interim/schools/schools_to_review.csv
Usage:   python scripts/fill_school_coordinates.py
"""

import difflib
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHOOLS = ROOT / "data" / "processed" / "schools" / "kerala_schools_govt_aided.csv"
DETAILS = ROOT / "data" / "interim" / "schools" / "sametham_school_details.csv"
UDISE = ROOT / "data" / "raw" / "udise_basic_details" / "basic-details-of-schools.csv"
OSM = ROOT / "data" / "raw" / "osm" / "osm_kerala_schools_2026-10-28.json"
DISTRICTS = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM2.geojson"
OUT_DIR = ROOT / "data" / "processed" / "schools"
REVIEW_CSV = ROOT / "data" / "interim" / "schools" / "schools_to_review.csv"
METRIC_CRS = "EPSG:32643"
NAME_MIN = 0.9
AGREE_M = 250
REVIEW_M = 250

GENERIC = (r"\b(govt|government|gov|g|aided|a|m|school|schools|higher|secondary|high|lower|upper|primary|"
           r"vocational|lps|ups|hs|hss|vhss|ghs|ghss|glps|gups|gvhss|amlps|amups|alps|aups|lp|up|l|p|u|s|h|v|"
           r"english|medium|em|the|and|for|girls|boys|mixed|model|st|saint|sree|sri|of)\b")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(s).lower().replace(".", "")))


def core(s):
    """Name without generic words: usually the place or founder name."""
    return re.sub(r"\s+", "", re.sub(GENERIC, " ", norm(s)))


def school_type(s):
    n = norm(s).replace(" ", "")
    if re.search(r"highersecondary|vhss|hss|vhse", n):
        return "HSS"
    if re.search(r"highschool|hs\b|ghs|hs$", n) or re.search(r"\bh s\b", norm(s)):
        return "HS"
    if re.search(r"upperprimary|ups|aup|gup|up\b", n):
        return "UP"
    if re.search(r"lowerprimary|lps|alp|glp|lp\b|lpgs|lpbs", n):
        return "LP"
    return None


def is_govt(s):
    return bool(re.match(r"^(govt|government|gov|g\b|gvhss|ghss|ghs|glps|gups|gwlps|gwups|gmlps|gmups)", norm(s)))


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(h))


def load_osm(districts):
    els = json.loads(OSM.read_text(encoding="utf-8"))["elements"]
    o = pd.DataFrame([{"osm_id": f"{e['type']}/{e['id']}", "osm_name": e.get("tags", {}).get("name", ""),
                       "osm_lat": e.get("lat", e.get("center", {}).get("lat")),
                       "osm_lon": e.get("lon", e.get("center", {}).get("lon"))} for e in els])
    o = o[o["osm_name"].ne("") & o["osm_lat"].notna()]
    g = gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o["osm_lon"], o["osm_lat"]), crs="EPSG:4326")
    g = gpd.sjoin(g, districts, predicate="within").drop(columns="index_right").rename(columns={"district": "osm_district"})
    g["ocore"], g["otype"], g["ogov"] = g["osm_name"].map(core), g["osm_name"].map(school_type), g["osm_name"].map(is_govt)
    return pd.DataFrame(g.drop(columns="geometry"))


def strict_osm_match(s, osm):
    """One unambiguous OSM school per Sametham school: same district, type, management, close name."""
    c = s.assign(core=s["school_name"].map(core), stype=s["school_name"].map(school_type), gov=s["school_name"].map(is_govt))
    c = c[(c["core"].str.len() >= 4) & c["stype"].notna()]
    p = c[["school_code_num", "district", "core", "stype", "gov"]].merge(osm, left_on="district", right_on="osm_district")
    p = p[(p["stype"] == p["otype"]) & (p["gov"] == p["ogov"])]
    p["osm_name_score"] = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(p["core"], p["ocore"])]
    p = p[p["osm_name_score"] >= NAME_MIN]
    p = p[p.groupby("school_code_num")["osm_id"].transform("size") == 1]
    p = p[~p.duplicated("osm_id", keep=False)]
    return p[["school_code_num", "osm_id", "osm_name", "osm_lat", "osm_lon", "osm_name_score"]]


def main():
    s = pd.read_csv(SCHOOLS, dtype={"school_code_num": str, "phone": str})
    districts = gpd.read_file(DISTRICTS)[["shapeName", "geometry"]].cx[74.5:77.8, 7.9:13.1].rename(columns={"shapeName": "district"})

    det = (pd.read_csv(DETAILS, dtype=str) if DETAILS.exists()
           else pd.DataFrame(columns=["school_code_num", "udise_code", "address", "pin_code"], dtype=str))
    s = s.merge(det[["school_code_num", "udise_code", "address", "pin_code"]], on="school_code_num", how="left")

    u = pd.read_csv(UDISE, usecols=["state_name", "udise_school_code", "school_name", "management", "latitude", "longitude"],
                    dtype=str, keep_default_na=False)
    u = u[u["state_name"].str.strip().str.lower().eq("kerala")]
    u = u.rename(columns={"udise_school_code": "udise_code", "school_name": "udise_name", "management": "udise_management",
                          "latitude": "udise_lat", "longitude": "udise_lon"}).drop(columns="state_name")
    u[["udise_lat", "udise_lon"]] = u[["udise_lat", "udise_lon"]].astype(float)
    s = s.merge(u, on="udise_code", how="left")

    s = s.merge(strict_osm_match(s, load_osm(districts)), on="school_code_num", how="left")

    has_s, has_o, has_u = s["has_coords"].astype(bool), s["osm_lat"].notna(), s["udise_lat"].notna()
    s["osm_vs_udise_m"] = haversine_m(s["osm_lat"], s["osm_lon"], s["udise_lat"], s["udise_lon"]).round(0)
    s["sametham_vs_osm_m"] = haversine_m(s["lat"], s["lon"], s["osm_lat"], s["osm_lon"]).where(has_s).round(0)
    conds = [has_s, has_o & (~has_u | s["osm_vs_udise_m"].le(AGREE_M)), has_o, has_u]
    s["location_source"] = np.select(conds, ["sametham", "osm", "osm_unconfirmed", "udise"], "missing")
    s["final_lat"] = np.select(conds, [s["lat"], s["osm_lat"], s["osm_lat"], s["udise_lat"]], np.nan)
    s["final_lon"] = np.select(conds, [s["lon"], s["osm_lon"], s["osm_lon"], s["udise_lon"]], np.nan)
    s["location_accuracy"] = s["location_source"].map(
        {"sametham": "good", "osm": "good", "osm_unconfirmed": "review", "udise": "lower", "missing": "none"})

    located = s[s["final_lat"].notna()]
    g = gpd.GeoDataFrame(located, geometry=gpd.points_from_xy(located["final_lon"], located["final_lat"]), crs="EPSG:4326")
    g = gpd.sjoin(g, districts.rename(columns={"district": "district_spatial"}), how="left", predicate="within")
    g = g.drop(columns="index_right")
    listed = districts.set_index("district").to_crs(METRIC_CRS).geometry
    listed = listed[~listed.index.duplicated()]
    gm = g.to_crs(METRIC_CRS)
    g["km_outside_district"] = [0.0 if a == b else round(pt.distance(listed.get(b)) / 1000, 2) if b in listed.index else np.nan
                                for pt, a, b in zip(gm.geometry, g["district_spatial"], g["district"])]
    s = s.merge(g[["school_code_num", "km_outside_district"]], on="school_code_num", how="left")

    # A UDISE point outside the school's own district is wrong: drop it rather than keep a bad location.
    reject = s["location_source"].eq("udise") & s["km_outside_district"].gt(1)
    s["note"] = np.where(reject, "UDISE point rejected: " + s["km_outside_district"].astype(str) + " km outside listed district", "")
    s.loc[reject, ["final_lat", "final_lon", "km_outside_district"]] = np.nan
    s.loc[reject, ["location_source", "location_accuracy"]] = ["missing", "none"]

    flags = pd.DataFrame({
        "district": np.where(s["km_outside_district"].gt(1), "outside_listed_district", ""),
        "osm": np.where(s["sametham_vs_osm_m"].gt(REVIEW_M), "sametham_osm_disagree", ""),
        "unconf": np.where(s["location_source"].eq("osm_unconfirmed"), "osm_udise_disagree", ""),
    }, index=s.index)
    s["qa_flags"] = flags.apply(lambda r: " ".join(x for x in r if x), axis=1)
    s["needs_review"] = s["qa_flags"].ne("")

    s["lat"], s["lon"] = s["final_lat"].round(7), s["final_lon"].round(7)
    s = s.drop(columns=["final_lat", "final_lon", "coord_check", "has_coords"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT_DIR / "kerala_schools_govt_aided_located.csv", index=False, encoding="utf-8")
    pts = s[s["lat"].notna()]
    gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326").to_file(
        OUT_DIR / "kerala_schools_govt_aided_located.gpkg", layer="schools", driver="GPKG")
    review = s[s["needs_review"]].sort_values(["qa_flags", "district", "school_name"])
    review.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    print("schools:", len(s), "| UDISE code found:", int(s["udise_code"].notna().sum()),
          "| UDISE row found:", int(s["udise_lat"].notna().sum()))
    print("location source:", s["location_source"].value_counts().to_dict())
    print(f"located: {int(s['lat'].notna().sum())} of {len(s)} ({100 * s['lat'].notna().mean():.1f}%)")
    t = s.groupby("district").agg(schools=("school_code_num", "size"), located=("lat", lambda x: int(x.notna().sum())))
    t["pct"] = (100 * t["located"] / t["schools"]).round(1)
    print(t.sort_values("pct").to_string())
    for sec in ["lp", "up", "hs", "hss"]:
        m = s[f"has_{sec}"].astype(bool)
        print(f"  {sec.upper():3s} located {int(s.loc[m, 'lat'].notna().sum())} of {int(m.sum())}")
    print("needs review:", int(s["needs_review"].sum()), s.loc[s["needs_review"], "qa_flags"].value_counts().to_dict())
    print("written:", OUT_DIR, "and", REVIEW_CSV)


if __name__ == "__main__":
    main()
