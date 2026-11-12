"""All-India Government + Aided schools with automatically corrected coordinates (plan section B).

Input: UDISE "Basic Details of Schools" (India Data Portal, retrieved 2022-01-12), 1,372,195 schools.
Kept: Government (Department of Education, local body, tribal / social welfare, other State and
Central Government schools) and Aided (Government Aided, recognised Madarsa) = about 1.07 M schools.

Coordinate correction, no manual checking. Each school gets one point, in order of trust:
  osm             OSM school (amenity=school, PROJECTS IPYNB/10) with nearly the same name (rapidfuzz >= 90,
                  unique) within 3 km of the school's anchor (its UDISE point, or its peers' median when the
                  UDISE point is bad)
  village_median  bad UDISE point -> median of the good points of the other schools in the same village
  cluster_median  bad UDISE point, no good village peers -> median of the good points in the same UDISE cluster
  udise           UDISE point kept ("udise_doubtful" when it has < 3 decimals or is > 5 km from its village's median)
  udise_flagged   bad UDISE point, no peers to replace it (kept, location_quality "poor")
A UDISE point is bad, and replaced, when it is shared by 4+ schools (placeholder) or lies > 1 km outside its listed
state. Rules were chosen by the Kerala calibration (verified points, 2026-11-07): replacing placeholders by the
village median cut their median error from 7.1 km to 2.6 km; replacing low-precision points (1.0 -> 2.7 km) or
village outliers (1.3 -> 5.5 km) made them worse, so those are only flagged.

Kerala is not corrected: its rows come from the verified Kerala layer
(FINAL COMPLETE DATASET/schools/kerala_schools_govt_aided_final.csv), joined to UDISE by the UDISE code on each
school's Sametham page. Kerala UDISE is only used to measure the method's error (calibration).

Output: FINAL COMPLETE DATASET/schools/india_schools_govt_aided.csv
        data/interim/schools/india_schools_calibration_kerala.csv
Usage:  python scripts/build_india_schools.py
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from build_india_health_facilities import decimals, fold, haversine_m, spatial_checks

ROOT = Path(__file__).resolve().parents[1]
UDISE = ROOT / "data" / "raw" / "udise_basic_details" / "basic-details-of-schools.csv"
OSM_INDIA = ROOT / "data" / "raw" / "osm" / "india"
OSM_KERALA = ROOT / "data" / "raw" / "osm" / "osm_kerala_schools_2026-11-07.json"
KERALA = ROOT / "FINAL COMPLETE DATASET" / "schools" / "kerala_schools_govt_aided_final.csv"
KERALA_CAP = ROOT / "FINAL COMPLETE DATASET" / "schools" / "kerala_schools_capacity.csv"
OUT = ROOT / "FINAL COMPLETE DATASET" / "schools" / "india_schools_govt_aided.csv"
CALIB = ROOT / "data" / "interim" / "schools" / "india_schools_calibration_kerala.csv"
METRIC_CRS = "EPSG:7755"
PLACEHOLDER_N, OUTSIDE_KM, OUTLIER_M, OSM_RADIUS_M, NAME_MIN = 4, 1, 5000, 3000, 90

GOVT = {"Department of Education", "Local body", "Tribal Welfare Department", "Social welfare Department",
        "Other Govt. managed schools", "Kendriya Vidyalaya", "Jawahar Navodaya Vidyalaya", "Ministry of Labor",
        "Railway School", "Others Central Government School", "Sainik School", "Central Tibetan School"}
AIDED = {"Government Aided", "Madarsa recognized (by Wakf board/Madarsa Board)"}
CLASSES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
COLS = ["state_name", "district_name", "subdistrict_name", "cluster_name", "village_name", "village_code", "pincode",
        "school_name", "udise_school_code", "school_category", "management", "location_type", "year_of_establishment",
        "class_from", "class_to", "latitude", "longitude", "class_rooms", "total_teachers", "class_students",
        "pre_primary_students"] + [f"{c}_students" for c in CLASSES]
# Words that describe the kind of school, not which school (Indian abbreviations and Hindi terms)
GENERIC = re.compile(
    r"\b(govt|government|gov|g|sarkari|rajkiya|raj|aided|a|school|schools|sch|vidyalaya|vidyalay|vidyala|vidhyalaya|"
    r"vidhyalay|vidya|vid|prathamik|prathmik|pry|primary|prim|p|ps|gps|gms|ms|mps|ups|upps|gups|lps|glps|lp|up|"
    r"uchch|uchcha|uchchtar|uchh|uchhtar|utkramit|madhya|madhyamik|middle|m|high|h|hs|hss|ghs|ghss|gss|gsss|sss|ss|"
    r"sr|sec|secondary|senior|higher|inter|intermediate|college|jr|junior|basic|b|balika|balak|girls|boys|kanya|"
    r"co|ed|coed|adarsh|model|new|nav|navin|u|s|gs|mpps|mpups|zphs|zpps|zpp|zp|zpups|mpl|mpp|municipal|corporation|"
    r"nagar|nigam|board|kgbv|kasturba|gandhi|kv|jnv|kendriya|navodaya|ashram|shala|shal|pathshala|pathsala|"
    r"english|hindi|medium|em|urdu|the|and|of|no|number|ii|iii|i)\b")


def core(name):
    s = re.sub(r"[^a-z ]", " ", fold(name).replace(".", " "))
    return re.sub(r"\s+", "", GENERIC.sub(" ", s))


def load_udise():
    u = pd.read_csv(UDISE, dtype=str, usecols=COLS)
    u = u[u["management"].isin(GOVT | AIDED)].copy()
    u["finance_group"] = np.where(u["management"].isin(GOVT), "Government", "Aided")
    u["lat"], u["lon"] = pd.to_numeric(u["latitude"], errors="coerce"), pd.to_numeric(u["longitude"], errors="coerce")
    u["dec"] = np.minimum(decimals(u["latitude"]), decimals(u["longitude"]))
    u["state_key"] = u["state_name"].map(fold)
    return u.reset_index(drop=True)


def load_osm():
    files = [f for f in sorted(OSM_INDIA.glob("osm_schools_*.json"))] + [OSM_KERALA]
    rows = []
    for f in files:
        for e in json.loads(f.read_text(encoding="utf-8"))["elements"]:
            t = e.get("tags", {})
            name = t.get("name:en") or t.get("name") or ""
            lat, lon = e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))
            if name and lat is not None:
                rows.append((f"{e['type']}/{e['id']}", name, lat, lon))
    o = pd.DataFrame(rows, columns=["osm_id", "osm_name", "osm_lat", "osm_lon"]).drop_duplicates("osm_id")
    o["ocore"] = o["osm_name"].map(core)
    print(f"OSM: {len(files)} files, {len(o)} named schools, {o['ocore'].str.len().ge(4).sum()} with a usable name")
    return o[o["ocore"].str.len() >= 4].reset_index(drop=True)


def flag_points(u):
    ok = u["lat"].between(6, 37.5) & u["lon"].between(68, 97.5)
    u.loc[~ok, ["lat", "lon"]] = np.nan
    n = u.groupby(["lat", "lon"])["udise_school_code"].transform("size")
    u["placeholder"] = ok & n.ge(PLACEHOLDER_N)
    u["low_precision"] = ok & u["dec"].lt(3)
    g = spatial_checks(u[["lat", "lon", "state_key"]].copy())
    u["km_outside_state"] = g["km_outside_state"].to_numpy()
    u["outside_state"] = ok & u["km_outside_state"].gt(OUTSIDE_KM)
    u["good"] = ok & ~u["placeholder"] & ~u["low_precision"] & ~u["outside_state"]
    # medians of good points per village and per cluster
    u["cluster_key"] = u["state_name"] + "|" + u["district_name"] + "|" + u["cluster_name"]
    for key, name in [("village_code", "village"), ("cluster_key", "cluster")]:
        good = u[u["good"]].groupby(key)
        u[f"{name}_n"] = u[key].map(good.size()).fillna(0).astype(int)
        u[f"{name}_lat"], u[f"{name}_lon"] = u[key].map(good["lat"].median()), u[key].map(good["lon"].median())
    d = haversine_m(u["lat"], u["lon"], u["village_lat"], u["village_lon"])
    u["village_outlier"] = u["good"] & u["village_n"].ge(3) & (d > OUTLIER_M)  # flag only (see docstring)
    u["bad"] = u["placeholder"] | u["outside_state"]
    return u


def anchor(u):
    """Where to look for the school: its UDISE point if good, else its peers' median."""
    v = u["village_n"].gt(u["good"].astype(int))  # at least one good peer other than itself
    lat = np.where(~u["bad"], u["lat"], np.where(v, u["village_lat"], u["cluster_lat"]))
    lon = np.where(~u["bad"], u["lon"], np.where(v, u["village_lon"], u["cluster_lon"]))
    return lat, lon


def osm_match(u, osm):
    """Unique OSM school with nearly the same name within OSM_RADIUS_M of the anchor."""
    u = u.assign(core=u["school_name"].map(core))
    lat, lon = anchor(u)
    s = u[(u["core"].str.len() >= 4) & ~np.isnan(lat)]
    a = gpd.GeoDataFrame({"i": s.index}, geometry=gpd.points_from_xy(lon[s.index], lat[s.index]), crs=4326)
    b = gpd.GeoDataFrame({"j": osm.index}, geometry=gpd.points_from_xy(osm["osm_lon"], osm["osm_lat"]), crs=4326)
    p = gpd.sjoin(a.to_crs(METRIC_CRS), b.to_crs(METRIC_CRS), predicate="dwithin", distance=OSM_RADIUS_M)
    p = pd.DataFrame({"i": p["i"].to_numpy(), "j": p["j"].to_numpy()})
    p["score"] = [fuzz.ratio(x, y) for x, y in zip(u.loc[p["i"], "core"].to_numpy(), osm.loc[p["j"], "ocore"].to_numpy())]
    p = p[p["score"] >= NAME_MIN]
    p = p[p.groupby("i")["j"].transform("size") == 1]   # exactly one OSM candidate for the school
    p = p[p.groupby("j")["i"].transform("size") == 1]   # and the OSM school is claimed by no other school
    print(f"OSM matches: {len(p)} schools ({len(p) / len(u):.1%})")
    return p.set_index("i")


def correct(u, osm):
    u = flag_points(u)
    m = osm_match(u, osm)
    o = osm.loc[m["j"]].set_index(m.index)
    u["udise_lat"], u["udise_lon"] = u["lat"], u["lon"]
    has_v = u["village_n"].gt(u["good"].astype(int))
    has_c = u["cluster_n"].gt(u["good"].astype(int))
    src = np.select([u.index.isin(m.index), u["lat"].isna(), ~u["bad"], u["bad"] & has_v, u["bad"] & has_c],
                    ["osm", "none", "udise", "village_median", "cluster_median"], "udise_flagged")
    u["location_source"] = src
    u["lat"] = np.select([src == "osm", src == "village_median", src == "cluster_median"],
                         [o["osm_lat"].reindex(u.index), u["village_lat"], u["cluster_lat"]], u["udise_lat"])
    u["lon"] = np.select([src == "osm", src == "village_median", src == "cluster_median"],
                         [o["osm_lon"].reindex(u.index), u["village_lon"], u["cluster_lon"]], u["udise_lon"])
    u["location_quality"] = pd.Series(src).map({"osm": "good", "udise": "udise", "village_median": "approximate",
                                                "cluster_median": "coarse", "udise_flagged": "poor",
                                                "none": "none"}).to_numpy()
    u.loc[u["location_source"].eq("udise") & (u["low_precision"] | u["village_outlier"]), "location_quality"] = "udise_doubtful"
    u["osm_id"], u["osm_name"], u["osm_score"] = o["osm_id"].reindex(u.index), o["osm_name"].reindex(u.index), m["score"].reindex(u.index)
    u["moved_m"] = haversine_m(u["udise_lat"], u["udise_lon"], u["lat"], u["lon"]).round(0)
    return u


def calibrate(k):
    """Kerala: compare raw and corrected UDISE points with the verified Kerala layer."""
    cap = pd.read_csv(KERALA_CAP, dtype=str, usecols=["school_code_num", "udise_code"])
    ver = pd.read_csv(KERALA, dtype={"school_code_num": str}, usecols=["school_code_num", "lat", "lon"], low_memory=False)
    v = cap.merge(ver, on="school_code_num").dropna(subset=["udise_code", "lat"])
    c = k.merge(v.rename(columns={"lat": "ver_lat", "lon": "ver_lon"}), left_on="udise_school_code", right_on="udise_code")
    c["err_raw_m"] = haversine_m(c["udise_lat"], c["udise_lon"], c["ver_lat"], c["ver_lon"])
    c["err_new_m"] = haversine_m(c["lat"], c["lon"], c["ver_lat"], c["ver_lon"])

    def stats(e):
        return f"median {np.nanmedian(e):6.0f} m | >250 m {np.mean(e > 250):5.1%} | >1 km {np.mean(e > 1000):5.1%}"
    print(f"Kerala calibration: {len(c)} schools linked to the verified layer")
    print("  raw UDISE      ", stats(c["err_raw_m"]))
    print("  corrected      ", stats(c["err_new_m"]))
    for s, g in c.groupby("location_source"):
        print(f"  {s:15s} n={len(g):5d}  raw: {stats(g['err_raw_m'])}\n  {'':15s}          new: {stats(g['err_new_m'])}")
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    c[["udise_school_code", "school_code_num", "school_name", "location_source", "udise_lat", "udise_lon", "lat", "lon",
       "ver_lat", "ver_lon", "err_raw_m", "err_new_m"]].to_csv(CALIB, index=False, encoding="utf-8")
    return c


def kerala_rows(k):
    """Kerala rows from the verified layer, with UDISE attributes joined by UDISE code."""
    ver = pd.read_csv(KERALA, dtype={"school_code_num": str}, low_memory=False)
    cap = pd.read_csv(KERALA_CAP, dtype=str, usecols=["school_code_num", "udise_code"])
    ver = ver.drop(columns="udise_code").merge(cap, on="school_code_num", how="left")
    keep = [c for c in k.columns if c not in ("lat", "lon", "location_source", "location_quality")]
    r = ver[["school_code_num", "school_name", "district", "finance_type", "lat", "lon", "location_source",
             "location_accuracy", "udise_code"]].merge(
        k[keep].drop(columns=["school_name"]), left_on="udise_code", right_on="udise_school_code", how="left")
    r["udise_school_code"] = r["udise_code"]
    r["state_name"], r["district_name"] = "Kerala", r["district"]
    r["finance_group"] = r["finance_type"]
    r["location_source"] = "kerala_verified:" + r["location_source"].astype(str)
    r["location_quality"] = r["location_accuracy"]
    r[["osm_id", "osm_name", "osm_score"]] = np.nan  # the verified point is used, not the UDISE correction
    r["moved_m"] = haversine_m(r["udise_lat"], r["udise_lon"], r["lat"], r["lon"]).round(0)
    r = r.rename(columns={"school_code_num": "kerala_school_code"})
    return r.drop(columns=["district", "finance_type", "location_accuracy", "udise_code"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    u, osm = load_udise(), load_osm()
    print(f"UDISE Government + Aided: {len(u)} ({u['finance_group'].value_counts().to_dict()})")
    u = correct(u, osm)
    calibrate(u[u["state_name"].eq("Kerala")])
    out = pd.concat([u[u["state_name"].ne("Kerala")], kerala_rows(u[u["state_name"].eq("Kerala")])], ignore_index=True)
    drop = ["latitude", "longitude", "dec", "state_key", "cluster_key", "good", "bad", "village_n", "village_lat",
            "village_lon", "cluster_n", "cluster_lat", "cluster_lon"]
    out = out.drop(columns=[c for c in drop if c in out]).rename(columns={"udise_school_code": "udise_code"})
    front = ["udise_code", "kerala_school_code", "school_name", "state_name", "district_name", "finance_group",
             "management", "school_category", "class_from", "class_to", "lat", "lon", "location_source", "location_quality"]
    out = out[front + [c for c in out.columns if c not in front]]

    # self-checks
    assert out.loc[out["state_name"].ne("Kerala"), "udise_code"].is_unique, "duplicate UDISE codes"
    print("Kerala schools sharing a UDISE code:", int(out.loc[out["state_name"].eq("Kerala"), "udise_code"].dropna().duplicated().sum()))
    assert out.loc[out["location_source"].isin(["osm", "village_median", "cluster_median"]), "lat"].notna().all()
    assert (out["moved_m"].dropna() >= 0).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"schools: {len(out)} | located: {out['lat'].notna().sum()}")
    print(out["location_source"].str.replace(r":.*", "", regex=True).value_counts().to_string())
    print("written:", OUT)


if __name__ == "__main__":
    main()
