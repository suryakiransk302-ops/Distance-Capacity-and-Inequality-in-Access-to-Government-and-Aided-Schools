"""Combine the two MoHFW health facility directories into one all-India dataset.

Sources (the same two that Esri India merged into "Healthcare Facilities in India"):
  NIN  NIN Health Facilities with Geo-code (National Health Portal), via Esri India
       "India: Hospital Directory" -> data/raw/health/nin_hospital_directory/nin_hospital_directory.csv
       (scripts/download_nin_hospital_directory.py). Layer last updated 2021-11-30.
  HCD  All India Health Centres Directory, as on 7 Oct 2016 (data.gov.in / AIKOSH)
       -> data/raw/health/geocode_health_centre.csv

Steps
  1. Clean: drop HCD "Andhra Pradesh Old" (undivided AP, repeated as Andhra Pradesh + Telangana),
     fix swapped lat/lon, mark points outside India as missing.
  2. Standardise facility types (SC, PHC, CHC, UHC, UHP, SDH, DH, HOSP, MCH, MAT, DISP, OTH).
  3. Match HCD to NIN one-to-one, same state and compatible type:
       pass 1  within 5 km and name similarity >= 0.85
       pass 2  within 25 km and identical core name (catches placeholder points)
     Matched pairs become one facility; unmatched records from both sources are kept.
  4. Choose the location: NIN (newer) unless its point is a placeholder or low precision and
     the HCD point is not.
  5. QA: placeholder points (>= 4 facilities on one exact point in a source), low precision
     (< 3 decimals), point outside the listed state (geoBoundaries ADM1, > 1 km).
     Each point is also given its geoBoundaries ADM2 district for analysis.

Outputs: data/processed/health/india_health_facilities.{csv,gpkg}
Usage:   python scripts/build_india_health_facilities.py
"""

import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NIN_CSV = ROOT / "data" / "raw" / "health" / "nin_hospital_directory" / "nin_hospital_directory.csv"
HCD_CSV = ROOT / "data" / "raw" / "health" / "geocode_health_centre.csv"
ADM1 = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM1.geojson"
ADM2 = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM2.geojson"
OUT_DIR = ROOT / "data" / "processed" / "health"
METRIC_CRS = "EPSG:7755"  # India NSF LCC
NAME_MIN = 0.85
NEAR_M, FAR_M = 5000, 25000
PLACEHOLDER_N = 4
OUTSIDE_STATE_KM = 1

NIN_TYPES = {
    "SubCentre": "SC", "Primary Health Centre": "PHC", "Community Health Center": "CHC",
    "Urban Health Centre": "UHC", "Urban Health Posts": "UHP", "Sub-District Hospital": "SDH",
    "District Hospital": "DH", "Civil Hospital/General Hospital": "HOSP", "Referral Hospital": "HOSP",
    "<100 Bedded Hospital": "HOSP", "100-500 Bedded Hospital": "HOSP", ">500 Bedded Hospital": "HOSP",
    "Medical Colleges Hospital": "MCH", "Maternity Home": "MAT", "M&CW Center": "MAT",
    "Post Partum Unit": "MAT", "Women Hospital": "MAT", "Dispensaries": "DISP",
    "Ayush Dispensaries": "DISP", "Others": "OTH"}
# s_t_h = sub-district / taluk hospital (Kerala: 77 in HCD vs 88 taluk hospitals in the DHS list)
HCD_TYPES = {"sub_cen": "SC", "phc": "PHC", "chc": "CHC", "s_t_h": "SDH", "dis_h": "DH"}
TYPE_NAMES = {
    "SC": "Sub Centre", "PHC": "Primary Health Centre", "CHC": "Community Health Centre",
    "UHC": "Urban Health Centre / Urban PHC", "UHP": "Urban Health Post",
    "SDH": "Sub-District / Taluk Hospital", "DH": "District Hospital",
    "HOSP": "Other public hospital", "MCH": "Medical College Hospital",
    "MAT": "Maternity / women and child facility", "DISP": "Dispensary", "OTH": "Other"}
# Types that may describe the same facility in the two sources.
GROUP = {"SC": "sc", "PHC": "phc", "UHC": "phc", "CHC": "chc", "SDH": "hosp", "DH": "hosp",
         "HOSP": "hosp", "MCH": "hosp", "MAT": "hosp", "UHP": "other", "DISP": "other", "OTH": "any"}
STATE_ALIASES = {"a & n islands": "andaman and nicobar islands", "telengana": "telangana",
                 "dadra & nagar haveli": "dadra and nagar haveli and daman and diu",
                 "daman & diu": "dadra and nagar haveli and daman and diu",
                 "jammu & kashmir": "jammu and kashmir"}
GENERIC = re.compile(
    r"\b(sub|subcent\w*|subcenter|centre|center|centr|cent|sc|hsc|shc|ssc|phc|aphc|bphc|uphc|chc|uhc|uhp|"
    r"hwc|abhwc|hwcsc|"
    r"health|wellness|primary|community|urban|rural|post|hospital|hosp|district|dh|sdh|govt|"
    r"government|area|main|maincentre|new|addl|additional|null|na|and|of|the|h|c|s|p)\b")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"\s+", " ", s)
    return STATE_ALIASES.get(s, s)


def core(name):
    s = re.sub(r"[^a-z ]", " ", fold(name))
    return re.sub(r"\s+", " ", GENERIC.sub(" ", s)).strip()


def decimals(x):
    return x.astype(str).str.extract(r"\.(\d+)")[0].str.len().fillna(0).astype(int)


def load_nin():
    d = pd.read_csv(NIN_CSV, dtype=str)
    out = pd.DataFrame({
        "src_id": "NIN:" + d["objectid"], "name": d["health_facility_name"].str.strip(),
        "type_original": d["facility_type"], "std_type": d["facility_type"].map(NIN_TYPES),
        "state": d["state_name"], "district": d["district_name"], "subdistrict": d["taluka_name"],
        "address": d[["address", "street", "locality"]].fillna("").agg(", ".join, axis=1).str.strip(", "),
        "pin_code": d["posatalcode"].str.replace(r"\.0$", "", regex=True),
        "lat": pd.to_numeric(d["lat"]), "lon": pd.to_numeric(d["lon"]),
        "location_type": None, "nin": None})
    out["dec"] = np.minimum(decimals(d["lat"]), decimals(d["lon"]))
    return out


def load_hcd():
    d = pd.read_csv(HCD_CSV, dtype=str)
    n0 = len(d)
    d = d[d["State Name"].ne("Andhra Pradesh Old") & d["ActiveFlag_C"].eq("Y")]
    print(f"HCD: {n0} rows, {n0 - len(d)} dropped (Andhra Pradesh Old / shifted rows)")
    lat, lon = pd.to_numeric(d["Latitude"], errors="coerce"), pd.to_numeric(d["Longitude"], errors="coerce")
    swap = lon.between(6, 37.5) & lat.between(68, 97.5)
    lat, lon = lat.where(~swap, lon), lon.where(~swap, lat)
    out = pd.DataFrame({
        "src_id": "HCD:" + d.index.astype(str), "name": d["Facility Name"].str.replace("?NULL", "", regex=False).str.strip(),
        "type_original": d["Facility Type"], "std_type": d["Facility Type"].map(HCD_TYPES),
        "state": d["State Name"], "district": d["District Name"], "subdistrict": d["Subdistrict Name"],
        "address": d["Facility Address"].replace("NA", None), "pin_code": None,
        "lat": lat, "lon": lon, "location_type": d["Location Type"],
        "nin": d["Nin_N"].str.replace(r"\.0$", "", regex=True)})
    out["dec"] = np.minimum(decimals(d["Latitude"]), decimals(d["Longitude"]))
    out["swapped"] = swap.to_numpy()
    print(f"HCD: {int(swap.sum())} swapped lat/lon fixed")
    return out


def add_point_quality(df):
    ok = df["lat"].between(6, 37.5) & df["lon"].between(68, 97.5)
    df.loc[~ok, ["lat", "lon"]] = np.nan
    n = df.groupby(["lat", "lon"])["src_id"].transform("size")
    df["placeholder"] = ok & n.ge(PLACEHOLDER_N)
    df["low_precision"] = ok & df["dec"].lt(3)
    df["good_point"] = ok & ~df["placeholder"] & ~df["low_precision"]
    df["state_key"] = df["state"].map(fold)
    df["group"] = df["std_type"].map(GROUP)
    df["core"] = df["name"].map(core)
    return df


def candidate_pairs(nin, hcd, radius):
    print(f"  candidate pairs within {radius} m ...", flush=True)
    a = gpd.GeoDataFrame(hcd[["src_id"]], geometry=gpd.points_from_xy(hcd["lon"], hcd["lat"]), crs=4326).to_crs(METRIC_CRS)
    b = gpd.GeoDataFrame(nin[["src_id"]], geometry=gpd.points_from_xy(nin["lon"], nin["lat"]), crs=4326).to_crs(METRIC_CRS)
    a = a[~a.geometry.is_empty & a.geometry.x.notna()]
    b = b[~b.geometry.is_empty & b.geometry.x.notna()]
    j = gpd.sjoin(a, b, predicate="dwithin", distance=radius, how="inner")
    p = pd.DataFrame({"hcd": j["src_id_left"].to_numpy(), "nin": j["src_id_right"].to_numpy()})
    pa, pb = a.set_index("src_id").geometry, b.set_index("src_id").geometry
    p["dist_m"] = np.hypot(pa.loc[p["hcd"]].x.to_numpy() - pb.loc[p["nin"]].x.to_numpy(),
                           pa.loc[p["hcd"]].y.to_numpy() - pb.loc[p["nin"]].y.to_numpy())
    return p


def compatible(p, nin, hcd):
    h, n = hcd.set_index("src_id"), nin.set_index("src_id")
    for col in ["state_key", "group", "core"]:
        p[f"h_{col}"] = h.loc[p["hcd"], col].to_numpy()
        p[f"n_{col}"] = n.loc[p["nin"], col].to_numpy()
    keep = p["h_state_key"].eq(p["n_state_key"]) & (p["h_group"].eq(p["n_group"]) | p["n_group"].eq("any"))
    return p[keep & p["h_core"].ne("") & p["n_core"].ne("")].copy()


def same_name_pairs(nin, hcd):
    keys = ["state_key", "group", "core"]
    h = hcd[hcd["core"].ne("") & hcd["lat"].notna()][["src_id", "lat", "lon"] + keys]
    n = nin[nin["core"].ne("") & nin["lat"].notna()][["src_id", "lat", "lon"] + keys]
    p = h.merge(n, on=keys, suffixes=("_h", "_n"))
    # NIN "Others" may be any type
    n_any = n[n["group"].eq("any")].drop(columns="group")
    p = pd.concat([p, h.merge(n_any, on=["state_key", "core"], suffixes=("_h", "_n"))], ignore_index=True)
    p["dist_m"] = haversine_m(p["lat_h"], p["lon_h"], p["lat_n"], p["lon_n"])
    return p.rename(columns={"src_id_h": "hcd", "src_id_n": "nin"})[["hcd", "nin", "dist_m"]]


def greedy(p, taken_h, taken_n):
    p = p.sort_values(["score", "dist_m"], ascending=[False, True])
    pairs = []
    for h, n in zip(p["hcd"], p["nin"]):
        if h not in taken_h and n not in taken_n:
            taken_h.add(h)
            taken_n.add(n)
            pairs.append((h, n))
    return pairs


def match(nin, hcd):
    taken_h, taken_n = set(), set()
    p = compatible(candidate_pairs(nin, hcd, NEAR_M), nin, hcd)
    p["score"] = [SequenceMatcher(None, x, y).ratio() for x, y in zip(p["h_core"], p["n_core"])]
    pass1 = greedy(p[p["score"].ge(NAME_MIN)], taken_h, taken_n)
    p = same_name_pairs(nin[~nin["src_id"].isin(taken_n)], hcd[~hcd["src_id"].isin(taken_h)])
    p = p[p["dist_m"].le(FAR_M)].copy()
    p["score"] = 1.0
    # the name must be unique among candidates on both sides, so a common village name is not paired blindly
    p = p[p.groupby("hcd")["nin"].transform("size").eq(1) & p.groupby("nin")["hcd"].transform("size").eq(1)]
    pass2 = greedy(p, taken_h, taken_n)
    print(f"matched pass 1 (<= {NEAR_M} m, name >= {NAME_MIN}): {len(pass1)} | pass 2 (<= {FAR_M} m, same name): {len(pass2)}")
    return pd.DataFrame(pass1 + pass2, columns=["hcd", "nin"]).assign(
        match_pass=[1] * len(pass1) + [2] * len(pass2))


def combine(nin, hcd, pairs):
    n, h = nin.set_index("src_id"), hcd.set_index("src_id")
    rows = []
    pn, ph = n.loc[pairs["nin"]].reset_index(), h.loc[pairs["hcd"]].reset_index()
    use_h = ~pn["good_point"] & ph["good_point"]
    both = pn.copy()
    both["nin"] = ph["nin"].to_numpy()
    both["location_type"] = ph["location_type"].to_numpy()
    for col in ["lat", "lon", "dec", "placeholder", "low_precision", "good_point"]:
        both[col] = np.where(use_h, ph[col], pn[col])
    both["location_from"] = np.where(use_h, "HCD_2016", "NIN")
    both["in_nin"], both["in_hcd"] = True, True
    both["nin_id"], both["hcd_id"] = pn["src_id"].to_numpy(), ph["src_id"].to_numpy()
    both["hcd_type"] = ph["type_original"].to_numpy()
    both["nin_vs_hcd_m"] = haversine_m(pn["lat"], pn["lon"], ph["lat"], ph["lon"]).round(0)
    both["match_pass"] = pairs["match_pass"].to_numpy()
    rows.append(both)

    only_n = n.drop(pairs["nin"]).reset_index()
    only_n = only_n.assign(location_from="NIN", in_nin=True, in_hcd=False, nin_id=only_n["src_id"])
    only_h = h.drop(pairs["hcd"]).reset_index()
    only_h = only_h.assign(location_from="HCD_2016", in_nin=False, in_hcd=True, hcd_id=only_h["src_id"])
    rows += [only_n, only_h]
    out = pd.concat(rows, ignore_index=True)
    out.loc[out["lat"].isna(), "location_from"] = "none"
    return out


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [np.asarray(lat1, float), np.asarray(lon1, float),
                                               np.asarray(lat2, float), np.asarray(lon2, float)])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def spatial_checks(df):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    has = g["lat"].notna()
    s1 = gpd.read_file(ADM1)[["shapeName", "geometry"]]
    s1["state_geo"] = s1["shapeName"].map(fold)
    pts = g.loc[has, ["geometry"]].to_crs(METRIC_CRS)
    near = gpd.sjoin_nearest(pts, s1[["state_geo", "geometry"]].to_crs(METRIC_CRS), how="left", distance_col="d")
    near = near[~near.index.duplicated()]
    # distance to the listed state's polygon
    listed = s1.set_index("state_geo").to_crs(METRIC_CRS).geometry
    key = g.loc[has, "state_key"].replace({"jammu and kashmir": "jammu and kashmir"})
    dist = pd.Series(np.nan, index=pts.index)
    for st, idx in key.groupby(key).groups.items():
        if st in listed.index:
            dist.loc[idx] = pts.loc[idx].distance(listed.loc[st]).to_numpy()
    g["state_geo"] = near["state_geo"]
    g["km_outside_state"] = (dist / 1000).round(2)
    # Ladakh was part of Jammu and Kashmir state until 2019
    lad = g["state_key"].eq("jammu and kashmir") & g["state_geo"].eq("ladakh")
    g.loc[lad, "km_outside_state"] = 0.0

    d2 = gpd.read_file(ADM2)[["shapeName", "shapeID", "geometry"]].rename(
        columns={"shapeName": "district_geo", "shapeID": "district_geo_id"})
    j = gpd.sjoin(g.loc[has, ["geometry"]], d2, predicate="within", how="left")
    j = j[~j.index.duplicated()]
    g["district_geo"], g["district_geo_id"] = j["district_geo"], j["district_geo_id"]
    return g


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    nin, hcd = add_point_quality(load_nin()), add_point_quality(load_hcd())
    for name, d in [("NIN", nin), ("HCD", hcd)]:
        print(f"{name}: {len(d)} records | no valid point {int(d['lat'].isna().sum())} | "
              f"placeholder {int(d['placeholder'].sum())} | low precision {int(d['low_precision'].sum())}")
    pairs = match(nin, hcd)
    df = combine(nin, hcd, pairs)
    g = spatial_checks(df)

    g["qa_flags"] = (pd.Series(np.where(g["lat"].isna(), "no_location ", ""), index=g.index)
                     + np.where(g["placeholder"].fillna(False), "placeholder_point ", "")
                     + np.where(g["low_precision"].fillna(False), "low_precision ", "")
                     + np.where(g["km_outside_state"].gt(OUTSIDE_STATE_KM), "outside_listed_state", "")).str.strip()
    g["location_quality"] = np.select(
        [g["lat"].isna(), g["qa_flags"].eq(""), g["qa_flags"].str.contains("outside_listed_state|placeholder")],
        ["none", "good", "poor"], "approximate")
    g["state"] = g["state_key"].str.title()
    g["type_name"] = g["std_type"].map(TYPE_NAMES)
    g["sources"] = np.select([g["in_nin"] & g["in_hcd"], g["in_nin"]], ["NIN+HCD", "NIN"], "HCD")
    g = g.sort_values(["state", "district", "std_type", "name"]).reset_index(drop=True)
    g["facility_id"] = [f"IN_HF_{i:06d}" for i in range(1, len(g) + 1)]
    g["service"] = "health"
    g["lat"], g["lon"] = g["lat"].round(7), g["lon"].round(7)

    cols = ["facility_id", "name", "std_type", "type_name", "type_original", "hcd_type", "state", "district",
            "subdistrict", "address", "pin_code", "location_type", "nin", "lat", "lon", "location_from",
            "location_quality", "qa_flags", "sources", "nin_id", "hcd_id", "match_pass", "nin_vs_hcd_m",
            "state_geo", "km_outside_state", "district_geo", "district_geo_id", "service", "geometry"]
    out = gpd.GeoDataFrame(g[cols], geometry="geometry", crs=4326)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.drop(columns="geometry").to_csv(OUT_DIR / "india_health_facilities.csv", index=False, encoding="utf-8")
    out[out["lat"].notna()].to_file(OUT_DIR / "india_health_facilities.gpkg", layer="health_facilities", driver="GPKG")

    print("\nfacilities:", len(out), "| sources:", out["sources"].value_counts().to_dict())
    print("location quality:", out["location_quality"].value_counts().to_dict())
    print("qa flags:", out["qa_flags"].str.split().explode().value_counts().to_dict())
    print("\nby type:\n", pd.crosstab(out["type_name"], out["sources"], margins=True).to_string())
    print("\nby state:\n", pd.crosstab(out["state"], out["location_quality"], margins=True).to_string())
    print("\nwritten:", OUT_DIR)


if __name__ == "__main__":
    main()
