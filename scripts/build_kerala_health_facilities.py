"""Build the Kerala public health facility layer for the Kerala case study.

Facility list: DHS Kerala "List of Modern Medicine Institutions 2024-25"
               (scripts/extract_kerala_dhs_institutions.py)
  institutions (hospitals, CHC, PHC/FHC, dispensaries, ...): coordinates printed in the list
  sub centres: no coordinates in the list -> taken from the All India Health Centres Directory
               (MoHFW, 7 Oct 2016), matched within the same district on the cleaned name
               (similarity >= 0.85, best match, one-to-one)

QA: every point inside its listed district (geoBoundaries ADM2, > 1 km = flag); institutions
compared with the 2016 directory point of the same NIN; each sub centre's distance to its
parent CHC/PHC/FHC (> 15 km = flag).

Outputs: data/processed/health/kerala_health_facilities.{csv,gpkg}
         data/interim/health/kerala_health_to_check.csv   (no location or flagged)
Usage:   python scripts/build_kerala_health_facilities.py
"""

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from build_india_health_facilities import haversine_m

ROOT = Path(__file__).resolve().parents[1]
INST = ROOT / "data" / "interim" / "health" / "kerala_dhs_institutions_2024_25.csv"
SUBC = ROOT / "data" / "interim" / "health" / "kerala_dhs_subcentres_2024_25.csv"
HCD_CSV = ROOT / "data" / "raw" / "health" / "geocode_health_centre.csv"
ADM2 = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM2.geojson"
OUT_DIR = ROOT / "data" / "processed" / "health"
CHECK_CSV = ROOT / "data" / "interim" / "health" / "kerala_health_to_check.csv"
NAME_MIN = 0.85
PARENT_KM = 15
OUTSIDE_KM = 1
KERALA_DISTRICTS = ["Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam", "Idukki",
                    "Ernakulam", "Thrissur", "Palakkad", "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod"]
GEO_NAMES = {"Kasargod": "Kasaragod", "Kasaragod": "Kasaragod"}
# DHS central category -> standard type used across the project (same codes as the India layer)
CATEGORY_TYPES = {"DH": "DH", "GH": "DH", "W&C": "MAT", "THQH": "SDH", "TH": "SDH", "CHC": "CHC", "PHC": "PHC",
                  "FHC": "PHC", "DTBC": "SPEC", "TBC": "SPEC", "MHC": "SPEC", "LEP": "SPEC", "HC": "DISP",
                  "GD": "DISP", "MU": "MOB"}
SECTION_TYPES = [("District Hospital", "DH"), ("General Hospital", "DH"), ("W & C", "MAT"), ("Taluk", "SDH"),
                 ("Community health", "CHC"), ("Primary Health", "PHC"), ("Family Health", "PHC"),
                 ("TB", "SPEC"), ("Mental", "SPEC"), ("Leprosy", "SPEC"), ("Speciality", "SPEC"),
                 ("Health Clinic", "DISP"), ("Dispensary", "DISP"), ("Mobile", "MOB")]
TYPE_NAMES = {"DH": "District / General Hospital", "MAT": "Women and Child Hospital", "SDH": "Taluk (HQ) Hospital",
              "CHC": "Community Health Centre", "PHC": "Primary / Family Health Centre", "SC": "Sub Centre",
              "SPEC": "Speciality centre (TB, mental health, leprosy)", "DISP": "Dispensary / health clinic",
              "MOB": "Mobile unit"}
OSM_JSON = ROOT / "data" / "raw" / "osm" / "osm_kerala_health_2026-11-02.json"
OSM_SKIP = {"pharmacy", "dentist", "alternative", "laboratory", "hospice", "rehabilitation", "blood_donation",
            "optometrist", "physiotherapist", "psychotherapist", "veterinary"}
SC_LIKE = r"(?i)sub.?cent|\bfwc\b|\bhwc\b|welfare|wellness|population project|\bicdp\b|\bipp\b|janakeeya|arogya"
OSM_REF_KM = 10
AGREE_M = 250
PARENT_WORDS = {"phc", "fhc", "chc", "thqh", "th", "dh", "gh", "mini", "family", "primary", "community", "health",
                "centre", "center", "hospital", "taluk", "headquarters", "head", "quarters", "district", "general",
                "govt", "government", "the", "of"}
OSM_WORDS = re.compile(r"\b(phc|fhc|chc|thqh|dh|gh|primary|community|hospital|taluk|taluka|headquarters|"
                       r"district|general|government|panchayat|grama|icdp|indian|population|project|h|fw|"
                       r"janakeeya|arogya|kendram|urban|rural|block|mini|homoeo|ayurveda|kerala|the|of)\b")
SC_WORDS = re.compile(r"\b(fwc|fw|hwc|ipp|rch|sc|s c|shc|sub|subcent\w*|subcenter|centre|center|centr|main|"
                      r"maincentre|maincenter|health|wellness|family|welfare|govt|new|additional|addl)\b")


def core(name):
    s = re.sub(r"[^a-z ]", " ", str(name).lower())
    return re.sub(r"\s+", " ", SC_WORDS.sub(" ", s)).strip()


def district_name(s):
    t = str(s).strip().title().replace("Kozhikkode", "Kozhikode").replace("Trivandrum", "Thiruvananthapuram")
    return {"Kasargod": "Kasaragod", "Palghat": "Palakkad"}.get(t, t)


def institution_type(r):
    t = CATEGORY_TYPES.get(str(r["central_category"]).strip().upper())
    return t or next((v for k, v in SECTION_TYPES if k.lower() in str(r["section"]).lower()), "OTH")


def load_hcd_kerala():
    h = pd.read_csv(HCD_CSV, dtype=str)
    h = h[h["State Name"].eq("Kerala")].copy()
    h["district"] = h["District Name"].map(district_name)
    h["lat"], h["lon"] = pd.to_numeric(h["Latitude"], errors="coerce"), pd.to_numeric(h["Longitude"], errors="coerce")
    h["nin"] = h["Nin_N"].str.replace(r"\.0$", "", regex=True)
    h["core"] = h["Facility Name"].map(core)
    return h


def match_subcentres(s, h):
    hs = h[h["Facility Type"].eq("sub_cen") & h["lat"].notna() & h["core"].ne("")]
    p = s.reset_index()[["index", "district", "core"]].merge(
        hs.reset_index()[["index", "district", "core"]], on="district", suffixes=("_s", "_h"))
    p["score"] = [SequenceMatcher(None, a, b).ratio() for a, b in zip(p["core_s"], p["core_h"])]
    p = p[p["score"].ge(NAME_MIN)].sort_values("score", ascending=False)
    used_s, used_h, pairs = set(), set(), []
    for a, b, sc in zip(p["index_s"], p["index_h"], p["score"]):
        if a not in used_s and b not in used_h:
            used_s.add(a)
            used_h.add(b)
            pairs.append((a, b, sc))
    return pd.DataFrame(pairs, columns=["s", "h", "name_score"]).set_index("s")


def parent_key(name):
    """'Kadangode FHC' and 'FHC Kadangode' -> 'kadangode'."""
    words = re.sub(r"[^a-z ]", " ", str(name).lower()).split()
    return " ".join(sorted(w for w in words if w not in PARENT_WORDS))


def ocore(name):
    s = re.sub(r"[^a-z ]", " ", str(name).lower())
    s = OSM_WORDS.sub(" ", SC_WORDS.sub(" ", s))
    return re.sub(r"\s+", " ", s).strip()


def load_osm():
    els = json.loads(OSM_JSON.read_text(encoding="utf-8"))["elements"]
    rows = []
    for e in els:
        t = e.get("tags", {})
        if not t.get("name") or t.get("amenity") in OSM_SKIP or t.get("healthcare") in OSM_SKIP:
            continue
        lat, lon = e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))
        rows.append({"osm_id": f"{e['type']}/{e['id']}", "osm_name": t["name"], "osm_lat": lat, "osm_lon": lon})
    o = pd.DataFrame(rows)
    o["sc_like"] = o["osm_name"].str.contains(SC_LIKE, regex=True)
    o["ocore"] = o["osm_name"].map(ocore)
    g = gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o["osm_lon"], o["osm_lat"]), crs=4326)
    d = gpd.read_file(ADM2)[["shapeName", "geometry"]].cx[74.5:77.8, 7.9:13.1]
    d["district"] = d["shapeName"].map(district_name)
    g = gpd.sjoin(g, d[d["district"].isin(KERALA_DISTRICTS)][["district", "geometry"]], predicate="within")
    return pd.DataFrame(g.drop(columns=["geometry", "index_right"]))


def attach_osm(df, osm, ref, alt_ref=None):
    """One-to-one OSM match: same district, cleaned name similarity >= NAME_MIN, and within
    OSM_REF_KM of the facility's reference point (its own point, parent facility or 2016 point)."""
    df = df.copy()
    df["ocore"] = df["name"].map(ocore)
    p = df.reset_index()[["index", "district", "ocore"]].merge(osm, on="district")
    p = p[p["ocore_x"].ne("") & p["ocore_y"].ne("")]
    p["score"] = [SequenceMatcher(None, a, b).ratio() for a, b in zip(p["ocore_x"], p["ocore_y"])]
    p = p[p["score"].ge(NAME_MIN)]
    d = pd.Series(np.inf, index=p.index)
    for lat, lon in [ref] + ([alt_ref] if alt_ref else []):
        dd = haversine_m(df.loc[p["index"], lat], df.loc[p["index"], lon], p["osm_lat"], p["osm_lon"]) / 1000
        d = np.fmin(d, pd.Series(dd, index=p.index).fillna(np.inf))
    p = p[d.le(OSM_REF_KM)].assign(ref_km=d).sort_values(["score", "ref_km"], ascending=[False, True])
    used_f, used_o, keep = set(), set(), []
    for i, f, o in zip(p.index, p["index"], p["osm_id"]):
        if f not in used_f and o not in used_o:
            used_f.add(f)
            used_o.add(o)
            keep.append(i)
    p = p.loc[keep].set_index("index")
    for c in ["osm_id", "osm_name", "osm_lat", "osm_lon"]:
        df[c] = p[c].reindex(df.index)
    df["osm_name_score"] = p["score"].reindex(df.index).round(2)
    return df.drop(columns="ocore")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    inst = pd.read_csv(INST, dtype={"nin": str, "sl_no": str})
    subc = pd.read_csv(SUBC, dtype={"nin": str, "sl_no": str})
    hcd = load_hcd_kerala()

    inst["std_type"] = inst.apply(institution_type, axis=1)
    inst["location_from"] = np.where(inst["lat"].notna(), "DHS_2024_25", "none")
    h_by_nin = hcd[hcd["nin"].notna()].drop_duplicates("nin").set_index("nin")
    inst["hcd_2016_vs_dhs_m"] = haversine_m(inst["lat"], inst["lon"], inst["nin"].map(h_by_nin["lat"]),
                                            inst["nin"].map(h_by_nin["lon"])).round(0)

    subc["district"] = subc["district"].map(district_name)
    subc["core"] = subc["name"].map(core)
    m = match_subcentres(subc, hcd)
    subc["std_type"] = "SC"
    subc["hcd_lat"], subc["hcd_lon"] = subc.index.map(m["h"]).map(hcd["lat"]), subc.index.map(m["h"]).map(hcd["lon"])
    subc["matched_hcd_name"] = subc.index.map(m["h"]).map(hcd["Facility Name"])
    subc["name_score"] = subc.index.map(m["name_score"]).astype(float).round(2)
    # parent facility: match the printed parent name to an institution of the same district
    inst["parent_key"] = inst["name"].map(parent_key)
    lookup = inst[inst["lat"].notna()].drop_duplicates(["district", "parent_key"]).set_index(["district", "parent_key"])
    key = pd.MultiIndex.from_arrays([subc["district"], subc["parent_facility"].map(parent_key)])
    subc["parent_lat"] = lookup["lat"].reindex(key).to_numpy()
    subc["parent_lon"] = lookup["lon"].reindex(key).to_numpy()

    osm = load_osm()
    inst = attach_osm(inst, osm[~osm["sc_like"]], ref=("lat", "lon"))
    subc = attach_osm(subc, osm[osm["sc_like"]], ref=("parent_lat", "parent_lon"), alt_ref=("hcd_lat", "hcd_lon"))

    inst["dhs_vs_osm_m"] = haversine_m(inst["lat"], inst["lon"], inst["osm_lat"], inst["osm_lon"]).round(0)
    no_dhs = inst["lat"].isna() & inst["osm_lat"].notna()
    inst.loc[no_dhs, ["lat", "lon"]] = inst.loc[no_dhs, ["osm_lat", "osm_lon"]].to_numpy()
    inst["location_from"] = np.select([no_dhs, inst["lat"].notna()], ["OSM", "DHS_2024_25"], "none")

    has_o, has_h, has_p = subc["osm_lat"].notna(), subc["hcd_lat"].notna(), subc["parent_lat"].notna()
    subc["lat"] = np.select([has_o, has_h, has_p], [subc["osm_lat"], subc["hcd_lat"], subc["parent_lat"]], np.nan)
    subc["lon"] = np.select([has_o, has_h, has_p], [subc["osm_lon"], subc["hcd_lon"], subc["parent_lon"]], np.nan)
    subc["location_from"] = np.select([has_o, has_h, has_p], ["OSM", "HCD_2016_name_match", "parent_facility_proxy"], "none")
    subc["osm_vs_hcd_m"] = haversine_m(subc["osm_lat"], subc["osm_lon"], subc["hcd_lat"], subc["hcd_lon"]).round(0)
    subc["km_to_parent"] = (haversine_m(subc["lat"], subc["lon"], subc["parent_lat"], subc["parent_lon"]) / 1000).round(1)
    subc.loc[subc["location_from"].eq("parent_facility_proxy"), "km_to_parent"] = np.nan

    cols_i = ["district", "name", "std_type", "central_category", "section", "listed_in_sections", "address",
              "phone", "rural_urban", "local_body", "health_block", "taluk", "nin", "lat", "lon", "location_from",
              "coord_note", "osm_id", "osm_name", "osm_name_score", "dhs_vs_osm_m", "hcd_2016_vs_dhs_m", "page"]
    cols_s = ["district", "name", "std_type", "address", "parent_facility", "parent_type", "rural_urban",
              "health_block", "local_body", "nin", "population_served", "lat", "lon", "location_from",
              "osm_id", "osm_name", "osm_name_score", "matched_hcd_name", "name_score", "osm_vs_hcd_m",
              "km_to_parent", "page"]
    df = pd.concat([inst[cols_i], subc[cols_s]], ignore_index=True)
    df["type_name"] = df["std_type"].map(TYPE_NAMES)

    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    d = gpd.read_file(ADM2)[["shapeName", "geometry"]].cx[74.5:77.8, 7.9:13.1]
    d["shapeName"] = d["shapeName"].map(district_name)
    d = d[d["shapeName"].isin(KERALA_DISTRICTS)].set_index("shapeName").to_crs(32643)
    pts = g.to_crs(32643).geometry
    g["km_outside_district"] = [round(p.distance(d.geometry[dn]) / 1000, 2) if (has and dn in d.index) else np.nan
                                for p, dn, has in zip(pts, g["district"], g["lat"].notna())]

    flags = (pd.Series(np.where(g["lat"].isna(), "no_location ", ""), index=g.index)
             + np.where(g["km_outside_district"].gt(OUTSIDE_KM), "outside_listed_district ", "")
             + np.where(g["km_to_parent"].gt(PARENT_KM), "far_from_parent_facility ", "")
             + np.where(g["dhs_vs_osm_m"].gt(AGREE_M), "dhs_osm_disagree ", "")
             + np.where(g["location_from"].eq("parent_facility_proxy"), "placed_at_parent_facility ", "")
             + np.where(g["coord_note"].fillna("").str.contains("restored"), "coordinate_repaired", ""))
    g["qa_flags"] = flags.str.strip()
    g["location_quality"] = np.select(
        [g["lat"].isna(), g["qa_flags"].str.contains("outside|far_from|disagree"),
         g["location_from"].eq("parent_facility_proxy"), g["location_from"].eq("HCD_2016_name_match")],
        ["none", "check", "proxy", "approximate"], "good")
    g["osm_confirmed"] = g["dhs_vs_osm_m"].le(AGREE_M) | g["location_from"].eq("OSM")
    g = g.sort_values(["district", "std_type", "name"]).reset_index(drop=True)
    g["facility_id"] = [f"KL_HF_{i:05d}" for i in range(1, len(g) + 1)]
    g["state"], g["service"] = "Kerala", "health"
    g["list_source"] = "DHS Kerala, List of Modern Medicine Institutions 2024-25"
    g = g[["facility_id"] + [c for c in g.columns if c not in ("facility_id", "geometry")] + ["geometry"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g.drop(columns="geometry").to_csv(OUT_DIR / "kerala_health_facilities.csv", index=False, encoding="utf-8")
    g[g["lat"].notna()].to_file(OUT_DIR / "kerala_health_facilities.gpkg", layer="health_facilities", driver="GPKG")
    chk = g[g["qa_flags"].ne("")].drop(columns="geometry").copy()
    chk["google_maps"] = np.where(chk["lat"].notna(), "https://www.google.com/maps?q=" + chk["lat"].astype(str) + "," + chk["lon"].astype(str), "")
    chk["search"] = "https://www.google.com/maps/search/" + (chk["name"] + " " + chk["district"] + " Kerala").str.replace(" ", "+")
    chk[["correct_lat", "correct_lon", "notes"]] = ""
    chk.to_csv(CHECK_CSV, index=False, encoding="utf-8")

    print("facilities:", len(g), "| institutions:", len(inst), "| sub centres:", len(subc))
    print("institutions location:", inst["location_from"].value_counts().to_dict(),
          "| OSM match:", int(inst["osm_id"].notna().sum()))
    print("institutions DHS vs OSM, m:", inst["dhs_vs_osm_m"].describe(percentiles=[.5, .75, .9]).round(0).to_dict())
    print("sub centres location:", subc["location_from"].value_counts().to_dict())
    print("sub centres OSM vs 2016 point, m:", subc["osm_vs_hcd_m"].describe(percentiles=[.5, .75, .9]).round(0).to_dict())
    print("institutions vs 2016 directory (same NIN), m:",
          inst["hcd_2016_vs_dhs_m"].describe(percentiles=[.5, .75, .9]).round(0).to_dict())
    print("sub centre to parent km:", subc["km_to_parent"].describe(percentiles=[.5, .9]).round(1).to_dict())
    print("location quality:", g["location_quality"].value_counts().to_dict())
    print("qa flags:", g["qa_flags"].str.split().explode().value_counts().to_dict())
    print(pd.crosstab(g["type_name"], g["location_quality"], margins=True).to_string())
    print(pd.crosstab(g["district"], g["location_quality"], margins=True).to_string())
    print("written:", OUT_DIR, "and", CHECK_CSV)


if __name__ == "__main__":
    main()
