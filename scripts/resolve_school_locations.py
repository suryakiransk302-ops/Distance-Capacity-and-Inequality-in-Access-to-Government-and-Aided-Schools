"""Resolve the schools left missing or doubtful by fill_school_coordinates.py, without manual work.

Candidates per school (all optional):
  sametham    point published on KITE Sametham
  osm_code    OSM school whose ref / schoolcode tag is the school's KITE code
  osm         strict OSM name match from fill_school_coordinates.py
  osm_name    looser OSM name match: same district, type and management, name similarity
              >= LOOSE_NAME_MIN, clearly the best candidate
  udise       UDISE point (UDISE code from the Sametham page), unless it is a placeholder
              (shared by >= PLACEHOLDER_N schools)
  osm_near_*  an OSM school with the school's name within NEAR_M of the Sametham / UDISE point
              (a different OSM feature from the matches above): supports that point
  osm_campus  OSM school of another level with the same name (LP/UP/HS share campuses); last resort
Area test: a candidate is plausible when it lies inside its listed district and inside the area of
its assembly constituency, i.e. the convex hull of the reliably located schools of that
constituency (Sametham/OSM points not under review) buffered by AREA_BUFFER_M.

Decision, first rule that applies:
  1 code_match      osm_code candidate inside the area
  2 sources_agree   two independent candidates within AGREE_M of each other, inside the area
                    (the more trusted one is used: sametham > osm_code > osm > osm_name > udise)
  3 only_in_area    exactly one candidate inside the area
  4 campus_proxy    no candidate inside the area except a same-name school of another level
  5 osm_preferred   sources inside the area disagree and nothing breaks the tie: the strict OSM match
                    is used as the probable point (where UDISE settled a Sametham-vs-OSM disagreement
                    it sided with OSM 123 times and with Sametham 35 times); kept on the manual list
  otherwise the school stays on the manual list (unresolved).
Reviewed decisions (data/raw/schools_manual/school_location_decisions_*.csv) override the rules above
(resolve_rule "reviewed"):
  osm / osm_campus  OSM feature chosen by the researcher (osm_kerala_schoolish_*.json, a wider OSM
                    download incl. school buildings and unnamed schools), checked against locality and PIN
  manual            point located by the researcher in Google Maps (lat, lon)
  sametham          Sametham point (lat, lon) kept: UDISE or locality supports it, only the area test failed
  unresolved        searched, no trustworthy point
Reviewed points that are not "good" stay on the manual list.

Inputs:  data/processed/schools/kerala_schools_govt_aided_located.csv
         data/raw/osm/osm_kerala_schools_2026-10-28.json, osm_kerala_schoolish_2026-11-04.json
         data/raw/schools_manual/school_location_decisions_*.csv
Outputs: FINAL COMPLETE DATASET/schools/kerala_schools_govt_aided_final.{csv,gpkg}
         data/interim/schools/schools_still_to_check.csv
Usage:   python scripts/resolve_school_locations.py
"""

import difflib
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from fill_school_coordinates import DISTRICTS, METRIC_CRS, OSM, core, haversine_m, is_govt, load_osm, school_type

ROOT = Path(__file__).resolve().parents[1]
LOCATED = ROOT / "data" / "processed" / "schools" / "kerala_schools_govt_aided_located.csv"
OUT_DIR = ROOT / "FINAL COMPLETE DATASET" / "schools"
STILL_CSV = ROOT / "data" / "interim" / "schools" / "schools_still_to_check.csv"
REVIEW_DIR = ROOT / "data" / "raw" / "schools_manual"
OSM_WIDE = ROOT / "data" / "raw" / "osm" / "osm_kerala_schoolish_2026-11-04.json"
REVIEW_SOURCES = {"osm": "osm_reviewed", "osm_campus": "osm_campus_reviewed", "manual": "manual_google_maps",
                  "sametham": "sametham_reviewed"}
AGREE_M = 250
AREA_BUFFER_M = 1000
LOOSE_NAME_MIN = 0.8
LOOSE_MARGIN = 0.1
EXACT_NAME = 0.97  # near-identical cleaned name wins when no runner-up is near-identical
PLACEHOLDER_N = 4
OUTSIDE_KM = 1
CAMPUS_NAME_MIN = 0.9
NEAR_M = 150
TRUST = ["sametham", "osm_code", "osm", "osm_name", "udise", "osm_near_sametham", "osm_near_udise", "osm_campus"]
# joined school-type abbreviations: lps, glps, amlps, gwups, ghss, gvhss, ...
ABBREV = re.compile(r"^(govt|g|a|am|gm|gw|m|mt|w|cms|lms)?(lp|up|lps|ups|lpgs|lpbs|hs|hss|vhss|tti)$")


def osm_code_points():
    els = json.loads(OSM.read_text(encoding="utf-8"))["elements"]
    rows = []
    for e in els:
        t = e.get("tags", {})
        code = (t.get("ref") or t.get("schoolcode") or "").strip()
        if code.isdigit() and len(code) == 5:
            rows.append({"school_code_num": code, "lat": e.get("lat", e.get("center", {}).get("lat")),
                         "lon": e.get("lon", e.get("center", {}).get("lon")), "osm_id": f"{e['type']}/{e['id']}"})
    return pd.DataFrame(rows).drop_duplicates("school_code_num", keep=False)


def loose_core(name):
    """core() plus joined abbreviations ('GOVT.L.P.S' -> 'govtlps', 'GLP School', 'AMLPS')."""
    n = re.sub(r"(?i)\b(govt|government|gov)\b\.?", " ", str(name))
    words = [w for w in re.sub(r"[^a-z ]", " ", n.lower().replace(".", "")).split() if not ABBREV.match(w)]
    return core(" ".join(words))


def loose_gov(name):
    return bool(re.search(r"(?i)\b(govt|government|gov)\b|\bg\.? ?[lu]\.? ?p|\bg\.? ?h\.? ?s|\bg\.? ?v\.? ?h|^g\b", str(name)))


def loose_osm_match(s, osm, same_type=True):
    """Best OSM school by cleaned name in the same district and management; clearly better than the
    runner-up. same_type=False finds a school of another level with the same name (shared campus)."""
    c = s.assign(core=s["school_name"].map(loose_core), stype=s["school_name"].map(school_type),
                 gov=s["school_name"].map(loose_gov))
    c = c[c["core"].str.len() >= 4]
    o = osm.assign(ocore=osm["osm_name"].map(loose_core), ogov=osm["osm_name"].map(loose_gov))
    p = c[["school_code_num", "district", "core", "stype", "gov"]].merge(o, left_on="district", right_on="osm_district")
    typ = (p["stype"] == p["otype"]) | p["stype"].isna() | p["otype"].isna()
    p = p[(p["gov"] == p["ogov"]) & (typ if same_type else ~typ)]
    p["score"] = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(p["core"], p["ocore"])]
    p = p[p["score"] >= (LOOSE_NAME_MIN if same_type else CAMPUS_NAME_MIN)].sort_values("score", ascending=False)
    second = p.groupby("school_code_num")["score"].transform(lambda x: x.iloc[1] if len(x) > 1 else 0)
    exact = p["score"].ge(EXACT_NAME) & second.lt(EXACT_NAME)
    p = p[(p["score"] - second >= LOOSE_MARGIN) | exact].drop_duplicates("school_code_num")
    return p.set_index("school_code_num")[["osm_id", "osm_name", "osm_lat", "osm_lon", "score"]]


def osm_near(points, names, osm):
    """OSM school with the school's name within NEAR_M of a candidate point (independent support)."""
    o = gpd.GeoDataFrame(osm, geometry=gpd.points_from_xy(osm["osm_lon"], osm["osm_lat"]), crs=4326).to_crs(METRIC_CRS)
    o["ocore"] = o["osm_name"].map(loose_core)
    pts = points.dropna()
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs=4326).to_crs(METRIC_CRS)
    j = gpd.sjoin(g, o[["osm_id", "ocore", "osm_lat", "osm_lon", "geometry"]], predicate="dwithin", distance=NEAR_M)
    j["score"] = [difflib.SequenceMatcher(None, loose_core(names[i]), b).ratio() for i, b in zip(j.index, j["ocore"])]
    j = j[j["score"] >= LOOSE_NAME_MIN].sort_values("score", ascending=False)
    j = j[~j.index.duplicated()]
    return j[["osm_id", "osm_lat", "osm_lon"]]


def constituency_areas(s):
    ok = s[s["location_source"].isin(["sametham", "osm"]) & ~s["needs_review"]]
    g = gpd.GeoDataFrame(ok, geometry=gpd.points_from_xy(ok["lon"], ok["lat"]), crs=4326).to_crs(METRIC_CRS)
    return g.dissolve(["district", "assembly"]).convex_hull.buffer(AREA_BUFFER_M)


def reviewed_decisions(todo_codes):
    """Researcher decisions with their points (OSM feature centre or manual lat/lon)."""
    files = sorted(REVIEW_DIR.glob("school_location_decisions_*.csv"))
    if not files:
        return pd.DataFrame(columns=["decision", "lat", "lon", "accuracy", "reason"])
    r = pd.concat([pd.read_csv(f, dtype={"school_code_num": str}) for f in files])
    r = r.drop_duplicates("school_code_num", keep="last").set_index("school_code_num")
    els = json.loads(OSM_WIDE.read_text(encoding="utf-8"))["elements"]
    pts = {f"{e['type']}/{e['id']}": (e.get("lat", e.get("center", {}).get("lat")),
                                      e.get("lon", e.get("center", {}).get("lon"))) for e in els}
    is_osm = r["decision"].str.startswith("osm")
    bad = r[is_osm & ~r["osm_id"].isin(pts.keys())].index.tolist() + r.index[~r.index.isin(todo_codes)].tolist()
    if bad:
        raise SystemExit(f"Reviewed schools with unknown OSM id or not on the manual list: {bad}")
    r.loc[is_osm, "lat"] = r.loc[is_osm, "osm_id"].map(lambda i: pts[i][0])
    r.loc[is_osm, "lon"] = r.loc[is_osm, "osm_id"].map(lambda i: pts[i][1])
    return r


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = pd.read_csv(LOCATED, dtype={"school_code_num": str, "udise_code": str, "phone": str, "pin_code": str},
                    low_memory=False)
    s["needs_review"] = s["needs_review"].astype(str).eq("True")
    todo = s["needs_review"] | s["lat"].isna()
    districts = gpd.read_file(DISTRICTS)[["shapeName", "geometry"]].cx[74.5:77.8, 7.9:13.1].rename(
        columns={"shapeName": "district"})

    # candidate points for the schools to resolve
    t = s[todo].set_index("school_code_num")
    cand = {"sametham": t[["lat", "lon"]].where(t["location_source"].eq("sametham"))}
    code = osm_code_points().set_index("school_code_num")
    cand["osm_code"] = code[["lat", "lon"]].reindex(t.index)
    cand["osm"] = t[["osm_lat", "osm_lon"]].set_axis(["lat", "lon"], axis=1)
    osm = load_osm(districts)
    loose = loose_osm_match(s[todo], osm)
    cand["osm_name"] = loose[["osm_lat", "osm_lon"]].set_axis(["lat", "lon"], axis=1).reindex(t.index)
    shared = s.groupby(["udise_lat", "udise_lon"])["school_code_num"].transform("size")
    placeholder = pd.Series(shared.ge(PLACEHOLDER_N).to_numpy(), index=s["school_code_num"]).reindex(t.index)
    cand["udise"] = t[["udise_lat", "udise_lon"]].set_axis(["lat", "lon"], axis=1).where(~placeholder.fillna(False))
    # an OSM feature counts once: drop the loose match when it is the strict one
    used = {k: t["osm_id"] for k in ["osm"]}
    cand["osm_name"] = cand["osm_name"].where(loose["osm_id"].reindex(t.index).ne(used["osm"]))
    used["osm_name"] = loose["osm_id"].reindex(t.index)
    for src in ["sametham", "udise"]:
        near = osm_near(cand[src], t["school_name"], osm).reindex(t.index)
        new = near["osm_id"].notna() & near["osm_id"].ne(used["osm"]) & near["osm_id"].ne(used["osm_name"])
        cand[f"osm_near_{src}"] = near[["osm_lat", "osm_lon"]].set_axis(["lat", "lon"], axis=1).where(new)
    campus = loose_osm_match(s[todo], osm, same_type=False)
    cand["osm_campus"] = campus[["osm_lat", "osm_lon"]].set_axis(["lat", "lon"], axis=1).reindex(t.index)

    long = pd.concat({k: v for k, v in cand.items()}, names=["candidate", "school_code_num"]).dropna().reset_index()
    long["district"] = long["school_code_num"].map(t["district"])
    long["assembly"] = long["school_code_num"].map(t["assembly"])
    g = gpd.GeoDataFrame(long, geometry=gpd.points_from_xy(long["lon"], long["lat"]), crs=4326).to_crs(METRIC_CRS)
    dist = districts.dissolve("district").to_crs(METRIC_CRS).geometry
    areas = constituency_areas(s)
    g["in_district"] = [dn in dist.index and p.distance(dist[dn]) <= OUTSIDE_KM * 1000
                        for p, dn in zip(g.geometry, g["district"])]
    g["in_area"] = [bool(ok) and ((dn, a) not in areas.index or areas[(dn, a)].contains(p))
                    for p, dn, a, ok in zip(g.geometry, g["district"], g["assembly"], g["in_district"])]
    long = pd.DataFrame(g.drop(columns="geometry"))
    long["trust"] = long["candidate"].map(TRUST.index)

    decisions = []
    for code_num, c in long.groupby("school_code_num"):
        c = c[c["in_area"]].sort_values("trust")
        campus, c = c[c["candidate"].eq("osm_campus")], c[c["candidate"].ne("osm_campus")]
        own = c[~c["candidate"].str.startswith("osm_near")]
        pick, rule, support = None, "", ""
        if c["candidate"].eq("osm_code").any():
            pick, rule = c[c["candidate"].eq("osm_code")].iloc[0], "code_match"
        else:
            for _, a in own.iterrows():
                d = haversine_m(a["lat"], a["lon"], c["lat"], c["lon"])
                agree = c[(d <= AGREE_M) & c["candidate"].ne(a["candidate"])]
                if len(agree):
                    pick, rule, support = a, "sources_agree", ",".join(agree["candidate"])
                    break
            if pick is None and len(own) == 1:
                pick, rule = own.iloc[0], "only_in_area"
            if pick is None and own.empty and len(campus):
                pick, rule = campus.iloc[0], "campus_proxy"
            if pick is None and own["candidate"].eq("osm").any():
                pick, rule = own[own["candidate"].eq("osm")].iloc[0], "osm_preferred"
        decisions.append({"school_code_num": code_num, "resolve_rule": rule or "unresolved",
                          "resolved_from": "" if pick is None else pick["candidate"], "agreeing": support,
                          "res_lat": np.nan if pick is None else pick["lat"],
                          "res_lon": np.nan if pick is None else pick["lon"],
                          "candidates": ", ".join(f"{r.candidate}{'' if r.in_area else '(out)'}"
                                                  for r in long[long["school_code_num"].eq(code_num)].itertuples())})
    dec = pd.DataFrame(decisions).set_index("school_code_num").reindex(t.index)
    dec["resolve_rule"] = dec["resolve_rule"].fillna("unresolved")
    dec["candidates"] = dec["candidates"].fillna("none")

    s = s.set_index("school_code_num")
    s["pre_resolve_lat"], s["pre_resolve_lon"], s["pre_resolve_source"] = s["lat"], s["lon"], s["location_source"]
    s = s.join(dec)
    ok = s["res_lat"].notna()
    s.loc[ok, "lat"], s.loc[ok, "lon"] = s.loc[ok, "res_lat"].round(7), s.loc[ok, "res_lon"].round(7)
    s.loc[ok, "location_source"] = s.loc[ok, "resolved_from"]
    unres = s["resolve_rule"].eq("unresolved")
    s["location_accuracy"] = np.select(
        [ok & s["resolve_rule"].isin(["code_match", "sources_agree"]), ok & s["resolve_rule"].eq("campus_proxy"),
         ok & s["resolve_rule"].eq("osm_preferred"), ok, unres & s["lat"].notna(), unres],
        ["good", "campus_proxy", "probable", "single_source", "review", "none"], s["location_accuracy"])

    rev = reviewed_decisions(t.index)
    placed = rev.index[rev["lat"].notna()]
    s.loc[rev.index, "resolve_rule"] = "reviewed"
    s.loc[rev.index, "review_note"] = rev["reason"]
    s.loc[rev.index, ["lat", "lon"]] = np.nan
    s.loc[placed, "lat"], s.loc[placed, "lon"] = rev.loc[placed, "lat"].round(7), rev.loc[placed, "lon"].round(7)
    s.loc[rev.index, "location_source"] = rev["decision"].map(REVIEW_SOURCES).fillna("none")
    s.loc[rev.index, "resolved_from"] = rev["decision"].map(REVIEW_SOURCES).fillna("")
    s.loc[rev.index, "location_accuracy"] = rev["accuracy"].fillna("none")
    s["resolve_moved_m"] = haversine_m(s["lat"], s["lon"], s["pre_resolve_lat"], s["pre_resolve_lon"]).round(0)
    s = s.drop(columns=["res_lat", "res_lon"]).reset_index()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT_DIR / "kerala_schools_govt_aided_final.csv", index=False, encoding="utf-8")
    pts = s[s["lat"].notna()]
    gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs=4326).to_file(
        OUT_DIR / "kerala_schools_govt_aided_final.gpkg", layer="schools", driver="GPKG")
    still = s[s["resolve_rule"].isin(["unresolved", "osm_preferred"])
              | (s["resolve_rule"].eq("reviewed") & s["location_accuracy"].ne("good"))].copy()
    for c, la, lo in [("current_point_map", "lat", "lon"), ("osm_point_map", "osm_lat", "osm_lon"),
                      ("udise_point_map", "udise_lat", "udise_lon")]:
        still[c] = np.where(still[la].notna(), "https://www.google.com/maps?q=" + still[la].astype(str) + ","
                            + still[lo].astype(str), "")
    still["search"] = "https://www.google.com/maps/search/" + (still["school_name"] + " " + still["district"]
                                                               + " Kerala").str.replace(" ", "+")
    still[["correct_lat", "correct_lon", "decision"]] = ""
    still[["district", "school_code_num", "school_name", "assembly", "resolve_rule", "pre_resolve_source", "note",
           "review_note", "candidates", "current_point_map", "osm_point_map", "udise_point_map", "search", "correct_lat",
           "correct_lon", "decision"]].sort_values(["district", "school_name"]).to_csv(
        STILL_CSV, index=False, encoding="utf-8-sig")

    r = s[s["resolve_rule"].notna()]
    print("schools to resolve:", len(r), "| missing:", int(r["pre_resolve_lat"].isna().sum()),
          "| doubtful:", int(r["pre_resolve_lat"].notna().sum()))
    print("rule:", r["resolve_rule"].value_counts().to_dict())
    print("resolved from:", r["resolved_from"].replace("", "-").value_counts().to_dict())
    print(pd.crosstab(r["pre_resolve_source"], r["resolve_rule"]).to_string())
    print("moved (m) where a point existed:", r.loc[r["pre_resolve_lat"].notna() & r["lat"].notna(), "resolve_moved_m"]
          .describe(percentiles=[.5, .9]).round(0).to_dict())
    print("located:", int(s["lat"].notna().sum()), "of", len(s), "| accuracy:", s["location_accuracy"].value_counts().to_dict())
    print("written:", OUT_DIR / "kerala_schools_govt_aided_final.csv", "and", STILL_CSV)


if __name__ == "__main__":
    main()
