"""Geocode Kerala Fire and Rescue stations from the official RTI officers list.

Source list: Kerala Fire and Rescue Services, RTI page
https://fire.kerala.gov.in/rti-ml/  (copied to data/raw/sametham/fire stations list.md)

Providers
---------
nominatim : OpenStreetMap Nominatim. No key. Max 1 request/second (usage policy).
            Results are ODbL and can be stored.
google    : Google Places API (New) Text Search. Needs GOOGLE_MAPS_API_KEY in the
            environment. Google's terms restrict storing lat/lng long term and
            showing them on non-Google maps: use these results to locate stations,
            then verify and record your own coordinates.

Every result is checked: the matched place's name must resemble the station name,
and two stations may not share one place. Anything else is flagged needs_review.

Usage
-----
python scripts/geocode_fire_stations.py --parse-only
python scripts/geocode_fire_stations.py --provider nominatim
python scripts/geocode_fire_stations.py --provider google
"""

import argparse
import datetime as dt
import difflib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC_MD = ROOT / "data" / "raw" / "sametham" / "fire stations list.md"
OUT_DIR = ROOT / "data" / "interim" / "fire_stations"

SOURCE_URL = "https://fire.kerala.gov.in/rti-ml/"
KERALA_BBOX = {"south": 8.1, "north": 12.9, "west": 74.8, "east": 77.5}
NAME_MATCH_MIN = 0.8

# Common spellings used in the search query only; the official name is kept as is.
QUERY_NAME_FIXES = {
    "Neyardam": "Neyyar Dam",
    "Malappuaram": "Malappuram",
    "Kozhikkode": "Kozhikode",
    "Koylandi": "Koyilandy",
    "Chackai": "Chakkai",
    "Seethathode": "Seethathodu",
    "Chengannoor": "Chengannur",
    "Angamali": "Angamaly",
    "Irinjalakkuda": "Irinjalakuda",
    "Thanur": "Tanur",
    "Peerumedu": "Peermade",
    "Kollamkode": "Kollengode",
    "Thrikkaripur": "Trikaripur",
    "Venjarammoodu": "Venjaramoodu",
    "Kalloorkkad": "Kalloorkad",
}


def norm(s):
    """Letters only, upper case: used to match officer names across rows."""
    return re.sub(r"[^A-Z]", "", s.upper().replace("DISTRICTFIREOFFICER", ""))


def parse_stations(md_path=SRC_MD):
    """Return one row per fire and rescue station with its district."""
    rows = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            rows.append(cells + [""] * (4 - len(cells)))

    dfo_to_district, stations, section = {}, [], None
    for sl, office, _spio, appellate in rows:
        label = sl.strip("* ").upper()
        if label in ("DISTRICT FIRE OFFICE", "FIRE AND RESCUE STATIONS"):
            section = label
            continue
        if section == "DISTRICT FIRE OFFICE" and "DISTRICT FIRE AND RESCUE OFFICE" in office:
            district = office.split("OFFICE")[-1].strip(" ,").title()
            dfo_to_district[norm(appellate.replace("DISTRICT FIRE OFFICER", ""))] = district
        elif section == "FIRE AND RESCUE STATIONS" and "FIRE AND RESCUE STATION" in office.upper():
            name = re.sub(r"(?i)^.*FIRE AND RESCUE STATION\s*,\s*", "", office)
            name = re.sub(r"(?i)^STATION\s+", "", name).strip().title()
            key = norm(appellate.replace("<br>", " ").replace("DISTRICT FIRE OFFICER", ""))
            district = next((d for k, d in dfo_to_district.items() if k and key.startswith(k)), None)
            if district is None and stations:  # blank appellate cell: list is ordered by district
                district = stations[-1]["district"]
            stations.append({"sl_no": int(sl), "station_official": office, "station_name": name,
                             "district": district})

    df = pd.DataFrame(stations)
    df["district"] = df["district"].replace({"Malapuram": "Malappuram", "Kasargod": "Kasaragod"})
    df["source_url"] = SOURCE_URL
    return df


def query_name(name):
    return QUERY_NAME_FIXES.get(name, name)


def name_score(station, text):
    """Best fuzzy similarity between the station name and any 1-2 word span of text."""
    target = re.sub(r"[^a-z]", "", query_name(station).lower())
    words = [w for w in re.split(r"[^a-z]+", (text or "").lower()) if w]
    spans = words + ["".join(p) for p in zip(words, words[1:])]
    return max((difflib.SequenceMatcher(None, target, s).ratio() for s in spans), default=0.0)


def search_nominatim(name, district, session):
    """Fire-station search first; fall back to the town (locality level)."""
    url = "https://nominatim.openstreetmap.org/search"
    base = {"format": "jsonv2", "countrycodes": "in", "limit": 5, "bounded": 1,
            "viewbox": f"{KERALA_BBOX['west']},{KERALA_BBOX['north']},{KERALA_BBOX['east']},{KERALA_BBOX['south']}"}
    attempts = [("fire_station", f"fire station {query_name(name)}, {district}, Kerala"),
                ("locality", f"{query_name(name)}, {district}, Kerala")]
    for level, q in attempts:
        time.sleep(1.1)
        r = session.get(url, params={**base, "q": q}, timeout=60)
        r.raise_for_status()
        results = r.json()
        if level == "fire_station":
            results = [x for x in results if x.get("type") == "fire_station"]
        if results:
            return {"query": q, "match_level": level, "raw": results}
    return {"query": attempts[-1][1], "match_level": "none", "raw": []}


def search_google(name, district, session, api_key):
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key,
               "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.types"}
    q = f"Fire and Rescue Station {query_name(name)}, {district}, Kerala"
    body = {"textQuery": q, "regionCode": "IN",
            "locationRestriction": {"rectangle": {
                "low": {"latitude": KERALA_BBOX["south"], "longitude": KERALA_BBOX["west"]},
                "high": {"latitude": KERALA_BBOX["north"], "longitude": KERALA_BBOX["east"]}}}}
    r = session.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    places = r.json().get("places", [])
    fire = [p for p in places if "fire_station" in p.get("types", [])]
    return {"query": q, "match_level": ("fire_station" if fire else "other_place") if places else "none",
            "raw": fire or places}


def pick(entry, station, provider):
    """Choose the candidate whose name best matches the station; return flat fields."""
    if not entry["raw"]:
        return {"name_score": 0.0}
    if provider == "nominatim":
        cands = [{"lat": float(x["lat"]), "lon": float(x["lon"]), "matched_name": x.get("name"),
                  "matched_address": x.get("display_name"), "place_id": f"osm:{x['osm_type']}/{x['osm_id']}"}
                 for x in entry["raw"]]
    else:
        cands = [{"lat": p["location"]["latitude"], "lon": p["location"]["longitude"],
                  "matched_name": p.get("displayName", {}).get("text"),
                  "matched_address": p.get("formattedAddress"), "place_id": f"google:{p['id']}"}
                 for p in entry["raw"]]
    for c in cands:
        c["name_score"] = round(max(name_score(station, c["matched_name"]),
                                    name_score(station, ",".join((c["matched_address"] or "").split(",")[:3]))), 2)
    return max(cands, key=lambda c: c["name_score"])


def run(provider):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = parse_stations()
    cache_path = OUT_DIR / f"cache_{provider}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    session = requests.Session()
    session.headers["User-Agent"] = "msc-spatial-inequality-study/0.1 (academic project)"
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if provider == "google" and not api_key:
        raise SystemExit("Set GOOGLE_MAPS_API_KEY in the environment first.")

    results = []
    for s in stations.itertuples():
        key = f"{query_name(s.station_name)}|{s.district}"
        if key not in cache:
            if provider == "nominatim":
                cache[key] = search_nominatim(s.station_name, s.district, session)
            else:
                cache[key] = search_google(s.station_name, s.district, session, api_key)
            cache[key]["geocoded_on"] = dt.date.today().isoformat()
            cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        e = cache[key]
        results.append({"query": e["query"], "match_level": e["match_level"], "geocoded_on": e["geocoded_on"]}
                       | pick(e, s.station_name, provider))

    out = pd.concat([stations.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    out["geocoder"] = provider
    shared = out["place_id"].notna() & out.duplicated("place_id", keep=False)
    reasons = pd.DataFrame({
        "level": out["match_level"].where(out["match_level"] != "fire_station", ""),
        "name": out["name_score"].lt(NAME_MATCH_MIN).map({True: "name_mismatch", False: ""}),
        "shared": shared.map({True: "shared_place", False: ""}),
    })
    out["review_reason"] = reasons.apply(lambda r: " ".join(x for x in r if x), axis=1)
    out["needs_review"] = out["review_reason"] != ""

    out_csv = OUT_DIR / f"kerala_fire_stations_{provider}.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")
    ok = out[~out["needs_review"]]
    print("match levels:", out["match_level"].value_counts().to_dict())
    print(f"accepted automatically: {len(ok)} of {len(out)} | needs review: {int(out['needs_review'].sum())}")
    print("review reasons:", out.loc[out["needs_review"], "review_reason"].value_counts().to_dict())
    print("written:", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["nominatim", "google"], default="nominatim")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    if a.parse_only:
        df = parse_stations()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "kerala_fire_stations_list.csv", index=False, encoding="utf-8")
        print(df.groupby("district", sort=False).size().to_string())
        print("total:", len(df), "| missing district:", int(df["district"].isna().sum()))
    else:
        run(a.provider)
