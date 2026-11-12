"""Build a browser tool to verify or place every fire station on satellite imagery.

Input is the merged table (scripts/merge_official_locations.py). Stations where the OSM
point and the department's official map agree are pre-marked verified. Each other station
starts at its best candidate: official map position, OSM point, town centre or district
centre. In the browser you confirm the point, drag it onto the fire-station building, or
click the map to place it. "Open official map pin" opens the exact place the department
embedded on its station page. Progress is kept in the browser; "Export CSV" downloads
the verified coordinates.

Output: data/interim/fire_stations/verify_fire_stations.html
Usage:  python scripts/build_fire_station_verifier.py
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FS_DIR = ROOT / "data" / "interim" / "fire_stations"
DISTRICTS = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM2.geojson"

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Fire Station Verifier</title><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{margin:0;font:14px system-ui,sans-serif;display:flex;height:100vh}
 #side{width:360px;display:flex;flex-direction:column;border-right:1px solid #ccc}
 #top{padding:10px;border-bottom:1px solid #ddd} #list{overflow:auto;flex:1}
 #map{flex:1} .row{padding:6px 10px;border-bottom:1px solid #eee;cursor:pointer}
 .row.sel{background:#e8f0fe} .tag{font-size:11px;padding:1px 6px;border-radius:8px;float:right}
 .v{background:#d9f2e3;color:#145a32} .r{background:#fde2e1;color:#8a1c14} .o{background:#eee;color:#444}
 #panel{padding:10px;border-top:1px solid #ddd} button{margin:3px 3px 3px 0;padding:6px 10px;cursor:pointer}
 textarea{width:100%;box-sizing:border-box} .muted{color:#666;font-size:12px}
</style></head><body>
<div id="side">
 <div id="top"><b>Fire station verification</b> <span id="prog" class="muted"></span><br>
  <label><input type="checkbox" id="onlyTodo"> show only not verified</label>
  <button id="export">Export CSV</button></div>
 <div id="list"></div>
 <div id="panel"><div id="info" class="muted">Select a station.</div>
  <button id="pin">Open official map pin</button><button id="page">Official station page</button>
  <button id="gmaps">Search Google Maps</button><br><button id="confirm"><b>Confirm location</b></button>
  <button id="reset">Reset</button>
  <textarea id="notes" rows="2" placeholder="Notes (e.g. signboard visible)"></textarea>
  <div class="muted">Drag the marker or click the map to place it on the station building.
   Reference points: <span style="color:#e66101">&#9679; OSM</span> <span style="color:#2166ac">&#9679; official map</span></div></div>
</div><div id="map"></div>
<script>
const DATA = __DATA__;
const KEY = 'fire_station_verify_v1';
let state = {}; try { state = JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const map = L.map('map').setView([10.5, 76.3], 8);
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 20, maxNativeZoom: 19, attribution: 'Imagery: Esri, Maxar, Earthstar Geographics'}).addTo(map);
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'});
L.control.layers({'Satellite': sat, 'OpenStreetMap': osm}).addTo(map);
const dots = L.layerGroup().addTo(map), refs = L.layerGroup().addTo(map);
let cur = null, marker = null;

const st = d => state[d.sl_no] || (d.auto_method ? {verified: true, method: d.auto_method} : {});
const pos = d => st(d).lat != null ? [st(d).lat, st(d).lon] : [d.lat, d.lon];
function tag(d) {
  if (st(d).verified) return '<span class="tag v">verified</span>';
  return d.needs_review ? '<span class="tag r">check</span>' : '<span class="tag o">OSM match</span>';
}
function renderList() {
  const only = document.getElementById('onlyTodo').checked;
  const n = DATA.filter(d => st(d).verified).length;
  document.getElementById('prog').textContent = `${n} / ${DATA.length} verified`;
  document.getElementById('list').innerHTML = DATA.filter(d => !only || !st(d).verified).map(d =>
    `<div class="row ${cur && cur.sl_no === d.sl_no ? 'sel' : ''}" data-id="${d.sl_no}">${d.sl_no}. ${d.station_name}
     ${tag(d)}<br><span class="muted">${d.district}${d.review_reason ? ' &middot; ' + d.review_reason : ''}</span></div>`).join('');
  document.querySelectorAll('.row').forEach(el => el.onclick = () => select(+el.dataset.id));
  dots.clearLayers();
  DATA.forEach(d => L.circleMarker(pos(d), {radius: 5, weight: 1, fillOpacity: .8,
    color: st(d).verified ? '#1a9850' : (d.needs_review ? '#d73027' : '#fdae61')})
    .on('click', () => select(d.sl_no)).addTo(dots));
}
function select(id) {
  cur = DATA.find(d => d.sl_no === id);
  if (marker) map.removeLayer(marker);
  marker = L.marker(pos(cur), {draggable: true}).addTo(map).on('dragend', e => move(e.target.getLatLng()));
  refs.clearLayers();
  if (cur.osm_lat != null) L.circleMarker([cur.osm_lat, cur.osm_lon], {radius: 9, color: '#e66101', weight: 3, fill: false})
    .bindTooltip('OSM: ' + (cur.matched_name || '')).addTo(refs);
  if (cur.official_lat != null) L.circleMarker([cur.official_lat, cur.official_lon], {radius: 9, color: '#2166ac', weight: 3, fill: false})
    .bindTooltip('Official map (' + cur.official_pos_quality + ')').addTo(refs);
  const placed = st(cur).moved;
  map.setView(pos(cur), cur.start === 'district_centre' && !placed ? 11 : cur.start === 'town_centre' && !placed ? 15 : 18);
  document.getElementById('pin').disabled = !cur.embed_feature_id;
  document.getElementById('page').disabled = !cur.page_url;
  document.getElementById('info').innerHTML = `<b>${cur.sl_no}. ${cur.station_official}</b><br>${cur.district}<br>
    ${cur.official_address ? 'Official address: ' + cur.official_address + '<br>' : ''}
    ${cur.official_phone ? 'Phone: ' + cur.official_phone + '<br>' : ''}
    Start point: ${cur.start}${cur.matched_name ? ' (OSM: ' + cur.matched_name + ')' : ''}<br>
    ${cur.osm_vs_official_m != null ? 'OSM vs official map: ' + cur.osm_vs_official_m + ' m<br>' : ''}
    Status: ${st(cur).verified ? 'verified (' + st(cur).method + ')' : 'not verified'}`;
  document.getElementById('notes').value = st(cur).notes || '';
  renderList();
}
function move(ll) {
  state[cur.sl_no] = {...st(cur), lat: +ll.lat.toFixed(6), lon: +ll.lng.toFixed(6), moved: true, verified: false};
  save(); select(cur.sl_no);
}
map.on('click', e => { if (cur) move(e.latlng); });
document.getElementById('onlyTodo').onchange = renderList;
document.getElementById('gmaps').onclick = () => cur && window.open('https://www.google.com/maps/search/?api=1&query=' +
  encodeURIComponent(`Fire and Rescue Station ${cur.station_name}, ${cur.district}, Kerala`), '_blank');
document.getElementById('pin').onclick = () => cur && cur.embed_feature_id &&
  window.open('https://www.google.com/maps?ftid=' + encodeURIComponent(cur.embed_feature_id), '_blank');
document.getElementById('page').onclick = () => cur && cur.page_url && window.open(cur.page_url, '_blank');
document.getElementById('notes').oninput = e => { if (cur) { state[cur.sl_no] = {...st(cur), notes: e.target.value}; save(); } };
document.getElementById('reset').onclick = () => { if (cur) { delete state[cur.sl_no]; save(); select(cur.sl_no); } };
document.getElementById('confirm').onclick = () => {
  if (!cur) return;
  if (['town_centre', 'district_centre'].includes(cur.start) && !st(cur).moved) {
    alert('This is only a town or district centre. Place the marker on the station first.'); return; }
  const [lat, lon] = pos(cur);
  state[cur.sl_no] = {...st(cur), lat, lon, verified: true, verified_on: new Date().toISOString().slice(0, 10),
    method: st(cur).moved ? 'manually_placed_on_imagery' : cur.start + '_confirmed_on_imagery'};
  save();
  const next = DATA.find(d => !st(d).verified && d.sl_no > cur.sl_no) || DATA.find(d => !st(d).verified);
  next ? select(next.sl_no) : select(cur.sl_no);
};
document.getElementById('export').onclick = () => {
  const cols = ['sl_no','station_official','station_name','district','lat','lon','verified','method','verified_on',
                'notes','start','start_lat','start_lon','osm_lat','osm_lon','osm_place_id','official_lat','official_lon',
                'official_pos_quality','osm_vs_official_m','official_address','official_phone','page_url','source_url'];
  const esc = v => v == null ? '' : /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : String(v);
  const rows = DATA.map(d => { const s = st(d), [lat, lon] = pos(d);
    return {...d, lat, lon, verified: !!s.verified, method: s.method || '', verified_on: s.verified_on || '',
            notes: s.notes || '', start_lat: d.lat, start_lon: d.lon}; });
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => esc(r[c])).join(','))).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  a.download = 'kerala_fire_stations_verified.csv'; a.click();
};
renderList();
</script></body></html>"""


def build():
    df = pd.read_csv(FS_DIR / "kerala_fire_stations_merged.csv")
    districts = gpd.read_file(DISTRICTS)
    districts = districts[districts["shapeName"].isin(df["district"])].cx[74.8:77.5, 8.1:12.9].set_index("shapeName")
    centres = districts.to_crs("EPSG:32643").geometry.centroid.to_crs("EPSG:4326")

    # Start each marker at the most trustworthy candidate; town and district centres must be moved.
    osm_fire = df["match_level"].eq("fire_station")
    official = df["official_lat"].notna()
    conds = [df["auto_method"].eq("osm_and_official_map_agree"),
             df["auto_method"].eq("official_map_pin"),
             df["official_pos_quality"].isin(["standard_zoom_corrected", "close_zoom"]),
             osm_fire & ~df["needs_review"].astype(bool),
             official,
             osm_fire,
             df["match_level"].eq("locality")]
    starts = ["osm_and_official_agree", "official_map_pin", "official_map", "osm_fire_station",
              "official_map_rough", "osm_fire_station_flagged", "town_centre"]
    df["start"] = np.select(conds, starts, "district_centre")
    df["osm_lat"], df["osm_lon"] = df["lat"].where(osm_fire), df["lon"].where(osm_fire)
    use_official = df["start"].isin(["official_map_pin", "official_map", "official_map_rough"])
    df.loc[use_official, "lat"] = df.loc[use_official, "official_lat"]
    df.loc[use_official, "lon"] = df.loc[use_official, "official_lon"]
    no_point = df["lat"].isna() | df["start"].eq("district_centre")
    df.loc[no_point, "lat"] = df.loc[no_point, "district"].map(centres.y)
    df.loc[no_point, "lon"] = df.loc[no_point, "district"].map(centres.x)
    df["osm_place_id"] = df["place_id"]
    df["needs_review"] = ~df["auto_accept"].astype(bool)

    keep = ["sl_no", "station_official", "station_name", "district", "lat", "lon", "matched_name",
            "needs_review", "review_reason", "start", "osm_place_id", "source_url", "auto_method",
            "osm_lat", "osm_lon", "official_lat", "official_lon", "official_pos_quality", "osm_vs_official_m",
            "official_address", "official_phone", "page_url", "embed_feature_id"]
    records = json.loads(df[keep].to_json(orient="records"))
    out = FS_DIR / "verify_fire_stations.html"
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(records, ensure_ascii=False)), encoding="utf-8")
    print("stations:", len(records), "| start points:", df["start"].value_counts().to_dict())
    print("written:", out)


if __name__ == "__main__":
    build()
