# Distance, Capacity and Inequality in Access to Government and Aided Schools

**A spatial big data analysis of India and Kerala**

[![Open the dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ttryxitsqtwbskb3xtucx9.streamlit.app)

**Live dashboard:** https://ttryxitsqtwbskb3xtucx9.streamlit.app

MSc Data Science with Geoinformatics · Spatial Big Data Analytics project · Digital University Kerala
Author: **Suryakiran M S**  · Submitted to: Dr. Manoj Kumar T.K. · November 2026

---

## What this project asks

The main question of this project is whether the government can give education to every child today through its own schools.
For this, **every 100 m populated cell in India** (140 million cells) and every **Government and Aided school**
(1.07 million schools) are studied, and two different problems are measured:

- **Too far:** there is no school of that level within the official distance norm, i.e.
  1 km for primary, 3 km for upper primary, 5 km for secondary and 7 km for higher secondary.
- **Too full:** a school is present nearby, but it does not have enough seats for all the children around it
  (measured as seats per child with the help of the E2SFCA method).


India is the big data part; **Kerala** is a detailed case study with the most complete school data.

## Main results

| | India | Kerala |
|---|---|---|
| Primary children beyond the 1 km norm | **12.4 million (16.0 %)** | 19.3 % (lower primary) |
| Seats per child, primary / lower primary | 1.10 | 1.32 |
| Seats per child, higher secondary | **0.40** (about 4 seats for 10 children) | 0.82 |
| Gini of seats per child, primary | 0.60 | 0.61 (lower primary) |



**Distance is the main problem at primary level, whereas capacity is the main problem at higher secondary level.** Bihar has the fewest seats (0.56 for primary and 0.13 for higher secondary).
- **Most of the inequality lies inside the states:** only 9 % of the primary-level inequality (Theil) lies between the states, whereas at higher secondary it is 43 %.
- **Kerala** is close to universal access above lower primary; the gap is mainly at lower primary in the hill districts (Idukki 44 % and Wayanad 42 % beyond 1 km).
- **Patterns:** 952 sub-districts are found in Gi\* hot spots of children beyond 1 km and 606 in cold spots of few seats; K-Means gives 5 area types (silhouette 0.415); FP-Growth shows that the seat shortages occur together (confidence 0.96, lift 4.0).
- **411 priority areas** are identified in Kerala (i.e. connected 100 m cells beyond the norm with at least 50 children), and 360 of them are for lower primary.
- **Big data:** Spark + Sedona ran the India E2SFCA state by state in about 5 hours (17.7 GB of results) on a single laptop. For small data, plain Python is faster, whereas from Kerala size upwards Spark is found to be about 1.8× faster.


The full report is in [`REPORT/School_Access_Report.pdf`](REPORT/School_Access_Report.pdf).

## Interactive dashboard (Streamlit)

**Open it online:** https://ttryxitsqtwbskb3xtucx9.streamlit.app

Pages *Explore map*, *Key findings* and *About the data*; filters for India / Kerala, state, area level, school level
and measure; click an area or search for it to read what was found there.

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

It can also be deployed on Streamlit Community Cloud (main file: `streamlit_app/app.py`). The app reads the results
written by notebook 29 (`FINAL COMPLETE DATASET/results/dashboard/school_access_dashboard_data.json`).

## Method in brief


1. **Demand:** the WorldPop 2026 population at 100 m is split into school ages and multiplied by the share of pupils in Government + Aided schools (UDISE).
2. **Supply (seats):** the number of seats is calculated as teachers × the official pupils-per-teacher norm (RTE, Samagra Shiksha, Kerala Education Rules); for higher secondary, it is taken from the HSCAP and VHSCAP batches. A second run is also done using the pupils enrolled at present.
3. **Catchment and distance weight:** the straight-line distance is measured on the ellipsoid, and a Gaussian weight is used which is 1 at the school and 0 at the norm.
4. **E2SFCA** (Luo & Qi, 2009): the seats of each school are shared among the children around it, and then each cell adds up the seats which it can reach.
5. **Areas and inequality:** the results are weighted by children and calculated for states, districts, sub-districts, Kerala taluks and local bodies; inequality is measured with Gini and Theil (between/within states).
6. **Patterns:** Getis-Ord Gi\* hot spots, K-Means area types, FP-Growth rules and Kerala priority areas.
7. **Decision support:** an interactive map dashboard is built with Streamlit.
8. **Benchmarks:** Spark is compared with plain Python setup , based on data size and number of cores.


## Repository layout

```
notebooks/                    32 Jupyter notebooks, run in order (00 -> 31), outputs saved
scripts/                      helper modules used by the notebooks (data collection, cleaning)
scripts/bigdata/              Spark / Sedona helpers (session, population cells, capacity, inequality, benchmark job)
FINAL COMPLETE DATASET/
  schools/                    verified Kerala school layer (11,768 schools) and the Kerala capacity file
  results/schools/            result tables: by state, district, sub-district, Kerala local body; inequality; patterns
  results/benchmarks/         benchmark results and charts
  results/dashboard/          the dashboard data (JSON) read by the Streamlit app
streamlit_app/                Streamlit version of the dashboard (app.py, requirements.txt)
REPORT/                       final report (PDF) and its figures
data/                         not included: raw downloads and the data lake are created by the notebooks
requirements.txt
```

## Notebooks

| # | Notebook | Step |
|---|---|---|
| 00 | `00_data_sources_index` | Catalogue of every dataset: source, date, licence |
| 01 | `01_schools_sametham_list` | Kerala Government + Aided school list with locations (Sametham) |
| 02 | `02_schools_sametham_details` | Kerala school pages: pupils, teachers, infrastructure |
| 03–06 | health and fire collection | Health facilities and fire stations (collected; not used in the final school analysis) |
| 07 | `07_osm_downloads` | OpenStreetMap snapshots used for checks |
| 08 | `08_boundaries` | States and districts (geoBoundaries) |
| 09 | `09_schools_capacity` | Kerala school capacity file |
| 10 | `10_osm_india_schools` | India school layer: UDISE locations corrected with OpenStreetMap |
| 11 | `11_worldpop` | WorldPop 2026 population and age bands, 100 m |
| 12 | `12_friction_surface` | Travel-time surface for health (collected; not used) |
| 13 | `13_boundaries_subdistricts` | Sub-districts (geoBoundaries ADM3) |
| 14 | `14_kerala_roads_local_bodies` | Kerala local bodies and roads (OpenStreetMap) |
| 15 | `15_spark_setup` | **D0** Spark + Sedona setup and tests |
| 16 | `16_data_lake` | **D1** Bronze / silver / gold data lake (Parquet, GeoParquet) |
| 17 | `17_population_cells` | **D2** 1.21 billion raster cells to 140 million populated cells with their areas |
| 18 | `18_quality_checks` | **D3** 107 data quality checks |
| 19–20 | `19_poi_sources`, `20_fix_shared_points` | **D3b** Location fixes for shared points, validated on Kerala |
| 21 | `21_partition_benchmark` | **D4** Spatial join methods compared (broadcast index join chosen) |
| 22–23 | `22_hscap_hss_seats`, `23_vhscap_vhse_seats` | Higher-secondary and VHSE seats (Kerala) |
| 24 | `24_school_capacity` | Seats per school and level (norm-based and enrolment) |
| 25 | `25_school_access_kerala` | **D5** E2SFCA and nearest school, Kerala |
| 26 | `26_school_access_india` | **D5** E2SFCA and nearest school, India (state by state, restartable) |
| 27 | `27_school_inequality` | **D6** Area results, data-coverage flags, Gini and Theil |
| 28 | `28_school_patterns` | **D7** Gi\*, K-Means, FP-Growth, Kerala priority areas |
| 29 | `29_school_dashboard` | **D8** Dashboard data for the Streamlit app |
| 30 | `30_benchmarks` | **D10** Spark vs plain Python, data size and cores |
| 31 | `31_report_figures` | Figures of the report |

## How to run

**Requirements:** Python 3.12, Java 17 (for Spark), about 30 GB free disk and 16 GB RAM for the full India run.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Run the notebooks in order. Each one downloads or builds what it needs into `data/` and checks its own results.

- **UDISE school file:** the *Basic details of schools* file has to be downloaded from the  Data Portal and saved as
  `data/raw/udise_basic_details/basic-details-of-schools.csv` before running notebook 10.
- **Windows:** notebook 15 downloads `winutils.exe` / `hadoop.dll` into `tools/hadoop/`, whereas notebook 29 downloads Leaflet into `tools/leaflet/`.
- **Google Colab:** upload the repository folder to Google Drive (`My Drive`), then open a notebook and run it; the first cell mounts the Drive and installs the required packages.
- **Notebook 05** (fire stations, which is not a part of the school analysis) needs a Google Maps key in the environment variable `GOOGLE_MAPS_API_KEY`.
- **Run time:** the Kerala steps take only a few minutes, whereas the India E2SFCA (notebook 26) takes about 5 hours on 6 cores.


**Not included because of size:** the raw downloads and the data lake (many GB) and the India school layer
`FINAL COMPLETE DATASET/schools/india_schools_govt_aided.csv` (309 MB); notebook 10 rebuilds it.

## Data sources and licences

| Data | Source | Licence / terms |
|---|---|---|
| School list (India) | UDISE, India Data Portal (retrieved January 2022) | Open government data, per the portal's terms |
| Kerala schools, pupils, teachers | KITE Sametham portal (2026-27) | Public government data |
| Higher-secondary and VHSE seats | HSCAP and VHSCAP 2026, Government of Kerala | Public government data |
| Population | WorldPop R2025A, 100 m, 2026 | CC BY 4.0 |
| Boundaries | geoBoundaries (ADM1–ADM3) | CC BY 4.0 |
| Kerala local bodies, school locations for checks | © OpenStreetMap contributors | ODbL |

Results are research estimates, not official government statistics. Distances are straight-line, so the share of
children beyond the norm is a minimum; seats are estimated from teachers and norms.

## Licence

The code in this repository is released under the MIT Licence (see [`LICENSE`](LICENSE)).
The data keep the licences of their sources listed above.

## Citation

Suryakiran M S (2026). *Distance, Capacity and Inequality in Access to Government and Aided Schools: A Spatial Big Data
Analysis of India and Kerala.* MSc project, Digital University Kerala.
