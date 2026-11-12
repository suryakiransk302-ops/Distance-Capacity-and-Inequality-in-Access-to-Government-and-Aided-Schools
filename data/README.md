# data/

This folder is filled by the notebooks and is not stored in the repository (it is many GB):

- `data/raw/` — downloads (WorldPop rasters, boundaries, OpenStreetMap extracts, Sametham pages, HSCAP / VHSCAP files).
  Put the UDISE file here by hand: `data/raw/udise_basic_details/basic-details-of-schools.csv`.
- `data/interim/` — intermediate tables.
- `data/lake/` — the bronze / silver / gold data lake (Parquet and GeoParquet) written by notebooks 16 onwards.
- `data/tmp/` — temporary files.
