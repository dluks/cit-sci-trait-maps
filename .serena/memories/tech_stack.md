## Languages & runtimes
- Python 3.11 (Poetry-managed)

## Core libraries
- Modeling: AutoGluon Tabular (LightGBM focus), scikit‑learn, statsmodels
- Geo: rasterio, rioxarray, geopandas, cartopy, rio‑cogeo
- Parallel: Dask (+ dask‑geopandas), joblib
- Spatial: H3, PyKrige
- Data: pandas, numpy, pyarrow
- Viz: matplotlib, seaborn, bokeh

## Tooling
- Workflow/data: DVC
- Containerization: Singularity/Apptainer (definition: `cit-sci-traits.def`)
- Packaging: Poetry (`pyproject.toml`)
- Testing: pytest
- Type checking: pyright
- Linting: pylint (ruff optional if added)
- Notebooks: Jupyter (organized under `notebooks/`)