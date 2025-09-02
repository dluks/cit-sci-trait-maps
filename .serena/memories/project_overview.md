## Purpose
Generate global, high‑resolution plant trait maps (31 TRY traits) as community‑weighted means by modeling crowdsourced biodiversity observations (GBIF, sPlot, TRY) against Earth observation predictors at ~1 km, with multi‑resolution outputs and publication‑ready rasters and figures.

## Data sources
- GBIF species observations
- sPlot vegetation plots (or sPlotOpen alternative)
- TRY v5/v6 trait data
- EO predictors: canopy height, MODIS reflectance, SoilGrids, VODCA, WorldClim

## Outputs
- Trained models per trait and PFT
- Predictions, coverage, and AOA layers per resolution
- Aggregated analyses and figures in `results/`

## Key configs
- `params.yaml`: parameters, dataset lists, HPC resource knobs
- `dvc.yaml`: pipeline stages and dependencies
- Containerized builds via Singularity/Apptainer