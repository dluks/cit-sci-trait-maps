# Source Data: Integrating Citizen Science and Professional Surveys for Global Plant Trait Mapping

This dataset contains the source data, model outputs, and supplementary materials for the study on integrating citizen science (GBIF) and professional survey (sPlot) data for global plant trait mapping.

## Dataset Contents

### SourceData.zip (692 MB)
A compressed archive containing the main source data files:

#### SourceData.xlsx
An Excel workbook with 5 sheets:

1. **all_results** (1,110 rows): Model performance metrics across all traits, resolutions, and data sources
   - Resolutions: 1km, 22km, 55km, 111km, 222km
   - Trait sets: SCI (sPlot), CIT (GBIF), COMB (combined)
   - Metrics: Pearson's r, R², RMSE, nRMSE, MAE, MedAE

2. **all_biome_results** (777 rows): Per-biome model performance and uncertainty metrics
   - 7 biomes (Boreal, Desert, Mediterranean, Temperate, Tropical, Tundra, Montane)
   - Includes mean coefficient of variation (COV) and area of applicability (AOA) fraction

3. **feature_importance** (137,678 rows): Permutation-based feature importance scores
   - 155 environmental predictor variables
   - Importance scores with standard deviations and p-values

4. **splot_gbif_correlation** (185 rows): Correlation between sPlot and GBIF sparse trait grids
   - Pearson correlation coefficients at each resolution

5. **trait_id_mapping** (37 rows): Mapping from trait IDs to human-readable names

#### spatial_folds.parquet (180 MB)
Spatial cross-validation fold assignments for all 37 traits (~95.6 million location-trait combinations).
- Columns: x, y, fold, trait_id
- Coordinates in EPSG:6933 (World Equidistant Cylindrical)

#### cv_obs_vs_pred.parquet (566 MB)
Cross-validation observed vs. predicted values (~35.6 million observations).
- Columns: x, y, obs, pred, trait_id, trait_set_abbr
- Used for generating observed vs. predicted scatter plots

### trait_maps_1km.zip (7.2 GB)
1-km resolution sparse community-weighted mean (CWM) trait maps derived from:
- **gbif/**: 37 GeoTIFF files from GBIF citizen science observations
- **splot/**: 37 GeoTIFF files from sPlot vegetation survey data

Each GeoTIFF contains 6 bands:
1. Mean trait value
2. Standard deviation
3. Median
4. 5th percentile
5. 95th percentile
6. Observation count

Coordinate reference system: EPSG:6933 (World Equidistant Cylindrical)

## Traits Included (37 total)

| ID | Trait Name |
|----|------------|
| X4 | Stem specific density (SSD) |
| X6 | Rooting depth |
| X13 | Leaf carbon content |
| X14 | Leaf nitrogen content (mass-based) |
| X15 | Leaf phosphorus content |
| X21 | Stem diameter |
| X26 | Seed mass |
| X27 | Seed length |
| X46 | Leaf thickness |
| X47 | Leaf dry matter content (LDMC) |
| X50 | Leaf nitrogen content (area-based) |
| X55 | Leaf dry mass |
| X78 | Leaf delta 15N |
| X95 | Seed germination rate |
| X138 | Seed number |
| X144 | Leaf length |
| X145 | Leaf width |
| X146 | Leaf C/N ratio |
| X163 | Leaf fresh mass |
| X169 | Stem conduit density |
| X223 | Chromosome number |
| X224 | Chromosome cDNA content |
| X237 | Dispersal unit length |
| X281 | Stem conduit diameter |
| X282 | Conduit element length |
| X289 | Wood fiber lengths |
| X297 | Wood ray density |
| X351 | Seed number (per dispersal unit) |
| X614 | Specific root length (fine roots) |
| X1080 | Specific root length (SRL) |
| X3106 | Plant height |
| X3107 | Plant height (generative) |
| X3112 | Leaf area (TRY ID 3112) |
| X3113 | Leaf area |
| X3114 | Leaf area (TRY ID 3114) |
| X3117 | Specific leaf area (SLA) |
| X3120 | Leaf water content |

## Data Sources

- **sPlot**: Global vegetation plot database (Bruelheide et al., 2019)
- **GBIF**: Global Biodiversity Information Facility occurrence records
- **TRY**: Plant trait database (Kattge et al., 2020)

## File Formats

- `.xlsx`: Microsoft Excel Open XML Format (readable with Excel, LibreOffice, pandas)
- `.parquet`: Apache Parquet columnar format (readable with pandas, R arrow, etc.)
- `.tif`: Cloud Optimized GeoTIFF (readable with GDAL, rasterio, QGIS, etc.)

## Usage

```python
import pandas as pd

# Read Excel sheets
all_results = pd.read_excel('SourceData.xlsx', sheet_name='all_results')

# Read parquet files
cv_data = pd.read_parquet('cv_obs_vs_pred.parquet')
spatial_folds = pd.read_parquet('spatial_folds.parquet')

# Read trait maps
import rasterio
with rasterio.open('gbif/X14.tif') as src:
    leaf_n_mean = src.read(1)  # Band 1: mean
    leaf_n_count = src.read(6)  # Band 6: observation count
```

## License

[Specify license here]

## Citation

[Specify citation here]

## Contact

[Specify contact information here]
