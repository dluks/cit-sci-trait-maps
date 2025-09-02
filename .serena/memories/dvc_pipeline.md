## Major stages (see dvc.yaml)
- build_container
- get_TRY_mean_traits; match_gbif_pfts; extract_splot; standardize_other_products
- harmonize_eo_data; build_predict
- build_gbif_maps; build_splot_maps; build_y
- calculate_spatial_autocorr; build_cv_splits
- train_models; aoa; predict; cov
- build_final_product; aggregate_all_stats; aggregate_aoa
- other_product_splot_correlation

### Examples
```bash
dvc repro train_models
poetry run python src/features/skcv_splits.py -o
```
