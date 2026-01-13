"""Export source data for Nature Communications submission.

This script exports all figure and table source data to Excel sheets and parquet
files, then zips them together for submission.

Output:
    - results/source_data/SourceData.xlsx (smaller datasets)
    - results/source_data/cv_obs_vs_pred.parquet (CV data - too large for Excel)
    - results/source_data/spatial_folds.parquet (spatial fold assignments - too large for Excel)
    - results/source_data/SourceData.zip (contains all files)
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_all_trait_models,
    get_cv_splits_dir,
    get_latest_run,
    get_models_dir,
)
from src.utils.plotting_utils import add_human_readables
from src.utils.trait_utils import get_trait_name_from_id
from src.visualization.figures.model_performance import add_biome_names

cfg = get_config()

OUTPUT_DIR = Path("results/source_data")
EXCEL_FILENAME = "SourceData.xlsx"
CV_PARQUET_FILENAME = "cv_obs_vs_pred.parquet"
SPATIAL_FOLDS_PARQUET_FILENAME = "spatial_folds.parquet"
ZIP_FILENAME = "SourceData.zip"
TRAIT_MAPS_ZIP_FILENAME = "trait_maps_1km.zip"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export source data for Nature Communications submission."
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip aggregating CV obs vs pred data (large and slow).",
    )
    return parser.parse_args()


def get_trait_id_mapping(trait_ids: list[str]) -> pd.DataFrame:
    """Create a mapping DataFrame from trait_id to trait_name.

    Strips the "_mean" suffix from trait_ids for cleaner mapping.
    """
    # Strip _mean suffix and deduplicate
    base_trait_ids = sorted(set(
        tid.replace("_mean", "") for tid in trait_ids
    ))
    return pd.DataFrame([
        {"trait_id": tid, "trait_name": get_trait_name_from_id(tid)[0]}
        for tid in base_trait_ids
    ])


def export_all_results() -> pd.DataFrame:
    """Export all_results.parquet, filtered to km resolutions only."""
    log.info("Exporting all_results...")
    df = pd.read_parquet("results/all_results.parquet")
    # Filter to km resolutions only (e.g., "1km", "55km", not "001" or "1")
    df = df[df["resolution"].str.contains("km", na=False)]
    # Add trait_set_abbr but drop trait_name (will be in separate mapping sheet)
    df = df.pipe(add_human_readables)
    # Drop columns we don't need
    cols_to_drop = ["trait_name", "pearsonr_wt"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


def export_all_biome_results() -> pd.DataFrame:
    """Export all_biome_results.parquet with biome names and trait_set_abbr."""
    log.info("Exporting all_biome_results...")
    df = pd.read_parquet("results/all_biome_results.parquet")
    df = df.pipe(add_human_readables).pipe(add_biome_names)
    # Drop columns we don't need
    cols_to_drop = ["trait_name", "pearsonr_wt"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


def export_feature_importance() -> pd.DataFrame:
    """Export all_fi.parquet, filtered to km resolutions only."""
    log.info("Exporting feature_importance...")
    df = pd.read_parquet("results/all_fi.parquet")
    # Filter to km resolutions only (e.g., "1km", "55km", not "001" or "1")
    df = df[df["resolution"].str.contains("km", na=False)]
    # Don't add trait_name (will be in separate mapping sheet)
    return df


def export_spatial_folds(output_path: Path) -> bool:
    """Export spatial CV fold assignments for all traits to parquet.

    Returns True if data was exported, False otherwise.
    """
    log.info("Exporting spatial_folds...")
    splits_dir = get_cv_splits_dir()

    all_splits = []
    for split_file in sorted(splits_dir.glob("*.parquet")):
        trait_id = split_file.stem
        split_df = pd.read_parquet(split_file).assign(trait_id=trait_id)
        all_splits.append(split_df)

    if not all_splits:
        log.warning("No spatial fold files found in %s", splits_dir)
        return False

    df = pd.concat(all_splits, ignore_index=True)
    # Don't add trait_name (will be in separate mapping sheet in Excel)

    log.info("Writing spatial folds to parquet: %s (%d rows)", output_path, len(df))
    df.to_parquet(output_path, index=False)
    return True


def export_splot_gbif_correlation() -> pd.DataFrame:
    """Export sPlot-GBIF correlation data (trait_id only, no trait_name)."""
    log.info("Exporting splot_gbif_correlation...")
    df = pd.read_csv("results/splot_gbif_correlation.csv")
    # Don't add trait_name (will be in separate mapping sheet)
    return df


def aggregate_cv_obs_vs_pred() -> pd.DataFrame:
    """Aggregate all CV observed vs predicted data from model directories.

    This aggregates data at 1km resolution only (main analysis resolution).
    Output columns: x, y, obs, pred, trait_id, trait_set_abbr
    """
    log.info("Aggregating CV obs vs pred data (this may take a while)...")

    all_cv_data = []
    models_dir = get_models_dir()

    trait_set_to_abbr = {"splot": "SCI", "splot_gbif": "COMB", "gbif": "CIT"}

    for trait_dir in sorted(models_dir.glob("X*")):
        trait_id = trait_dir.name

        try:
            latest_run = get_latest_run(trait_dir / cfg.train.arch)
        except (StopIteration, ValueError):
            log.warning("No runs found for trait %s", trait_id)
            continue

        for trait_set_dir in latest_run.iterdir():
            if not trait_set_dir.is_dir():
                continue

            trait_set = trait_set_dir.name
            cv_file = trait_set_dir / "cv_obs_vs_pred.parquet"

            if not cv_file.exists():
                log.warning("CV file not found: %s", cv_file)
                continue

            cv_df = pd.read_parquet(cv_file).assign(
                trait_id=trait_id,
                trait_set_abbr=trait_set_to_abbr.get(trait_set, trait_set),
            )
            all_cv_data.append(cv_df)

    if not all_cv_data:
        log.warning("No CV data found")
        return pd.DataFrame()

    df = pd.concat(all_cv_data, ignore_index=True)

    return df


def create_excel_file(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Create Excel file with multiple sheets."""
    log.info("Creating Excel file: %s", output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if df.empty:
                log.warning("Skipping empty sheet: %s", sheet_name)
                continue
            log.info("  Writing sheet: %s (%d rows)", sheet_name, len(df))
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def create_zip_archive(output_path: Path, files: list[Path]) -> None:
    """Create a zip archive containing the specified files."""
    log.info("Creating zip archive: %s", output_path)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            if file_path.exists():
                log.info("  Adding: %s", file_path.name)
                zipf.write(file_path, file_path.name)
            else:
                log.warning("  File not found, skipping: %s", file_path)


def export_trait_maps(output_path: Path) -> bool:
    """Zip up all 1km trait maps from GBIF and sPlot sources.

    Creates a zip file with structure:
        gbif/X4.tif
        gbif/X14.tif
        ...
        splot/X4.tif
        splot/X14.tif
        ...

    Returns True if data was exported, False otherwise.
    """
    log.info("Exporting 1km trait maps...")

    gbif_dir = Path("data/interim/gbif/trait_maps/Shrub_Tree_Grass/1km")
    splot_dir = Path("data/interim/splot/trait_maps/Shrub_Tree_Grass/1km")

    gbif_files = sorted(gbif_dir.glob("*.tif")) if gbif_dir.exists() else []
    splot_files = sorted(splot_dir.glob("*.tif")) if splot_dir.exists() else []

    if not gbif_files and not splot_files:
        log.warning("No trait map files found")
        return False

    log.info("Found %d GBIF and %d sPlot trait maps", len(gbif_files), len(splot_files))

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in gbif_files:
            arcname = f"gbif/{f.name}"
            log.info("  Adding: %s", arcname)
            zipf.write(f, arcname)

        for f in splot_files:
            arcname = f"splot/{f.name}"
            log.info("  Adding: %s", arcname)
            zipf.write(f, arcname)

    return True


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = cli()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export data for Excel sheets (smaller datasets only)
    sheets = {
        "all_results": export_all_results(),
        "all_biome_results": export_all_biome_results(),
        "feature_importance": export_feature_importance(),
        "splot_gbif_correlation": export_splot_gbif_correlation(),
    }

    # Create trait_id -> trait_name mapping from all_results only
    all_results_trait_ids = list(sheets["all_results"]["trait_id"].unique())
    sheets["trait_id_mapping"] = get_trait_id_mapping(all_results_trait_ids)

    # Create Excel file
    excel_path = OUTPUT_DIR / EXCEL_FILENAME
    create_excel_file(excel_path, sheets)

    files_to_zip = [excel_path]

    # Export spatial folds to parquet (too large for Excel - ~95M rows)
    spatial_folds_path = OUTPUT_DIR / SPATIAL_FOLDS_PARQUET_FILENAME
    if export_spatial_folds(spatial_folds_path):
        files_to_zip.append(spatial_folds_path)

    # Export CV data to parquet (too large for Excel)
    cv_parquet_path = OUTPUT_DIR / CV_PARQUET_FILENAME

    if not args.skip_cv:
        cv_df = aggregate_cv_obs_vs_pred()
        if not cv_df.empty:
            log.info("Writing CV data to parquet: %s (%d rows)", cv_parquet_path, len(cv_df))
            cv_df.to_parquet(cv_parquet_path, index=False)
            files_to_zip.append(cv_parquet_path)
    else:
        log.info("Skipping CV data aggregation (--skip-cv flag)")
        if cv_parquet_path.exists():
            files_to_zip.append(cv_parquet_path)

    # Create zip archive
    zip_path = OUTPUT_DIR / ZIP_FILENAME
    create_zip_archive(zip_path, files_to_zip)

    # Export 1km trait maps as separate zip (not included in main SourceData.zip)
    trait_maps_path = OUTPUT_DIR / TRAIT_MAPS_ZIP_FILENAME
    export_trait_maps(trait_maps_path)

    log.info("Done! Source data exported to: %s", OUTPUT_DIR)
    log.info("Files created:")
    for f in [excel_path, spatial_folds_path, cv_parquet_path, zip_path, trait_maps_path]:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            log.info("  %s (%.2f MB)", f.name, size_mb)


if __name__ == "__main__":
    main()
