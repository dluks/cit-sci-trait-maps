"""Extract sample sizes for subplot b box plots in spatial-transfer figure.

This script retrieves the sample sizes (n) for each box in subplot b of
results/figures/spatial-transfer.png, which shows the mean coefficient of
variation by biome for SCI and COMB trait sets.

Each box plot shows the distribution of mean_cov values across traits, where
each mean_cov value is computed from pixel-level COV values within a biome.
The sample size (n) for each biome/trait_set is the number of pixels used
to compute those mean_cov values.
"""

import numpy as np
import pandas as pd

from src.utils.dataset_utils import get_biome_map_fn, get_cov_dir
from src.utils.raster_utils import open_raster
from src.utils.trait_utils import get_trait_number_from_id
from src.visualization.figures.model_performance import get_biome_mapping
from src.visualization.figures.spatial_transferability import get_traits_of_interest


def count_pixels_per_biome(cov_da, biome_da) -> dict[int, int]:
    """Count non-NaN COV pixels per biome."""
    cov_vals = cov_da.sel(band=1).values.ravel()
    biome_vals = biome_da.sel(band=1).values.ravel()

    # Create mask for valid (non-NaN) COV pixels
    valid_mask = ~np.isnan(cov_vals)

    # Get biome values where COV is valid
    valid_biomes = biome_vals[valid_mask]

    # Count pixels per biome
    unique, counts = np.unique(valid_biomes[~np.isnan(valid_biomes)], return_counts=True)
    return {int(b): int(c) for b, c in zip(unique, counts)}


def main():
    traits_of_interest = get_traits_of_interest()
    print(f"Traits of interest ({len(traits_of_interest)}): {traits_of_interest}")

    # Get COV file paths (same logic as spatial_transferability.get_cov_filepaths)
    cov_dir = get_cov_dir()
    splot_fns = [
        list((f / "splot").glob("*.tif"))[0]
        for f in sorted(list(cov_dir.iterdir()))
        if f.is_dir() and get_trait_number_from_id(f.stem) in traits_of_interest
    ]
    comb_fns = [
        list((f / "splot_gbif").glob("*.tif"))[0]
        for f in sorted(list(cov_dir.iterdir()))
        if f.is_dir() and get_trait_number_from_id(f.stem) in traits_of_interest
    ]

    print(f"Found {len(splot_fns)} SCI COV files and {len(comb_fns)} COMB COV files")

    # Load biome map once
    biome_da = open_raster(get_biome_map_fn())

    # Count pixels per biome for one representative COV file per trait set
    # (pixel counts should be consistent across traits within same trait_set)
    print("Processing SCI COV file...")
    splot_cov = open_raster(splot_fns[0])
    biome_reproj = biome_da.rio.reproject_match(splot_cov)
    splot_counts = count_pixels_per_biome(splot_cov, biome_reproj)
    splot_cov.close()

    print("Processing COMB COV file...")
    comb_cov = open_raster(comb_fns[0])
    biome_reproj_comb = biome_da.rio.reproject_match(comb_cov)
    comb_counts = count_pixels_per_biome(comb_cov, biome_reproj_comb)
    comb_cov.close()

    biome_da.close()

    # Get biome name mapping
    biome_mapping = {int(k): v for k, v in get_biome_mapping().items()}

    # Build results dataframe
    results = []
    for biome_code in sorted(set(splot_counts.keys()) | set(comb_counts.keys())):
        if biome_code == 0 or biome_code == 98:  # Skip invalid biomes
            continue
        biome_name = biome_mapping.get(biome_code, f"Biome {biome_code}")
        results.append({
            "biome_name": biome_name,
            "SCI": splot_counts.get(biome_code, 0),
            "COMB": comb_counts.get(biome_code, 0),
        })

    sample_sizes = pd.DataFrame(results).set_index("biome_name")

    # Output results
    print("\nSample sizes (pixel counts) for subplot b box plots:")
    print("=" * 60)
    print(sample_sizes.to_string())
    print("=" * 60)

    output_path = "results/figures/subplot_b_sample_sizes.csv"
    sample_sizes.to_csv(output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
