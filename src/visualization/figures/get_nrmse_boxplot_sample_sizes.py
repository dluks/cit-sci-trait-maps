"""Extract sample sizes for nRMSE by biome box plots in model-performance figure.

This script retrieves the sample sizes (n) for each box in the nRMSE by biome
subplot of results/figures/model-performance.png.

Each box plot shows the distribution of nRMSE values across traits for a given
biome and trait_set. The sample size (n) is the number of traits examined.
"""

import pandas as pd

from src.utils.plotting_utils import add_human_readables
from src.visualization.figures.model_performance import add_biome_names


def main():
    # Load and process exactly as the figure does
    biome_results = (
        pd.read_parquet("results/all_biome_results.parquet")
        .assign(base_trait_id=lambda df: df.trait_id.str.split("_").str[0])
        .pipe(add_human_readables)
        .pipe(add_biome_names)
        .query("transform == 'power'")
    )

    # Count number of traits per (biome, trait_set) - this is the sample size
    sample_sizes = (
        biome_results.groupby(["biome_name", "trait_set_abbr"])
        .size()
        .unstack(fill_value=0)
    )

    # Reorder columns to match TRAIT_SET_ORDER (SCI, CIT, COMB)
    col_order = ["SCI", "CIT", "COMB"]
    sample_sizes = sample_sizes[[c for c in col_order if c in sample_sizes.columns]]

    # Output results
    print("\nSample sizes (number of traits) for nRMSE by biome box plots:")
    print("=" * 60)
    print(sample_sizes.to_string())
    print("=" * 60)

    output_path = "results/figures/nrmse_biome_sample_sizes.csv"
    sample_sizes.to_csv(output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
