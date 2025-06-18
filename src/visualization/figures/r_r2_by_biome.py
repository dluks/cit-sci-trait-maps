import argparse
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_biome_mapping
from src.utils.plotting_utils import add_human_readables, set_font

TRAIT_SET_ORDER = ["SCI", "COMB", "CIT"]
tricolor_palette = sns.color_palette(["#b0b257", "#66a9aa", "#b95fa1"])
CFG = get_config()


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot r and R² by biome figure.")
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./results/figures/r-r2-by-biome.png",
        help="Output file path.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Generate r and R² by biome figure."""
    set_font("FreeSans")

    log.info("Loading biome results...")
    biome_res = load_biome_results()

    log.info("Building figure...")
    with sns.plotting_context("paper", 1.5):
        build_figure(biome_res)

    if args is not None:
        log.info("Saving figure...")
        plt.savefig(args.out_path, dpi=300, bbox_inches="tight")

    plt.show()


def load_biome_results() -> pd.DataFrame:
    """Load and process biome results data."""
    drop_cols = [
        "trait_set",
        "run_id",
        "pearsonr_wt",
        "root_mean_squared_error",
        "mean_squared_error",
        "mean_absolute_error",
        "median_absolute_error",
    ]

    # Only keep traits listed in params
    keep_traits = [f"X{t}" for t in CFG.datasets.Y.traits]

    biome_results = (
        pd.read_parquet("results/all_biome_results.parquet")
        .assign(base_trait_id=lambda df: df.trait_id.str.split("_").str[0])
        .query("base_trait_id in @keep_traits")
        .pipe(add_human_readables)
        .pipe(add_biome_names)
        .drop(columns=drop_cols + ["base_trait_id"])
        .query("transform == 'power'")
        .query("resolution == '1km'")
        .assign(r_squared=lambda df: df.pearsonr**2)  # Calculate R²
    )

    return biome_results


def build_figure(biome_res: pd.DataFrame) -> Figure:
    """Build the main figure with r and R² subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=100)

    keep_cols = ["trait_name", "trait_set_abbr", "biome_name"]

    # Left subplot: Pearson's r by biome
    r_by_biome(
        df=biome_res[keep_cols + ["pearsonr"]],
        ax=ax1,
    )

    # Right subplot: R² by biome
    r2_by_biome(
        df=biome_res[keep_cols + ["r_squared"]],
        ax=ax2,
    )

    # Add subplot letters
    letter_size = 12
    x_offset = -0.05
    y_offset = 1.02

    ax1.text(
        x_offset,
        y_offset,
        "a",
        transform=ax1.transAxes,
        fontsize=letter_size,
        verticalalignment="top",
        fontweight="bold",
    )
    ax2.text(
        x_offset,
        y_offset,
        "b",
        transform=ax2.transAxes,
        fontsize=letter_size,
        verticalalignment="top",
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def r_by_biome(df: pd.DataFrame, ax: Axes) -> Axes:
    """Plot Pearson's r by biome."""
    ax = trait_set_by_biome_box_plot(
        df,
        ax,
        metric_col="pearsonr",
        metric_label="Pearson's r",
        order=TRAIT_SET_ORDER,
    )
    # Remove legend from left plot
    ax.legend().remove()
    # Remove top and right borders
    sns.despine(ax=ax)
    return ax


def r2_by_biome(df: pd.DataFrame, ax: Axes) -> Axes:
    """Plot R² by biome."""
    ax = trait_set_by_biome_box_plot(
        df,
        ax,
        metric_col="r_squared",
        metric_label="R²",
        order=TRAIT_SET_ORDER,
    )
    # Remove y-tick labels from right plot
    ax.set_yticklabels([])
    ax.set_ylabel("")
    # Remove top and right borders
    sns.despine(ax=ax)
    return ax


def trait_set_by_biome_box_plot(
    df: pd.DataFrame,
    ax: Axes,
    metric_col: str,
    metric_label: str,
    order: list[str],
    palette: Any | None = tricolor_palette,
    max_x: float | None = None,
    fliersize: int = 0,
) -> Axes:
    """Create boxplot showing trait set performance by biome."""
    ax = sns.boxplot(
        data=df,
        x=metric_col,
        y="biome_name",
        ax=ax,
        hue="trait_set_abbr",
        hue_order=order,
        palette=palette,
        dodge=True,
        fliersize=fliersize,
        linewidth=0.5,
    )
    if max_x is not None:
        ax.set_xlim(right=max_x)

    ax.set_xlabel(metric_label, fontweight="bold")
    ax.set_ylabel("Biome", fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    for patch in ax.get_legend().get_patches():
        patch.set_edgecolor("none")
    ax.legend(
        handles,
        labels,
        title=None,
        loc="upper left",
        bbox_to_anchor=(0.8, 1),
        frameon=False,
    )

    # Add horizontal dashed lines between biome groups
    y_positions = range(len(df["biome_name"].unique()))
    for i in range(len(y_positions) - 1):
        ax.axhline(
            y=y_positions[i] + 0.5,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=0.8,
            zorder=0,
        )

    # Remove top and right borders
    sns.despine(ax=ax)

    return ax


def add_biome_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable biome names to the dataframe."""
    biome_mapping = {int(k): v for k, v in get_biome_mapping().items()}
    return df.pipe(lambda _df: _df.assign(biome_name=_df.biome.map(biome_mapping)))


if __name__ == "__main__":
    main(cli())
