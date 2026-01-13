"""Plot observation density comparison between CIT (GBIF) and SCI (sPlot)."""

import argparse
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LogNorm

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import read_trait_map
from src.utils.plotting_utils import set_font

cfg = get_config()

TRAIT_ID = "X14"  # Leaf N (by mass)
REDUCE_FACTOR = 55
DPI = 300


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot observation density comparison figure."
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./results/figures/obs_density.png",
        help="Output file path.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = cli()

    log.info("Loading trait maps...")
    splot = read_trait_map(TRAIT_ID, "splot", band=6)
    gbif = read_trait_map(TRAIT_ID, "gbif", band=6)

    log.info("Coarsening data...")
    splot_coarsened = splot.coarsen(
        x=REDUCE_FACTOR, y=REDUCE_FACTOR, boundary="trim"
    ).sum()
    gbif_coarsened = gbif.coarsen(
        x=REDUCE_FACTOR, y=REDUCE_FACTOR, boundary="trim"
    ).sum()

    log.info("Building figure...")
    set_font("FreeSans")
    sns.set_theme(context="paper", style="ticks", font="FreeSans", font_scale=2)

    fig = plt.figure(figsize=(20, 10), dpi=DPI)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth())
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth())

    for ax, data, title in zip(
        [ax0, ax1], [gbif_coarsened, splot_coarsened], ["CIT", "SCI"]
    ):
        ax.set_title(title, fontsize=20, y=1.05)
        ax.coastlines()
        xx, yy = data.x, data.y
        data.values[data.values == 0] = np.nan
        im = ax.pcolormesh(
            xx,
            yy,
            data.values,
            shading="auto",
            rasterized=True,
            norm=LogNorm(vmin=1),
            cmap="cool",
            transform=ccrs.epsg("6933"),
        )
        obs_type = "GBIF observations" if title == "CIT" else "sPlot vegetation surveys"
        plt.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            shrink=0.7,
            pad=0.03,
            label=f"Number of {obs_type}",
        )

    if args.out_path:
        out_path = Path(args.out_path)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        # Only add dpi for raster formats
        if out_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
            save_kwargs["dpi"] = DPI
        plt.savefig(out_path, **save_kwargs)
        log.info(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
