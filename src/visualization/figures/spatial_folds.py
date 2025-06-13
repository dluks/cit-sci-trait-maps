"""Visualize spatial cross-validation folds with hex grid overlay."""

from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Polygon

from src.utils.df_utils import reproject_xy_to_geo
from src.utils.plotting_utils import set_font
from src.utils.spatial_utils import acr_to_h3_res, assign_hexagons


def load_autocorr_data(trait: str = "X50_mean") -> float:
    """
    Load autocorrelation data and extract median range for specified trait.

    Parameters:
        trait (str): The trait name to extract median range for.

    Returns:
        float: The median autocorrelation range in meters.
    """
    autocorr_path = Path("reference/spatial_autocorr_1km.parquet")
    autocorr_df = pd.read_parquet(autocorr_path)

    trait_data = autocorr_df[autocorr_df["trait"] == trait]
    if trait_data.empty:
        raise ValueError(f"Trait '{trait}' not found in autocorrelation data")

    median_range = trait_data["median"].iloc[0]
    return median_range


def load_spatial_folds_data(trait: str = "X50_mean") -> pd.DataFrame:
    """
    Load spatial cross-validation folds data.

    Parameters:
        trait (str): The trait name to load folds for.

    Returns:
        pd.DataFrame: DataFrame with x, y, fold columns.
    """
    folds_path = Path(
        f"data/features/Shrub_Tree_Grass/22km/skcv_splits/{trait}.parquet"
    )
    folds_df = pd.read_parquet(folds_path)
    return folds_df


def create_hex_grid(
    points_df: pd.DataFrame, autocorr_range: float, crs: str = "EPSG:6933"
) -> gpd.GeoDataFrame:
    """
    Create hex grid from points data using autocorrelation range.

    Parameters:
        points_df (pd.DataFrame): DataFrame with x, y, fold columns.
        autocorr_range (float): Autocorrelation range in meters.
        crs (str): Coordinate reference system of input points.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with hex polygons and fold assignments.
    """
    # Create copy to avoid modifying original
    df = points_df.copy()

    # Convert coordinates to WGS84 for H3 hexagon assignment
    if crs == "EPSG:6933":
        # Reproject from EPSG:6933 to WGS84
        df_geo = reproject_xy_to_geo(df, from_crs=crs)
        df_with_coords = df.copy()
        df_with_coords["lon"] = df_geo["lon"]
        df_with_coords["lat"] = df_geo["lat"]
    else:
        df_with_coords = df.copy()
        df_with_coords["lon"] = df["x"]
        df_with_coords["lat"] = df["y"]

    # Get H3 resolution from autocorrelation range
    h3_res = acr_to_h3_res(autocorr_range)

    # Assign hexagons using the utility function
    df_with_hex = assign_hexagons(
        df_with_coords, h3_res, lat="lat", lon="lon", dask=False
    )

    # Create hex polygons
    unique_hexes = df_with_hex["hex_id"].unique()
    hex_polygons = []
    hex_ids = []

    for hex_id in unique_hexes:
        # Get hex boundary coordinates
        boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)

        # Create polygon
        polygon = Polygon(boundary)

        # Check if polygon crosses antimeridian (longitude > 150 or < -150)
        x_coords = [x for x, y in boundary]
        crosses_antimeridian = any(x > 150 for x in x_coords) and any(
            x < -150 for x in x_coords
        )

        if not crosses_antimeridian:
            hex_polygons.append(polygon)
            hex_ids.append(hex_id)

    # Create GeoDataFrame
    hex_gdf = gpd.GeoDataFrame(
        {"hex_id": hex_ids, "geometry": hex_polygons}, crs="EPSG:4326"
    )

    # Map fold assignments to hexagons
    hex_fold_mapping = df_with_hex.drop_duplicates("hex_id")[["hex_id", "fold"]]
    hex_gdf = gpd.GeoDataFrame(
        hex_gdf.merge(hex_fold_mapping, on="hex_id", how="left"),
        geometry="geometry",
        crs="EPSG:4326",
    )

    return hex_gdf


def crosses_antimeridian(polygon: Polygon) -> bool:
    """Check if polygon crosses the antimeridian."""
    x_coords = [x for x, y in polygon.exterior.coords]
    return any(x > 150 for x in x_coords) and any(x < -150 for x in x_coords)


def create_spatial_folds_plot(
    points_df: pd.DataFrame,
    hex_gdf: gpd.GeoDataFrame,
    crs: str = "EPSG:6933",
    figsize: tuple = (16, 10),
    dpi: int = 300,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create publication-quality plot of spatial cross-validation folds.

    Parameters:
        points_df (pd.DataFrame): DataFrame with x, y, fold columns.
        hex_gdf (gpd.GeoDataFrame): GeoDataFrame with hex polygons and folds.
        crs (str): Coordinate reference system of input points.
        figsize (tuple): Figure size in inches.
        dpi (int): Figure resolution.
        save_path (Optional[str]): Path to save figure.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Set style for publication quality
    plt.style.use("default")
    sns.set_palette("husl")
    set_font("FreeSans")

    # Create figure with EqualEarth projection
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())

    # Set global extent
    ax.set_global()

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black", alpha=0.7)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="gray", alpha=0.5)
    # ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    # ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.2)

    # Convert points to GeoDataFrame for plotting
    if crs == "EPSG:6933":
        # Reproject points to WGS84 for plotting
        points_geo = reproject_xy_to_geo(points_df, from_crs=crs)
        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_geo["lon"], points_geo["lat"]),
            crs="EPSG:4326",
        )
    else:
        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df["x"], points_df["y"]),
            crs=crs,
        )

    # Get unique folds and create color palette
    unique_folds = sorted(points_df["fold"].unique())
    n_folds = len(unique_folds)

    # Use a modern, visually appealing categorical palette
    colors = sns.color_palette("Set2", n_folds)
    fold_colors = dict(zip(unique_folds, colors))

    # Plot points as small dots first (behind hexagons)
    points_gdf.plot(
        ax=ax,
        markersize=0.4,
        color="darkblue",
        alpha=1,
        edgecolor="none",
        transform=ccrs.PlateCarree(),
    )

    # Plot hexagons colored by fold on top
    hex_gdf.plot(
        ax=ax,
        column="fold",
        categorical=True,
        legend=True,
        legend_kwds={
            "title": "SKCV Fold",
            "loc": "center left",
            "bbox_to_anchor": (1.05, 0.5),
            "frameon": False,
            "facecolor": "none",
            "ncol": 1,
            "fontsize": 14,
            "title_fontsize": 16,
            "markerscale": 2.5,
            "labelspacing": 1.2,
            "handletextpad": 0.8,
        },
        cmap="Set2",
        edgecolor="white",
        linewidth=0.2,
        alpha=0.75,
        transform=ccrs.PlateCarree(),
    )

    # Add gridlines
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.3,
        linestyle="--",
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig


def main():
    """Main function to create spatial folds visualization."""
    try:
        print("Loading autocorrelation data...")
        autocorr_range = load_autocorr_data("X50_mean")
        print(f"Median autocorrelation range for X50_mean: {autocorr_range:.2f} meters")

        print("Loading spatial folds data...")
        points_df = load_spatial_folds_data("X50_mean")
        print(
            f"Loaded {len(points_df)} points with {points_df['fold'].nunique()} folds"
        )

        print("Creating hex grid...")
        hex_gdf = create_hex_grid(points_df, autocorr_range, crs="EPSG:6933")
        print(f"Created {len(hex_gdf)} hexagons")

        print("Creating visualization...")
        fig = create_spatial_folds_plot(
            points_df,
            hex_gdf,
            crs="EPSG:6933",
            figsize=(16, 10),
            dpi=300,
            save_path="results/figures/spatial_folds_visualization.png",
        )

        plt.show()
        print("Visualization complete!")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
