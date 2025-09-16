import matplotlib.pyplot as plt
import pandas as pd


def plot_spatial_overlap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str = "splot",
    df2_name: str = "moreno",
    figsize: tuple[int, int] = (15, 8),
    point_size: float = 0.1,
    alpha: float = 0.8,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot two spatial dataframes to visualize their overlap.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (e.g., splot) with x, y coordinates as index
    df2 : pd.DataFrame
        Second dataframe (e.g., moreno) with x, y coordinates as index
    df1_name : str
        Name of the first dataset for legend
    df2_name : str
        Name of the second dataset for legend
    figsize : Tuple[int, int]
        Figure size for the plot
    point_size : float
        Size of the plotted points
        alpha : float
        Transparency of the points
    dpi : int
        Resolution of the plot in dots per inch

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Get coordinates - assuming x, y are in the index
    df1_coords = df1.index.to_frame()[["x", "y"]].reset_index(drop=True)
    df2_coords = df2.index.to_frame()[["x", "y"]].reset_index(drop=True)

    # Plot all moreno points in grey first (background)
    ax.scatter(
        df2_coords["x"],
        df2_coords["y"],
        c="lightgrey",
        s=point_size,
        alpha=alpha,
        label=f"{df2_name} (background)",
    )

    # Find overlap: splot points that have corresponding moreno data
    # Create a set of moreno coordinates for fast lookup
    moreno_coords_set = set(zip(df2_coords["x"], df2_coords["y"]))

    # Check which splot coordinates are in moreno
    splot_overlap_mask = df1_coords.apply(
        lambda row: (row["x"], row["y"]) in moreno_coords_set, axis=1
    )

    # Plot overlapping splot points in green
    overlap_coords = df1_coords[splot_overlap_mask]
    if len(overlap_coords) > 0:
        ax.scatter(
            overlap_coords["x"],
            overlap_coords["y"],
            c="green",
            s=point_size,
            alpha=alpha,
            label=f"{df1_name} overlap (n={len(overlap_coords)})",
        )

    # Plot non-overlapping splot points in red
    no_overlap_coords = df1_coords[~splot_overlap_mask]
    if len(no_overlap_coords) > 0:
        ax.scatter(
            no_overlap_coords["x"],
            no_overlap_coords["y"],
            c="red",
            s=point_size,
            alpha=alpha,
            label=f"{df1_name} no overlap (n={len(no_overlap_coords)})",
        )

    # Set up the plot
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Overlap: {df1_name} vs {df2_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set aspect ratio to be appropriate for global data
    ax.set_aspect("equal", adjustable="box")

    # Add statistics as text
    total_splot = len(df1_coords)
    total_moreno = len(df2_coords)
    overlap_count = len(overlap_coords) if len(overlap_coords) > 0 else 0
    no_overlap_count = len(no_overlap_coords) if len(no_overlap_coords) > 0 else 0

    stats_text = f"""Dataset Statistics:
{df2_name}: {total_moreno:,} points
{df1_name}: {total_splot:,} points
Overlap: {overlap_count:,} points ({overlap_count / total_splot * 100:.1f}%)
No overlap: {no_overlap_count:,} points ({no_overlap_count / total_splot * 100:.1f}%)"""

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def plot_spatial_overlap_detailed(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str = "splot",
    df2_name: str = "moreno",
    figsize: tuple[int, int] = (20, 10),
    point_size: float = 0.1,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a detailed 3-panel plot showing both datasets and their overlap.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (e.g., splot) with x, y coordinates as index
    df2 : pd.DataFrame
        Second dataframe (e.g., moreno) with x, y coordinates as index
    df1_name : str
        Name of the first dataset
    df2_name : str
        Name of the second dataset
    figsize : Tuple[int, int]
        Figure size for the plot
        point_size : float
        Size of the plotted points
    dpi : int
        Resolution of the plot in dots per inch

    Returns
    -------
    plt.Figure
        The matplotlib figure object with 3 subplots
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # Get coordinates
    df1_coords = df1.index.to_frame()[["x", "y"]].reset_index(drop=True)
    df2_coords = df2.index.to_frame()[["x", "y"]].reset_index(drop=True)

    # Panel 1: Show moreno dataset
    axes[0].scatter(df2_coords["x"], df2_coords["y"], c="grey", s=point_size, alpha=0.6)
    axes[0].set_title(f"{df2_name} Dataset\n({len(df2_coords):,} points)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal", adjustable="box")

    # Panel 2: Show splot dataset
    axes[1].scatter(df1_coords["x"], df1_coords["y"], c="blue", s=point_size, alpha=0.6)
    axes[1].set_title(f"{df1_name} Dataset\n({len(df1_coords):,} points)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="box")

    # Panel 3: Show overlap analysis
    # Plot moreno in grey background
    axes[2].scatter(
        df2_coords["x"], df2_coords["y"], c="lightgrey", s=point_size, alpha=0.4
    )

    # Find overlap
    moreno_coords_set = set(zip(df2_coords["x"], df2_coords["y"]))
    splot_overlap_mask = df1_coords.apply(
        lambda row: (row["x"], row["y"]) in moreno_coords_set, axis=1
    )

    # Plot overlapping and non-overlapping points
    overlap_coords = df1_coords[splot_overlap_mask]
    no_overlap_coords = df1_coords[~splot_overlap_mask]

    if len(overlap_coords) > 0:
        axes[2].scatter(
            overlap_coords["x"],
            overlap_coords["y"],
            c="green",
            s=point_size,
            alpha=0.8,
            label="Overlap",
        )

    if len(no_overlap_coords) > 0:
        axes[2].scatter(
            no_overlap_coords["x"],
            no_overlap_coords["y"],
            c="red",
            s=point_size,
            alpha=0.8,
            label="No overlap",
        )

    overlap_pct = (
        len(overlap_coords) / len(df1_coords) * 100 if len(df1_coords) > 0 else 0
    )
    axes[2].set_title(f"Overlap Analysis\n({overlap_pct:.1f}% overlap)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    return fig
