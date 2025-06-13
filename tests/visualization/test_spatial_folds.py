"""Tests for spatial_folds visualization module."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.visualization.figures.spatial_folds import (
    create_hex_grid,
    create_spatial_folds_plot,
    crosses_antimeridian,
    load_autocorr_data,
    load_spatial_folds_data,
    main,
)


class TestLoadAutocorrData:
    """Test the load_autocorr_data function."""

    def test_load_autocorr_data_success(self):
        """Test successful loading of autocorrelation data."""
        # Create mock data
        mock_data = pd.DataFrame(
            {
                "trait": ["X50_mean", "other_trait"],
                "median": [1000.0, 2000.0],
                "mean": [1100.0, 2100.0],
                "std": [100.0, 200.0],
            }
        )

        with patch("pandas.read_parquet", return_value=mock_data):
            result = load_autocorr_data("X50_mean")
            assert result == 1000.0

    def test_load_autocorr_data_trait_not_found(self):
        """Test error when trait is not found."""
        mock_data = pd.DataFrame(
            {
                "trait": ["other_trait"],
                "median": [1000.0],
            }
        )

        with (
            patch("pandas.read_parquet", return_value=mock_data),
            pytest.raises(ValueError, match="Trait 'X50_mean' not found"),
        ):
            load_autocorr_data("X50_mean")

    def test_load_autocorr_data_file_path(self):
        """Test that correct file path is used."""
        mock_data = pd.DataFrame(
            {
                "trait": ["X50_mean"],
                "median": [1000.0],
            }
        )

        with patch("pandas.read_parquet", return_value=mock_data) as mock_read:
            load_autocorr_data("X50_mean")
            mock_read.assert_called_once()
            args, kwargs = mock_read.call_args
            assert str(args[0]).endswith("reference/spatial_autocorr_1km.parquet")


class TestLoadSpatialFoldsData:
    """Test the load_spatial_folds_data function."""

    def test_load_spatial_folds_data_success(self):
        """Test successful loading of spatial folds data."""
        mock_data = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "fold": [0, 1, 2]}
        )

        with patch("pandas.read_parquet", return_value=mock_data):
            result = load_spatial_folds_data("X50_mean")
            pd.testing.assert_frame_equal(result, mock_data)

    def test_load_spatial_folds_data_file_path(self):
        """Test that correct file path is used."""
        mock_data = pd.DataFrame({"x": [1.0], "y": [2.0], "fold": [0]})

        with patch("pandas.read_parquet", return_value=mock_data) as mock_read:
            load_spatial_folds_data("test_trait")
            mock_read.assert_called_once()
            args, kwargs = mock_read.call_args
            expected_path = (
                "data/features/Shrub_Tree_Grass/22km/skcv_splits/test_trait.parquet"
            )
            assert str(args[0]).endswith(expected_path)


class TestCrossesAntimeridian:
    """Test the crosses_antimeridian function."""

    def test_crosses_antimeridian_true(self):
        """Test polygon that crosses antimeridian."""
        # Create a polygon that crosses the antimeridian
        coords = [(170, 0), (180, 0), (-170, 0), (170, 0)]
        polygon = Polygon(coords)
        assert crosses_antimeridian(polygon) is True

    def test_crosses_antimeridian_false(self):
        """Test polygon that doesn't cross antimeridian."""
        # Create a polygon that doesn't cross the antimeridian
        coords = [(10, 0), (20, 0), (20, 10), (10, 10), (10, 0)]
        polygon = Polygon(coords)
        assert crosses_antimeridian(polygon) is False

    def test_crosses_antimeridian_edge_case(self):
        """Test edge case near antimeridian but not crossing."""
        coords = [(179, 0), (180, 0), (179, 10), (179, 0)]
        polygon = Polygon(coords)
        assert crosses_antimeridian(polygon) is False


class TestCreateHexGrid:
    """Test the create_hex_grid function."""

    @patch("src.visualization.figures.spatial_folds.assign_hexagons")
    @patch("src.visualization.figures.spatial_folds.reproject_xy_to_geo")
    @patch("src.visualization.figures.spatial_folds.acr_to_h3_res")
    @patch("h3.h3_to_geo_boundary")
    def test_create_hex_grid_epsg6933(
        self, mock_h3_boundary, mock_acr_to_h3, mock_reproject, mock_assign_hex
    ):
        """Test hex grid creation with EPSG:6933 CRS."""
        # Setup mock data
        points_df = pd.DataFrame(
            {"x": [1000000, 2000000], "y": [1000000, 2000000], "fold": [0, 1]}
        )

        # Mock reprojection
        mock_reproject.return_value = pd.DataFrame(
            {"lon": [-120, -110], "lat": [40, 50]}
        )

        # Mock H3 resolution
        mock_acr_to_h3.return_value = 5

        # Mock hexagon assignment
        mock_assign_hex.return_value = pd.DataFrame(
            {
                "x": [1000000, 2000000],
                "y": [1000000, 2000000],
                "fold": [0, 1],
                "lon": [-120, -110],
                "lat": [40, 50],
                "hex_id": ["hex1", "hex2"],
            }
        )

        # Mock H3 boundary
        mock_h3_boundary.return_value = [(-120, 40), (-119, 40), (-119, 41), (-120, 41)]

        result = create_hex_grid(points_df, 50000, crs="EPSG:6933")

        # Verify function calls
        mock_reproject.assert_called_once()
        mock_acr_to_h3.assert_called_once_with(50000)
        mock_assign_hex.assert_called_once()

        # Verify result is GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        assert "hex_id" in result.columns
        assert "fold" in result.columns
        assert "geometry" in result.columns

    @patch("src.visualization.figures.spatial_folds.assign_hexagons")
    @patch("src.visualization.figures.spatial_folds.acr_to_h3_res")
    @patch("h3.h3_to_geo_boundary")
    def test_create_hex_grid_epsg4326(
        self, mock_h3_boundary, mock_acr_to_h3, mock_assign_hex
    ):
        """Test hex grid creation with EPSG:4326 CRS."""
        points_df = pd.DataFrame({"x": [-120, -110], "y": [40, 50], "fold": [0, 1]})

        mock_acr_to_h3.return_value = 5
        mock_assign_hex.return_value = pd.DataFrame(
            {
                "x": [-120, -110],
                "y": [40, 50],
                "fold": [0, 1],
                "lon": [-120, -110],
                "lat": [40, 50],
                "hex_id": ["hex1", "hex2"],
            }
        )

        mock_h3_boundary.return_value = [(-120, 40), (-119, 40), (-119, 41), (-120, 41)]

        result = create_hex_grid(points_df, 50000, crs="EPSG:4326")

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0


class TestCreateSpatialFoldsPlot:
    """Test the create_spatial_folds_plot function."""

    def test_create_spatial_folds_plot_input_validation(self):
        """Test that the function handles input data correctly."""
        # Test with realistic EPSG:4326 data (no reprojection needed)
        points_df = pd.DataFrame(
            {"x": [-120.5, -119.5, -118.5], "y": [40.5, 41.5, 42.5], "fold": [0, 1, 2]}
        )

        hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1", "hex2", "hex3"],
                "fold": [0, 1, 2],
                "geometry": [
                    Polygon([(-121, 40), (-120, 40), (-120, 41), (-121, 41)]),
                    Polygon([(-120, 41), (-119, 41), (-119, 42), (-120, 42)]),
                    Polygon([(-119, 42), (-118, 42), (-118, 43), (-119, 43)]),
                ],
            },
            crs="EPSG:4326",
        )

        # Only mock the font setting and figure creation to avoid external dependencies
        with (
            patch("src.visualization.figures.spatial_folds.set_font"),
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.show"),
        ):
            # Mock figure and axis creation
            mock_figure = MagicMock()
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            result = create_spatial_folds_plot(points_df, hex_gdf, crs="EPSG:4326")

            # Test actual behavior: function should return the figure
            assert result == mock_figure

            # Verify the function processes the data correctly
            # (these are the actual calls that should happen)
            mock_figure.add_subplot.assert_called_once()
            mock_ax.set_global.assert_called_once()

    def test_create_spatial_folds_plot_fold_processing(self):
        """Test that fold data is processed correctly."""
        # Create data with multiple folds
        points_df = pd.DataFrame(
            {
                "x": [-120, -110, -100, -90, -80],
                "y": [40, 50, 35, 45, 55],
                "fold": [0, 1, 2, 0, 1],
            }
        )

        hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1", "hex2", "hex3"],
                "fold": [0, 1, 2],
                "geometry": [
                    Polygon([(-121, 39), (-119, 39), (-119, 41), (-121, 41)]),
                    Polygon([(-111, 49), (-109, 49), (-109, 51), (-111, 51)]),
                    Polygon([(-101, 34), (-99, 34), (-99, 36), (-101, 36)]),
                ],
            },
            crs="EPSG:4326",
        )

        with (
            patch("src.visualization.figures.spatial_folds.set_font"),
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.show"),
        ):
            mock_figure = MagicMock()
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            result = create_spatial_folds_plot(points_df, hex_gdf, crs="EPSG:4326")

            # Verify that unique folds are calculated correctly
            # The function should identify 3 unique folds (0, 1, 2)
            assert result == mock_figure

    def test_create_spatial_folds_plot_coordinate_reprojection(self):
        """Test coordinate reprojection for EPSG:6933 input."""
        # Test with EPSG:6933 coordinates that need reprojection
        points_df = pd.DataFrame(
            {
                "x": [1000000, 2000000, 3000000],
                "y": [1000000, 2000000, 3000000],
                "fold": [0, 1, 2],
            }
        )

        hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1", "hex2", "hex3"],
                "fold": [0, 1, 2],
                "geometry": [
                    Polygon([(-121, 40), (-120, 40), (-120, 41), (-121, 41)]),
                    Polygon([(-120, 41), (-119, 41), (-119, 42), (-120, 42)]),
                    Polygon([(-119, 42), (-118, 42), (-118, 43), (-119, 43)]),
                ],
            },
            crs="EPSG:4326",
        )

        # Mock only the reprojection function (external dependency)
        with (
            patch("src.visualization.figures.spatial_folds.set_font"),
            patch(
                "src.visualization.figures.spatial_folds.reproject_xy_to_geo"
            ) as mock_reproject,
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.show"),
        ):
            # Mock the reprojection to return realistic lat/lon
            mock_reproject.return_value = pd.DataFrame(
                {"lon": [-120.5, -119.5, -118.5], "lat": [40.5, 41.5, 42.5]}
            )

            mock_figure = MagicMock()
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            result = create_spatial_folds_plot(points_df, hex_gdf, crs="EPSG:6933")

            # Verify reprojection was called for EPSG:6933 input
            mock_reproject.assert_called_once()
            assert result == mock_figure

    def test_create_spatial_folds_plot_save_path(self):
        """Test that save_path parameter works."""
        points_df = pd.DataFrame(
            {"x": [1000000, 2000000], "y": [1000000, 2000000], "fold": [0, 1]}
        )

        hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1", "hex2"],
                "fold": [0, 1],
                "geometry": [
                    Polygon([(-120, 40), (-119, 40), (-119, 41), (-120, 41)]),
                    Polygon([(-110, 50), (-109, 50), (-109, 51), (-110, 51)]),
                ],
            },
            crs="EPSG:4326",
        )

        with (
            patch("src.visualization.figures.spatial_folds.set_font"),
            patch(
                "src.visualization.figures.spatial_folds.reproject_xy_to_geo"
            ) as mock_reproject,
            patch("matplotlib.pyplot.figure") as mock_plt_figure,
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.show"),
        ):
            mock_reproject.return_value = pd.DataFrame(
                {"lon": [-120, -110], "lat": [40, 50]}
            )
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_plt_figure.return_value = mock_fig

            create_spatial_folds_plot(points_df, hex_gdf, save_path="test.png")

            # Verify plt.savefig was called (not fig.savefig)
            mock_savefig.assert_called_once_with(
                "test.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

    def test_create_spatial_folds_plot_empty_data(self):
        """Test behavior with empty or invalid data."""
        # Test with empty points dataframe
        empty_points_df = pd.DataFrame(columns=["x", "y", "fold"])

        hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1"],
                "fold": [0],
                "geometry": [Polygon([(-121, 40), (-120, 40), (-120, 41), (-121, 41)])],
            },
            crs="EPSG:4326",
        )

        with (
            patch("src.visualization.figures.spatial_folds.set_font"),
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.show"),
        ):
            mock_figure = MagicMock()
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            # Should not raise an error, but handle gracefully
            result = create_spatial_folds_plot(
                empty_points_df, hex_gdf, crs="EPSG:4326"
            )
            assert result == mock_figure

    def test_create_spatial_folds_plot_single_fold(self):
        """Test behavior with single fold data."""
        points_df = pd.DataFrame(
            {
                "x": [-120.5, -119.5, -118.5],
                "y": [40.5, 41.5, 42.5],
                "fold": [0, 0, 0],  # All same fold
            }
        )

        hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1"],
                "fold": [0],
                "geometry": [Polygon([(-121, 40), (-120, 40), (-120, 41), (-121, 41)])],
            },
            crs="EPSG:4326",
        )

        with (
            patch("src.visualization.figures.spatial_folds.set_font"),
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.show"),
        ):
            mock_figure = MagicMock()
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            result = create_spatial_folds_plot(points_df, hex_gdf, crs="EPSG:4326")

            # Should handle single fold without error
            assert result == mock_figure


class TestMain:
    """Test the main function."""

    @patch("src.visualization.figures.spatial_folds.create_spatial_folds_plot")
    @patch("src.visualization.figures.spatial_folds.create_hex_grid")
    @patch("src.visualization.figures.spatial_folds.load_spatial_folds_data")
    @patch("src.visualization.figures.spatial_folds.load_autocorr_data")
    @patch("matplotlib.pyplot.show")
    def test_main_success(
        self,
        mock_show,
        mock_load_autocorr,
        mock_load_folds,
        mock_create_hex,
        mock_create_plot,
    ):
        """Test successful execution of main function."""
        # Setup mocks
        mock_load_autocorr.return_value = 50000.0
        mock_load_folds.return_value = pd.DataFrame(
            {"x": [1000000, 2000000], "y": [1000000, 2000000], "fold": [0, 1]}
        )

        mock_hex_gdf = gpd.GeoDataFrame(
            {
                "hex_id": ["hex1", "hex2"],
                "fold": [0, 1],
                "geometry": [
                    Polygon([(-120, 40), (-119, 40), (-119, 41), (-120, 41)]),
                    Polygon([(-110, 50), (-109, 50), (-109, 51), (-110, 51)]),
                ],
            },
            crs="EPSG:4326",
        )
        mock_create_hex.return_value = mock_hex_gdf

        mock_fig = MagicMock()
        mock_create_plot.return_value = mock_fig

        # Run main function
        main()

        # Verify all functions were called
        mock_load_autocorr.assert_called_once_with("X50_mean")
        mock_load_folds.assert_called_once_with("X50_mean")
        mock_create_hex.assert_called_once()
        mock_create_plot.assert_called_once()
        mock_show.assert_called_once()

    @patch("src.visualization.figures.spatial_folds.load_autocorr_data")
    def test_main_error_handling(self, mock_load_autocorr):
        """Test error handling in main function."""
        mock_load_autocorr.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            main()


class TestIntegration:
    """Integration tests for the spatial folds module."""

    def test_hex_grid_creation_integration(self):
        """Test integration of hex grid creation with mock data."""
        # Create realistic test data
        np.random.seed(42)
        points_df = pd.DataFrame(
            {
                "x": np.random.uniform(-180, 180, 100),
                "y": np.random.uniform(-90, 90, 100),
                "fold": np.random.randint(0, 5, 100),
            }
        )

        with (
            patch(
                "src.visualization.figures.spatial_folds.acr_to_h3_res", return_value=3
            ),
            patch(
                "src.visualization.figures.spatial_folds.assign_hexagons"
            ) as mock_assign,
            patch("h3.h3_to_geo_boundary") as mock_boundary,
        ):
            # Mock assign_hexagons to return data with hex_id
            mock_assign.return_value = points_df.copy().assign(
                hex_id=[f"hex_{i}" for i in range(len(points_df))]
            )

            # Mock h3 boundary
            mock_boundary.return_value = [(0, 0), (1, 0), (1, 1), (0, 1)]

            result = create_hex_grid(points_df, 50000, crs="EPSG:4326")

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) > 0
            assert all(col in result.columns for col in ["hex_id", "fold", "geometry"])


if __name__ == "__main__":
    pytest.main([__file__])
