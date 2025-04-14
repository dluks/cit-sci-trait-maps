"""Test the parallel processing of sPlot data with trait-specific options."""

import argparse
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from src.conf.conf import get_config
from src.data.build_splot_maps import main


class TestSplotParallel(unittest.TestCase):
    """Test the build_splot_maps.py script with trait-specific options."""

    def setUp(self):
        """Set up the test environment."""
        self.config = get_config()
        self.traits_to_test = [
            str(self.config.datasets.Y.traits[0])
        ]  # Test with first trait

        # Create a mock arguments object
        class Args(argparse.Namespace):
            def __init__(self):
                super().__init__()
                self.resume = False
                self.trait = self.traits_to_test[0]

        self.args = Args()

    @patch("src.data.build_splot_maps.dd.read_parquet")
    @patch("src.data.build_splot_maps.xr_to_raster")
    @patch("src.data.build_splot_maps.rasterize_points")
    @patch("src.data.build_splot_maps.init_dask")
    def test_single_trait_processing(
        self,
        mock_init_dask,
        mock_rasterize_points,
        mock_xr_to_raster,
        mock_read_parquet,
    ):
        """Test that only the specified trait is processed."""
        # Setup mock client and cluster
        mock_client = MagicMock()
        mock_cluster = MagicMock()
        mock_init_dask.return_value = (mock_client, mock_cluster)

        # Mock dataframes
        mock_header = MagicMock()
        mock_header.pipe.return_value.pipe.return_value.drop.return_value.astype.return_value.set_index.return_value.map_partitions.return_value.drop.return_value = mock_header

        mock_traits = MagicMock()
        mock_traits.pipe.return_value.set_index.return_value = mock_traits

        mock_pfts = MagicMock()
        mock_pfts.pipe.return_value.pipe.return_value.drop.return_value.dropna.return_value.pipe.return_value.drop.return_value.drop_duplicates.return_value.set_index.return_value = mock_pfts

        mock_merged = MagicMock()
        mock_merged.pipe.return_value.dropna.return_value.pipe.return_value.drop.return_value.set_index.return_value.join.return_value.join.return_value.reset_index.return_value.drop.return_value.persist.return_value = mock_merged

        # Mock df and grid operations
        mock_grouped_df = MagicMock()
        mock_df = pd.DataFrame(
            {
                "cwm": np.random.rand(10),
                "cw_std": np.random.rand(10),
                "cw_med": np.random.rand(10),
                "cw_q05": np.random.rand(10),
                "cw_q95": np.random.rand(10),
                "x": np.random.rand(10),
                "y": np.random.rand(10),
            }
        )

        mock_merged.set_index.return_value.groupby.return_value.apply.return_value.join.return_value.reset_index.return_value.compute.return_value = mock_df

        # Mock rasterize functions
        mock_ds = MagicMock()
        mock_rasterize_points.return_value = mock_ds

        # Mock merged grids and xarray operations
        mock_gridded = MagicMock()
        mock_ds.rename.return_value = mock_ds
        xr.merge.return_value = mock_gridded

        # Mock read_parquet to return our mocks
        mock_read_parquet.side_effect = [
            mock_header,
            mock_traits,
            mock_pfts,
            mock_merged,
        ]

        # Run the main function with our mocked args
        main(self.args)

        # Check that traits was called with columns for the specified trait
        # (second call to read_parquet)
        _, trait_call_kwargs = mock_read_parquet.call_args_list[1]
        self.assertIn("columns", trait_call_kwargs)

        expected_columns = ["speciesname", f"X{self.traits_to_test[0]}"]
        self.assertEqual(set(trait_call_kwargs["columns"]), set(expected_columns))

        # Check that the columns list was created with only the specified trait
        # This is harder to test directly, but we can verify that xr_to_raster was called once
        # which means only one trait was processed
        mock_xr_to_raster.assert_called_once()


if __name__ == "__main__":
    unittest.main()
