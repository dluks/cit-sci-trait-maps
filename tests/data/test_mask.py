"""Tests for the mask module."""

import numpy as np
import xarray as xr

from src.data.mask import mask_raster


def test_mask_raster():
    """Test the mask_raster function."""
    # Create sample data
    rast_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask_data = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    rast = xr.DataArray(rast_data)
    mask = xr.DataArray(mask_data)

    # Apply mask
    masked_rast = mask_raster(rast, mask)

    # Check masked values
    expected_result = np.array([[1, np.nan, 3], [np.nan, 5, np.nan], [7, np.nan, 9]])
    np.testing.assert_array_equal(
        masked_rast.values,
        expected_result,  # pyright: ignore[reportArgumentType]
    )
