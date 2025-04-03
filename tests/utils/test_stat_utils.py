import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.preprocessing import PowerTransformer

from src.utils.stat_utils import power_back_transform


@pytest.fixture
def power_transformer_fn(tmp_path):
    """Create a temporary power transformer file with a real PowerTransformer."""
    import pickle
    
    # Create and fit a real PowerTransformer with realistic data
    X = pd.DataFrame({
        'X1': [1.0, 2.0, 3.0, 4.0],  # Positive skewed data
        'X2': [0.1, 0.5, 2.0, 5.0],  # More skewed data
        'X3': [1.0, 1.2, 1.4, 1.6]   # Less skewed data
    })
    transformer = PowerTransformer(method='yeo-johnson')
    transformer.fit(X)
    
    # Save the transformer
    transformer_file = tmp_path / "power_transformer.pkl"
    with open(transformer_file, 'wb') as f:
        pickle.dump(transformer, f)
    return str(transformer_file)


def test_power_back_transform_basic(power_transformer_fn):
    """Test basic functionality of power back-transform with real transformer."""
    with patch('src.utils.stat_utils.get_power_transformer_fn', return_value=power_transformer_fn):
        # Test with values similar to the training data
        data = np.array([2.0, 3.0, 4.0])
        result = power_back_transform(data, "1")
        
        # Results should be positive and maintain order
        assert np.all(result > 0)
        assert np.all(np.diff(result) > 0)
        
        # Test that transformed values are reasonable
        assert np.all((result >= 0.1) & (result <= 10.0))


def test_power_back_transform_2d(power_transformer_fn):
    """Test power back-transform with 2D array input using real transformer."""
    with patch('src.utils.stat_utils.get_power_transformer_fn', return_value=power_transformer_fn):
        # Create a 2D array
        data = np.array([[2.0, 3.0], [3.0, 4.0]])
        original_shape = data.shape
        flattened = data.ravel()
        
        # Transform the flattened array
        result = power_back_transform(flattened, "1")
        
        # Reshape back to original
        result = result.reshape(original_shape)
        
        # Check shape and properties
        assert result.shape == original_shape
        assert np.all(result > 0)
        # Check that ordering is preserved within rows
        assert np.all(np.diff(result, axis=1) > 0)


def test_power_back_transform_nan(power_transformer_fn):
    """Test power back-transform with NaN values using real transformer."""
    with patch('src.utils.stat_utils.get_power_transformer_fn', return_value=power_transformer_fn):
        data = np.array([2.0, np.nan, 4.0])
        result = power_back_transform(data, "1")
        
        # Check NaN preservation
        assert np.isnan(result[1])
        # Check non-NaN values are transformed
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])
        # Check values are reasonable
        assert result[0] > 0 and result[2] > 0


def test_power_back_transform_errors(power_transformer_fn):
    """Test error handling in power back-transform with real transformer."""
    with patch('src.utils.stat_utils.get_power_transformer_fn', return_value=power_transformer_fn):
        # Test case 1: invalid trait number
        data = np.array([2.0, 3.0, 4.0])
        with pytest.raises(IndexError):
            power_back_transform(data, "999")

        # Test case 2: invalid transformer file
        with patch('src.utils.stat_utils.get_power_transformer_fn', return_value="nonexistent.pkl"):
            with pytest.raises(FileNotFoundError):
                power_back_transform(data, "1")


def test_power_back_transform_range(power_transformer_fn):
    """Test power back-transform with different ranges of values."""
    with patch('src.utils.stat_utils.get_power_transformer_fn', return_value=power_transformer_fn):
        # Test with small values
        small_data = np.array([0.1, 0.2, 0.3])
        small_result = power_back_transform(small_data, "1")
        assert np.all(small_result > 0)  # Should maintain positivity
        
        # Test with large values
        large_data = np.array([10.0, 20.0, 30.0])
        large_result = power_back_transform(large_data, "1")
        assert np.all(large_result > 0)
        
        # Results should maintain relative ordering
        assert np.all(np.diff(small_result) > 0)
        assert np.all(np.diff(large_result) > 0) 