import numpy as np
import pytest
from robust_kernels.tukey_proof import tukey_kernel 

def test_tukey_kernel_small_vectors():
    """Test 1: Small vectors with distance sqrt(2)."""
    vector1 = np.array([[1, 0]])
    vector2 = np.array([[0, 1]])
    
    # Using c=2.0, the manual calculation yields exactly 0.0625
    result = tukey_kernel(vector1, vector2, c=2.0)
    assert result[0, 0] == pytest.approx(0.0625)

def test_tukey_kernel_medium_vectors():
    """Test 2: Medium vectors with distance sqrt(20000)."""
    vector1 = np.array([[100, 50, 0]])
    vector2 = np.array([[0, 50, 100]])
    
    # Using c=200.0, the manual calculation yields exactly 0.0625
    result = tukey_kernel(vector1, vector2, c=200.0)
    assert result[0, 0] == pytest.approx(0.0625)

def test_tukey_kernel_large_vectors():
    """Test 3: Large vectors with distance sqrt(2,000,000,000)."""
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    
    # Using c=50000.0, the manual calculation yields exactly 0.004
    result = tukey_kernel(vector1, vector2, c=50000.0)
    assert result[0, 0] == pytest.approx(0.004)

def test_tukey_kernel_exceeds_c():
    """Test 4: Verifies the kernel returns 0 when distance > c."""
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    
    # The distance is ~44,721. If we set c=40,000, distance > c, so result must be 0.
    result = tukey_kernel(vector1, vector2, c=40000.0)
    assert result[0, 0] == 0.0