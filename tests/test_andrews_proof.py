import numpy as np
import pytest
from robust_kernels.andrews_proof import andrews_kernel 

def test_andrews_kernel_small_vectors():
    """Test 1: Small vectors matching hardcoded independent calculation."""
    vector1 = np.array([[1, 0]])
    vector2 = np.array([[0, 1]])
    
    result = andrews_kernel(vector1, vector2, c=2.0)
    assert result[0, 0] == pytest.approx(-0.2397554, rel=1e-5)

def test_andrews_kernel_medium_vectors():
    """Test 2: Medium vectors matching hardcoded independent calculation."""
    vector1 = np.array([[100, 50, 0]])
    vector2 = np.array([[0, 50, 100]])
    
    result = andrews_kernel(vector1, vector2, c=200.0)
    assert result[0, 0] == pytest.approx(-23.97554, rel=1e-5)

def test_andrews_kernel_large_vectors():
    """Test 3: Large vectors matching hardcoded independent calculation."""
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    
    result = andrews_kernel(vector1, vector2, c=50000.0)
    assert result[0, 0] == pytest.approx(-9346.010, rel=1e-3)

def test_andrews_kernel_exceeds_pi():
    """Test 4: Verifies the kernel returns -c when r/c > pi."""
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    
    c = 10000.0
    result = andrews_kernel(vector1, vector2, c=c)
    assert result[0, 0] == -10000.0