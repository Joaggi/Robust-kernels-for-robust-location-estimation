import numpy as np
import pytest
from robust_kernels.cauchy_proof import cauchy_kernel

def test_cauchy_kernel_small_vectors():
    vector1 = np.array([[1, 0]])
    vector2 = np.array([[0, 1]])
    result = cauchy_kernel(vector1, vector2, c=2.0)
    assert result[0, 0] == pytest.approx(-0.81093, rel=1e-5)

def test_cauchy_kernel_medium_vectors():
    vector1 = np.array([[100, 50, 0]])
    vector2 = np.array([[0, 50, 100]])
    result = cauchy_kernel(vector1, vector2, c=200.0)
    assert result[0, 0] == pytest.approx(-8109.30216, rel=1e-5)

def test_cauchy_kernel_large_vectors():
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    result = cauchy_kernel(vector1, vector2, c=50000.0)
    assert result[0, 0] == pytest.approx(-734728562.99, rel=1e-5)