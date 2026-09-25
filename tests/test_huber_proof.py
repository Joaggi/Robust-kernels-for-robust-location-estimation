import numpy as np
import pytest
from robust_kernels.huber_proof import huber_kernel

def test_huber_kernel_small_vectors():
    vector1 = np.array([[1, 0]])
    vector2 = np.array([[0, 1]])
    result = huber_kernel(vector1, vector2, c=2.0)
    assert result[0, 0] == pytest.approx(-0.5)

def test_huber_kernel_medium_vectors():
    vector1 = np.array([[100, 50, 0]])
    vector2 = np.array([[0, 50, 100]])
    result = huber_kernel(vector1, vector2, c=200.0)
    assert result[0, 0] == pytest.approx(-5000.0)

def test_huber_kernel_large_vectors():
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    result = huber_kernel(vector1, vector2, c=50000.0)
    assert result[0, 0] == pytest.approx(-500000000.0)

def test_huber_kernel_exceeds_c():
    vector1 = np.array([[-10000, 10000, -10000, 10000, -10000]])
    vector2 = np.array([[10000, -10000, 10000, -10000, 10000]])
    result = huber_kernel(vector1, vector2, c=10000.0)
    # The linear branch retains the square root, requiring tolerance
    assert result[0, 0] == pytest.approx(-198606797.75, rel=1e-5)
