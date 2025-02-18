import pytest
import numpy as np
import cupy as cp

def test_basic_operations():
    """Test basic CuPy array operations."""
    # Test array creation
    x_cpu = np.array([1, 2, 3, 4, 5])
    x_gpu = cp.array([1, 2, 3, 4, 5])
    assert cp.allclose(x_gpu, cp.asarray(x_cpu))
    
    # Test arithmetic operations
    assert cp.allclose(x_gpu + 1, cp.asarray(x_cpu + 1))
    assert cp.allclose(x_gpu * 2, cp.asarray(x_cpu * 2))
    
    # Test reduction operations
    assert abs(cp.mean(x_gpu).get() - np.mean(x_cpu)) < 1e-6
    assert abs(cp.sum(x_gpu).get() - np.sum(x_cpu)) < 1e-6

def test_matrix_operations():
    """Test matrix operations with CuPy."""
    # Create random matrices
    A = cp.random.rand(10, 10)
    B = cp.random.rand(10, 10)
    
    # Test matrix multiplication
    C_gpu = cp.matmul(A, B)
    C_cpu = np.matmul(cp.asnumpy(A), cp.asnumpy(B))
    assert cp.allclose(C_gpu, cp.asarray(C_cpu))
    
    # Test matrix transpose
    assert cp.allclose(A.T, cp.asarray(cp.asnumpy(A).T))

def test_device_memory():
    """Test device memory management."""
    # Test memory allocation and deallocation
    x = cp.zeros((1000, 1000))
    del x
    # Memory should be freed
    cp.get_default_memory_pool().free_all_blocks()
    
    # Test memory transfer
    x_cpu = np.random.rand(100, 100)
    x_gpu = cp.asarray(x_cpu)
    x_back = cp.asnumpy(x_gpu)
    assert np.allclose(x_cpu, x_back)

if __name__ == "__main__":
    pytest.main([__file__])
