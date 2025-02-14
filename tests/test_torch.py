import torch
import pytest

@pytest.fixture
def device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    return device

def test_device_availability(device):
    """Test if CUDA is available and device properties"""
    if torch.cuda.is_available():
        assert device.type == "cuda"
        # Test basic CUDA properties
        assert torch.cuda.device_count() > 0
        assert isinstance(torch.cuda.current_device(), int)
        assert isinstance(torch.cuda.get_device_name(), str)
        
        # Get device properties
        props = torch.cuda.get_device_properties(device)
        print(f"\nGPU Properties:")
        print(f"Name: {torch.cuda.get_device_name()}")
        print(f"Total Memory: {props.total_memory/1024**2:.1f} MB")
        print(f"Multi Processors: {props.multi_processor_count}")
        
        assert props.total_memory > 0
        assert props.multi_processor_count > 0
    else:
        print("\nRunning on CPU")
        assert device.type == "cpu"
        
def test_tensor_device_placement(device):
    """Test tensor creation and movement between devices"""
    x = torch.randn(10, 10)
    print(f"\nInitial tensor device: {x.device}")
    assert x.device.type == "cpu"
    
    # Move tensor to target device
    x = x.to(device)
    print(f"Tensor moved to: {x.device}")
    assert x.device.type == device.type
    
def test_dtype_characteristics(device):
    """Test different dtype characteristics"""
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    print("\nTesting different dtypes:")
    for dtype in dtypes:
        x = torch.ones(1, dtype=dtype, device=device)
        print(f"Created tensor with dtype: {dtype}")
        assert x.dtype == dtype
        assert x.device.type == device.type

def print_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

if __name__ == "__main__":
    pytest.main()
    print_device()