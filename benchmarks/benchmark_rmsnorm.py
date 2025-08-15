"""
Point of this file is to compare the custom RMSNorm i implemented within my
cuda implementation against the Pytorch baseline
"""

import torch
import torch.nn as nn
import time
import time
import numpy as np
from rmsnorm.rmsnorm_layer import CustomRMSNorm

def benchmark_rmsnorm():
    """Benchmark custom RMSNorm vs PyTorch RMSNorm"""
    
    # Test configurations
    configs = [
        {"batch": 8, "seq_len": 512, "dim": 4096},    # Small
        {"batch": 4, "seq_len": 2048, "dim": 4096},   # Medium  
        {"batch": 2, "seq_len": 4096, "dim": 4096},   # Large
        {"batch": 1, "seq_len": 8192, "dim": 4096},   # Very Large
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config in configs:
        print(f"\n=== Config: B={config['batch']}, S={config['seq_len']}, D={config['dim']} ===")
        
        # Create test data
        x = torch.randn(config['batch'], config['seq_len'], config['dim'], 
                       device=device, dtype=torch.float32)
        
        # Initialize layers
        pytorch_rmsnorm = nn.RMSNorm(config['dim'], eps=1e-6).to(device)
        custom_rmsnorm = CustomRMSNorm(config['dim'], eps=1e-6).to(device)
        
        # Copy weights to ensure fair comparison
        custom_rmsnorm.weight.data.copy_(pytorch_rmsnorm.weight.data)
        
        # Warmup
        for _ in range(10):
            _ = pytorch_rmsnorm(x)
            _ = custom_rmsnorm(x)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch RMSNorm
        start_time = time.time()
        for _ in range(100):
            pytorch_output = pytorch_rmsnorm(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 100
        
        # Benchmark Custom RMSNorm
        start_time = time.time()
        for _ in range(100):
            custom_output = custom_rmsnorm(x)
        torch.cuda.synchronize()
        custom_time = (time.time() - start_time) / 100
        
        # Correctness check
        max_diff = torch.max(torch.abs(pytorch_output - custom_output)).item()
        relative_error = (max_diff / torch.max(torch.abs(pytorch_output)).item())
        
        # Results
        speedup = pytorch_time / custom_time
        print(f"PyTorch RMSNorm: {pytorch_time*1000:.2f} ms")
        print(f"Custom RMSNorm:  {custom_time*1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Relative error: {relative_error:.2e}")
        print(f"Correctness: {'✓ PASS' if relative_error < 1e-5 else '✗ FAIL'}")

if __name__ == "__main__":
    benchmark_rmsnorm()
