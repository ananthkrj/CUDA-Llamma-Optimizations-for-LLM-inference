import torch
import torch.nn as nn
import time
import math
from rope.rope_layer import CustomRoPE

# add lit llama to path
# include it in this directory
# import llama model
# and import llama utils

class PyTorchRoPE(nn.Module):
    """Reference PyTorch RoPE implementation for comparison"""
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs_grid = torch.outer(t, freqs)
        
        self.register_buffer('cos_cached', freqs_grid.cos())
        self.register_buffer('sin_cached', freqs_grid.sin())
    
    def forward(self, x):
        """Apply RoPE using PyTorch operations"""
        batch, heads, seq_len, head_dim = x.shape
        
        cos = self.cos_cached[:seq_len].to(x.device)  # [seq_len, head_dim//2]
        sin = self.sin_cached[:seq_len].to(x.device)  # [seq_len, head_dim//2]
        
        # Split into pairs
        x1 = x[..., ::2]   # First half of each pair
        x2 = x[..., 1::2]  # Second half of each pair
        
        # Apply rotation
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back
        output = torch.stack([rotated_x1, rotated_x2], dim=-1)
        output = output.flatten(-2)
        
        return output

def benchmark_rope():
    """Benchmark custom RoPE vs PyTorch RoPE"""
    
    # Test configurations - typical transformer sizes
    configs = [
        {"batch": 8, "heads": 32, "seq_len": 512, "head_dim": 128},   # Small
        {"batch": 4, "heads": 32, "seq_len": 2048, "head_dim": 128},  # Medium
        {"batch": 2, "heads": 32, "seq_len": 4096, "head_dim": 128},  # Large
        {"batch": 1, "heads": 32, "seq_len": 8192, "head_dim": 128},  # Very Large
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config in configs:
        print(f"\n=== Config: B={config['batch']}, H={config['heads']}, S={config['seq_len']}, D={config['head_dim']} ===")
        
        # Create test data [batch, heads, seq_len, head_dim]
        x = torch.randn(config['batch'], config['heads'], config['seq_len'], 
                       config['head_dim'], device=device, dtype=torch.float32)
        
        # Initialize layers
        pytorch_rope = PyTorchRoPE(config['head_dim']).to(device)
        custom_rope = CustomRoPE(config['head_dim']).to(device)
        
        # Warmup
        for _ in range(10):
            _ = pytorch_rope(x)
            _ = custom_rope(x)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch RoPE
        start_time = time.time()
        for _ in range(100):
            pytorch_output = pytorch_rope(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 100
        
        # Benchmark Custom RoPE
        start_time = time.time()
        for _ in range(100):
            custom_output = custom_rope(x)
        torch.cuda.synchronize()
        custom_time = (time.time() - start_time) / 100
        
        # Correctness check
        max_diff = torch.max(torch.abs(pytorch_output - custom_output)).item()
        relative_error = (max_diff / torch.max(torch.abs(pytorch_output)).item())
        
        # Results
        speedup = pytorch_time / custom_time
        print(f"PyTorch RoPE: {pytorch_time*1000:.2f} ms")
        print(f"Custom RoPE:  {custom_time*1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Relative error: {relative_error:.2e}")
        print(f"Correctness: {'✓ PASS' if relative_error < 1e-5 else '✗ FAIL'}")

def benchmark_litllama_integration():
    """Test integration with actual LitLlama model (if available)"""
    try:
        # This would test your RoPE in actual LitLlama context
        print("\n=== LitLlama Integration Test ===")
        print("TODO: Load LitLlama model and replace RoPE layers")
        print("Run inference and measure end-to-end performance")
    except ImportError:
        print("LitLlama not available for integration testing")

if __name__ == "__main__":
    benchmark_rope()
    benchmark_litllama_integration()