# unit test to verify output correctness

# this will be the script to run
import torch
import sys
import os
# Find out what this line means
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmsnorm.rmsnorm_layer import CustomRMSNorm

def test_rmsnorm_correctness():
    # print statement to check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Small test case
    B, D = 2, 64
    input_tensor = torch.randn(B, D, device=device, dtype=torch.float32)
    weight = torch.ones(D, device=device, dtype=torch.float32)
    
    # Your custom implementation
    custom_norm = CustomRMSNorm(D).to(device)
    custom_norm.weight.data.copy_(weight)
    custom_output = custom_norm(input_tensor)

    # reference implementation, same parameters as kernel
    def reference_rmsnorm(x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x

    # check correctness through print statements
    reference_output = reference_rmsnorm(input_tensor, weight)
    
    # Check correctness
    max_diff = torch.max(torch.abs(custom_output - reference_output)).item()
    print(f"Max difference: {max_diff:.2e}")
    
    assert max_diff < 1e-4, f"Too large difference: {max_diff}"
    print("RMSNorm correctness test passed!")

if __name__ == "__main__":
    # call main function in this file
    test_rmsnorm_correctness()