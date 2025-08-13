# Layer onto the cuda and binding files to connect to the actual
# python pytorch function
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# fully understand each component of this file, as it relates
# to the binding file and cuda file before moving on
# to benchmark file

# JIT compile the extension, figure out what join does
_rmsnorm_cuda = load(
    name="rmsnorm_cuda",
    sources=[
        os.path.join(os.path.dirname(__file__), "../kernels/rmsnorm_kernel.cu"),
        os.path.join(os.path.dirname(__file__), "../kernels/rmsnorm_binding.cpp"),
    ],
    verbose=True,
    extra_cuda_cflags=["-O3", "-use_fast_math", "--expt-relaxed-constexpr"]
)

# class and functions to replace rmsnorm functions in python
# with my own cuda + cpp implementation
class CustomRMSNorm(nn.module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # call the forward cuda function from cpp binding
        if not x.is_contigous():
            x = x.contigous() 
        
        # input is x
        return _rmsnorm_cuda.forward(x, self.weight, self.eps)

    @staticmethod
    def from_standard_rmsnorm(standard_rmsnorm):
        """convert from standard rmsnorm to custom implementation"""
        custom = CustomRMSNorm(
            dim=standard_rmsnorm.weight.size(0),
            eps=getattr(standard_rmsnorm, 'eps', 1e-6)
        )
        custom.weight.data.copy_(standard_rmsnorm.weight.data)
        return custom
