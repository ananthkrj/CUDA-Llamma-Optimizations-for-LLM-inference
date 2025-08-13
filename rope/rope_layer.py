# layer onto the cuda and binding files to connect to the actual pytorch function
import torch
import torch.nn as nn
import torch.utils.cpp_extension import load
import os

# load cuda and cpp binding file paths
# find out if these cuda flags will even
# compile on colab
_rope_cuda = load(
    name="rope_cuda",
    sources=[
        os.path.join(os.path.dirname(__file__), "rope.cu"),
        os.path.join(os.path.dirname(__File__), "rope_binding.cpp"),
    ],
    verbose=True,
    extra_cuda_flags=["-O3", "-use_fast_math", "--expt-relaxed-constexpr"]
)


class CustomRope(nn.module):
    def __init__(self, dim: int):
       # figure oiut what to do in init
       # why do i need super init, is there 
       # a derived class
       # likely parameters of the actual pytorch function
       # doiuble check if correct
       """
       initializes the class and most important 
       parameters of loaded files for forward pass
       """
       super().__init__()
       self.cos_cached = nn.Parameter(torch.ones(dim))
       self.sin_cached = nn.Parameter(torch.ones(dim))

    # make sure that this is a type hint
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls forward cuda function from the cpp binding
        """
        if not x.is_contigous():
            x = x.contigous()
        
        # call function, input is z
        return _rope_cuda.forward(x, self.cos_cached, self.sin_cached)
        

    def from_standard_rope():
        """
        Places my rope implementation in place of the
        standard rope implementation
        """
        # figure out what even is the standard rope
        # implemenation in pytorch
        # what am i actually replacing