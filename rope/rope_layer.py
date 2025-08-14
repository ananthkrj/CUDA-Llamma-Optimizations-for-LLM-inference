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
        os.path.join(os.path.dirname(__file__), "rope_binding.cpp"),
    ],
    verbose=True,
    extra_cuda_flags=["-O3", "-use_fast_math", "--expt-relaxed-constexpr"]
)


class CustomRope(nn.module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
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
       self.dim = dim
       self.max_seq_len = max_seq_len
       self.base = base

       # Rope has no learnable parameters
       # save buffers with model but not trained
       self.register_buffer('cos_cached', None, persistent=False)
       self.register_buffer('sin_cached', None, persistent=False)

       # precompute cos/sin values
       # call precompute freqs method
       self._precompute_freq(max_seq_len)

    # understand every aspect of this method
    def _precompute_freqs(self, seq_len: int):
        # create frequency values
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        # create position indices
        t = torch.arange(seq_len).float()

        # compute outer priduct: position * frequency
        freqs_grid = torch.outer(t, freqs)

        # precompute cos and sin
        self.cos_cached = freqs_grid.cos()
        self.sin_cached = freqs_grid.sin()

    # make sure that this is a type hint
    # implement fully
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls forward cuda function from the cpp binding
        """
        if not x.is_contigous():
            x = x.contigous()
        
        # call function, input is z
        return _rope_cuda.forward(x, self.cos_cached, self.sin_cached)
        
    @staticmethod
    def from_standard_rope(standard_rope):
        # extarct parameters from standard implementation
        # need to implement this properly

def replace_model_with_custom(model):
    """
    Replace all RoPE layers in model with custom Rope implementation
    """
    for name, module in model.named_modules():
        # Look for common RoPE layer names/types
        if 'rope' in name.lower() or 'rotary' in name.lower():
            # Replace with custom implementation
            parent = model
            components = name.split('.')
            for component in components[:-1]:
                parent = getattr(parent, component)
            
            custom_rope = CustomRoPE.from_standard_rope(module)
            setattr(parent, components[-1], custom_rope)
    
    return model

