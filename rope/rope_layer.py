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


class CustomRoPE(nn.module):
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
       # using super init, become some of the 
       # other methods are inheriting from here
       # so need to intiialize this init method first
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

    def _precompute_freqs(self, seq_len: int):
        """
        Used to extend the cos/sin cache (if more size/space needed)
        """
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
        Apply RoPE to input tensor

        args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            because the parameters are [B, H, S, D]
        """
        # All the different input tensors [batch, heads, seq_len, head_dim]
        # intiialize them to be the shape of the input x
        batch, heads, seq_len, head_dim = x.shape
        # Extend the cos/sin cache if sequence is longer than precomputed
        if seq_len > self.cos_cached.size(0):
            self._precompute_freqs(seq_len)

        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        # move cos/sin to the same device as the input
        # to device conversion should be stored in cos_cached and sin_cache
        # find out why pass :seq_len
        cos_cached = self.cos_cached[:seq_len].to(x.device)
        sin_cached = self.sin_cached[:seq_len].to(x.device)
        
        # call the kernel (forward pass)
        # pass in singular x parameter, and cos_cahced, sin_cached
        return _rope_cuda.forward(x, cos_cached, sin_cached)
        
        
    @staticmethod
    def from_standard_rope(standard_rope):
        # extarct parameters from standard implementation
        # need to implement this properly
        """
        Convert from standard RoPE implementation to the custom
        Cuda version
        """

        # extract dim, max_seq_len, and base from standard rope implementation
        dim = getattr(standard_rope, 'dim', standard_rope.head_dim)
        max_seq_len = getattr(standard_rope, 'max_seq_len', 8192)
        base = getattr(standard_rope, 'base', 10000.0)

        # store class inheritance and custom kernel launched in 
        # custom variable
        custom = CustomRoPE(dim=dim, max_seq_len=max_seq_len, base=base)
        return custom


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

