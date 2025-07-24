
// This binding defines a c++ function, in this case
// the rmnnorm one such as torch::Tensor rmsnorm_forward(...)

// within that torch imported function, binding calls extern 
// wrapper that launches cuda kernel from .cu file.

// Use pytorch's c++ APi (ATenm torch::Tensor) to handle tensors

// export function to pyrthon using PYBIND11 module or torch library

// Connext this binding to python using setup.py with
// torch.utils.cpp_extension