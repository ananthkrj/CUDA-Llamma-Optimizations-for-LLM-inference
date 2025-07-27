
// This binding defines a c++ function, in this case
// the rmnnorm one such as torch::Tensor rmsnorm_forward(...)

// within that torch imported function, binding calls extern 
#include <torch.extension.h>
#include <cuda_runtime.h>


// wrapper that launches cuda kernel from .cu file.

// Use pytorch's c++ APi (ATenm torch::Tensor) to handle tensors

// export function to pyrthon using PYBIND11 module or torch library

// Connext this binding to python using setup.py with
// torch.utils.cpp_extension

// declare kuda kernel

void rmsnorm_forward_cuda() {

}

// C++ wrapper function
torch::Tensor rmsnorm_forward() {
    // input validation

    // shape validation

    // create output tensor

    // launch cuda kernel (end function from last file)

    // check for cuda errors
}

// backward pass implementation
// find out difference between the two implementations
std::vector<torch::Tensor> rmsnorm_backward() {
    // input valdiation

    // enable gradients for input and weight

    // forward pass to get gradients

    // backward pass

    // handle case where gradients miught be done

    // return gradient input and weight


}


// Register functions with pytorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

}