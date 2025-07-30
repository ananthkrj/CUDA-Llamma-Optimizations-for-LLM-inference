
// This binding defines a c++ function, in this case
// the rmnnorm one such as torch::Tensor rmsnorm_forward(...)

// within that torch imported function, binding calls extern 
#include <torch.extension.h>
#include <cuda_runtime.h>


// wrapper that launches cuda kernel from .cu file.
// find out why exactly i need to restate launch function here
void rmsnorm_forward_cuda(const float* input, const float* weight,
float* output, int B, int D, float eps);

// Use pytorch's c++ APi (ATenm torch::Tensor) to handle tensors
// C++ wrapper function for function that launches kernel
// forward pass, input parametersneeds to be tensors
// rmsnorm_forward or rmsnorm_forward_cuda
torch::Tensor rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, float eps = 1e-6f);

// backward pass implementation
// find out difference between the two implementations, why i need backward and forward
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