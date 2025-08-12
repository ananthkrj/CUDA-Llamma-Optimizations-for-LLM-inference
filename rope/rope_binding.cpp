#include <torch.extension.h>
#include <cuda_runtime.h>

// really find out if this is for a pytorch function or, just purely
// integrating cuda into lit llama
// call launch function from here
void rope_forward(const float* input, float* output, const float* cos_cached,
const float* sin_cached, int B, int H, int S, int D);

// wrap using torch
// all inputs should be tensors
torch::rope_forward(torch::Tensor input, torch::Tensor cos_cached,
torch::Tensor sin_cached) {
    // input validation

    // shape validation

    // validate integers

    // crete outpute tensor to launch kernel

    // launch kernel

    // check for kernel launch errors
}


// backward pass implemementation
std::vector<torch::Tensor> rope_backward() {
    
}