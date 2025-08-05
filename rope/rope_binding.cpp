#include <torch.extension.h>
#include <cuda_runtime.h>

// really find out if this is for a pytorch function or, just purely
// integrating cuda into lit llama
// call launch function from here
void rope_forward(const float* input, float* output, const float* cos_cached,
const float* sin_cached, int B, int H, int S, int D);

// wrap using torch
torch::rope_forward() {

}


// backward pass implemementation
std::vector<torch::Tensor> rope_backward() {
    
}