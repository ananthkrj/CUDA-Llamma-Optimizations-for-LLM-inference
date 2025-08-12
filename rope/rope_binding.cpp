#include <torch.extension.h>
#include <cuda_runtime.h>

// really find out if this is for a pytorch function or, just purely
// integrating cuda into lit llama
// call launch function from here
void rope_forward(const float* input, float* output, const float* cos_cached,
const float* sin_cached, int B, int H, int S, int D);

// wrap using torch
// all inputs should be tensors
torch::Tensor rope_forward(torch::Tensor input, torch::Tensor cos_cached,
torch::Tensor sin_cached) {
    // input validation, is cuda, and is contigous, dtype of floats
    // double check if i should be validating cos_cached and sin_cached
    TORCH_CHECK(input.is_cuda(), "Input should be a tensor");
    TORCH_CHECK(cos_cached.is_cuda(), "cos cached should be a tensor");
    TORCH_CHECK(sin_cached.is_cuda(), "sin cached should be a tensor");
    TORCH_CHECK(input.is_contigous(), "Input should be contingous");
    TORCH_CHECK(cos_cached.is_contigous(), "cos cached is contigpus");
    TORCH_CHECK(sin_cached.is_contigous(), "sin cached is contigous");
    // assert data types
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input should be a float");
    TORCH_CHECK(cos_cached.dtype() == torch::kFloat32, "cos cached should be a float");
    TORCH_CHECK(sin_cached.dtype() == torch::kFloat32, "sin cached should be a float");

    // shape validation

    // validate integers

    // crete outpute tensor to launch kernel
    // use torch::empty_like()
    auto output = torch::empty_like(input);

    // launch kernel, cuda kernels expect raw pointers to device
    // memory as arguments
    void rope_forward(input.data_ptr<float>, cos_cached.data_ptr<float>,
    sin_cached.data_ptr<float>, output.data_ptr<float>, int B, int H, int S, int D);

    // check for kernel launch errors
    cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "kernel launch failed",
    cudaErrorString(cuda_err));

    return output;
}


// backward pass implemementation
std::vector<torch::Tensor> rope_backward() {
    
}

// register functions with pytorch, need it to expose 
// cuda and cpp code to pytorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

}