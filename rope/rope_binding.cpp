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
// backward pass implements process of backpropagation
// takes gradient of loss with respect to the output (grad_output)
std::vector<torch::Tensor> rope_backward(torch::Tensor grad_output,
torch::Tensor input, torch::Tensor cos_cached, torch::Tensor sin_cached) {
    // validate grad output
    TORCH_CHECK(grad_output.is_cuda(), "Grad output is a tensor");
    TORCH_CHECK(input.is_cuda(), "input is a tensor");
    TORCH_CHECK(cos_cached.is_cuda(), "cos cached is a tensor");
    TORCH_CHECK(sin_csched.is_cuda(), "sin cached is a tensor");

    // enable gradient variables for input, cos_cached, and sin_cached
    // probably use grad to intiilaize all parameters?
    // copy, detatch, and requires_grad(true)
    auto input_copy = input.copy().detatch().requires_grad(true);
    auto cos_cached_copy = cos_cached.copy().detatch().requires_grad(true);
    auto sin_cached_copy = sin_cached.copy().detatch().requires_grad(true);

    // create output variable using forward pass and copy parameters
    auto output = rope_forward(input_copy, cos_cached_copy, sin_cached_copy);

    // backward pass to output and pass in grad_output
    output.backward(grad_output);

    // initialize grad_input, cos_cached, and sin_cached using grad
    // copy original variables
    auto grad_input = input.copy.grad();
    auto grad_cos_cached = cos_cached.copy.grad();
    auto grad_sin_cached = sin_cached.copy.grad();

    // handle case where gradients might be zero
    if (!grad_input.defined()) {
        grad_input = torch::zeroes_like(input);
    }

    if (!grad_cos_cached.defined()) {
        grad_cos_cached = torch::zeroes_like(cos_cached);
    }

    if (!grad_sin_cached.defined()) {
        grad_sin_cached = torch::zeroes_like(sin_cached);
    }

    // no errors then return object
    return {grad_input, grad_cos_cached, grad_sin_cached};
}

// register functions with pytorch, need it to expose 
// cuda and cpp code to pytorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rope_forward, "Rope forward pass in Cuda",
    py::arg("input"), py::arg("cos_cached"), py::arg("sin_cached"));

    m.def("backward", &rope_backward, "Rope backward pass in Cuda",
    py::arg("grad_output"), py::arg("input"), py::arg("cos_cached"),
    py::arg("sin_cached"));
}