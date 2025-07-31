
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

    // input validation
    // input and weight are the tensor inputs
    // so need to validate that they are tensors, contigous, and float32
    TORCH_CHECK(input.is_cuda(), "Input should be a tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight should be a tensor");
    TORCH_CHECK(input.is_contigous(), "Input should be contigous");
    TORCH_CHECK(weight.is_contigous(), "Weight should be contigous");
    // assert data types of input and weight
    TORCH_CHECK(input.dtype() == torch::kFloat32, "inputs data type must be Float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weights data type muyst be Float32");


    // shape validation
    // [B, D] == [B] so input(1) should be weight(0)
    // and size of input must be equal to size of weight 

    // first verify dimensions (shape) of 
    // input is [B, D] and weight is [D]
    TORCH_CHECK(input.dim() == 2, "dimensions of input are B and D");
    TORCH_CHECK(weight.dim() == 1, " dimensions of weight are D");
    // validate that D from input == D from weight
    TORCH_CHECK(input.size(1) == weight.size(0));

    // validate what B and D are within input
    int B = input.size(0);
    int D = input.size(1);
    

    // create output tensor, to launch kernel
    // use torch::empty_like() as it uses the properties of an existing
    // tensor (input) to create a new one
    auto output = torch::empty_like(input);

    // launch cuda kernel, use data_ptr<float>
    // cuda kernels expect raw pointers to device memory as arguments, when launching
    // .data_ptr<> allows me to access raw memory address of the tensor, which
    // can then be passed to the cuda kernel as an argument
    void rmsnorm_forward_cuda(input.data_ptr<float>, weight.data_ptr<float>,
    output.data_ptr<float>, int B, int D);

    // check for launch errors
    // utilize torch_check to assert, due to nature of torch_check, if 
    // statement within isnt true then error will prompt
    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "Kernel failed to launch",
    cudaErrorString(cuda_err));

    // return output
    return output;

// backward pass implementation
// find out difference between the two implementations, why i need backward and forward
// what goes into backward
// need to enable automatic differentiation (autograd)
// backward pass implements the process of backpropagation
// takes gradient of the loss with respect to the output (grad_output)
std::vector<torch::Tensor> rmsnorm_backward(torch::Tensor grad_output, torch::Tensor input,
torch::Tensor weight, float eps = 1e-6f) {

    // validate grad output, input and weight tensors first
    TORCH_CHECK(grad_output.is_cuda(), "Grad output is a tensor");
    TORCH_CHECK(input.is_cuda(), "input is a tensor");
    TORCH_CHECK(weight.is_cuda(), "weight is a tensor");

    // enable gradient variables for input and weight by cloning and detaching input + weight
    // find out what detatch().requires_grad_(true) does
    auto input_copy = input.copy().detatch().requires_grad(true);
    auto weight_copy = weight.copy().detatch().requires_grad(true);

    // forward pass to get gradients, create output variable
    auto output = rms_forward_cuda(input_copy, weight_copy, eps);

    // backward pass using output variable and passing in grad_output
    // find out why its backward and not rmsnorm_backward
    output.backward(grad_output);

    // initalize grad_input and grad_output variables using gradient variables
    // find out why performing grad on the copied input and weight variables
    // allows for creation of grad_input and grad_weight
    auto grad_input = input_copy.grad();
    auto grad_weight = weight_copy.grad();

    // handle case where gradients miught be None

    // return gradient input and weight


}


// Register functions with pytorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

}