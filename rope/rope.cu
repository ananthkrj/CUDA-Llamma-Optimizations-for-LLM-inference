#include <cuda_runtime.h>

/*
Workflow:
1. simple kernel
2. test simple
3. optimized kernel
4. test
5. finalize forward launch function
*/

// First kernel for simple rope implementation
// parameters, input tensor, output tesnor, cos_cached, sin_cached,
// B for batch size, H for number of heads, S for sequence length,
// D is dimension of each head in attention module
// in total B, H, S, D represent the number dimensions of input processed
// by model, or the total elements

__global__ void rope_kernel(const float* input, float* input,
const float* cos_cached, const float* sin_cached, int B, int H, int S, int D) {
    // existing logic, elementwise processing
    // this is good for small sequences, debugging, and validatation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * S * D;

    // edge case, index should be less than total elements always
    if (idx >= total_elements) {
        return;
    }

    // decode indices, so calculate using idx
    // always modulate each variable by themelves
    // need to convert the index thread into meaningful
    // multidimensional coordinates as that is the form of the params
    // thread processes specific tensor with four dimensions

    // writeout examplle

    // which dimension within head_dim
    int d = idx % D;
    // which seqence position
    int s = (idx / D) % S;
    // which attention head
    int h = (idx / (D * S)) % H;
    // which batch
    int b = idx / (D * S * H);

    // rope implemented using elementwise processing
}

__global__ void rope_optimized_kernel(const float* input, float* output,
const float* cos_cached, sin_cached, int B, int H, int S, int D) {
    // better for production use cases, utilizes memory coalescing
    // each thread processes complete pairs
    
    // compute index thread and num of elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = B * S * H * D;

    // edge cadse, index thread needs to be less than element nums
    if (idx >= num_elements) {
        return;
    }

    // concert individual index thread into multi dimensional array
    // for proper tensor acces
    int d = idx % D;
    int s = (idx / d) % S;
    int h = (idx / (S * D)) % H;
    int b = idx / (S * D * H);

    // rope implemented through pairs (x_i, x_i{i + D/2})
}

void rope_forward() {
    // if the length of the sequence is less than the
    // threshold use the simple kernel
    // otherwise, memory bound scenario then use
    // optimized kernel
    // launch said kernel with blockdim and gridim
}