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
    // Converts a flat thread into 4d thread coordinates
    // writeout examplle

    // which dimension within head_dim
    int d = idx % D;
    // which seqence position
    int s = (idx / D) % S;
    // which attention head
    int h = (idx / (D * S)) % H;
    // which batch
    int b = idx / (D * S * H);

    // rope implemented using elementwise processing and in pairs (x_i, x_i{i + D/2})
    // each thread handles one output (inefficient, but implement anyways)
    // pair index calculations: rope doesnt rotate individual elements,
    // it rotates pairs of elements

    // which pair this element belongs do
    int pair_index = d % (D / 2);
    // first value in pair
    bool is_first_in_pair = d < D  / 2;

    // get rotational angle for a specific pair at sequence position
    // s * (D/2) results in sequence position
    int cos_val = cos_cached[s * (D / 2) + pair_index];
    int sin_val = sin_cached[s * (D / 2) + pair_index];

    // first value, so do cos - sin calculation based on rope formula:
    if (is_first_in_pair) {

    }


}

__global__ void rope_optimized_kernel(const float* input, float* output,
const float* cos_cached, sin_cached, int B, int H, int S, int D) {
    // better for production use cases, utilizes memory coalescing
    // each thread processes complete pairs
    
}

void rope_forward() {
    // if the length of the sequence is less than the
    // threshold use the simple kernel
    // otherwise, memory bound scenario then use
    // optimized kernel
    // launch said kernel with blockdim and gridim
}