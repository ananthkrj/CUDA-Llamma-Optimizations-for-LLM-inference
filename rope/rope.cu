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

    // key is to understand D / 2
    // first value, so do cos - sin calculation based on rope formula:
    if (is_first_in_pair) {
        // do x*cos - y*sin calc, need to make variable for y (partner idx)
        // partner idx should be after D / 2
        int partner_index = idx + D/2;
        int x = input[idx];
        int y = input[partner_index];
        // find out why i am passing idx into output here
        output[idx] = x * cos_val - y * sin_val;
    // second value in formula sin + cos
    } else {
        // partner idx here, should be the one before D/2
        int partner_index = idx - D/2;
        int x = input[partner_index];
        int y = input[idx];
        output[idx] = x * sin_val + y * cos_val;
    }
}

__global__ void rope_optimized_kernel(const float* input, float* output,
const float* cos_cached, sin_cached, int B, int H, int S, int D) {
    // better for production use cases, utilizes memory coalescing
    // each thread processes complete pairs

    // changes to make towards optimized kernel

    // pain points:
    // 1. pair based thread mapping instad of individual 1 thread, map 
    // 1 thread to a pair of elements, so it is pair_index thread
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total pairs = B * H * S * (D / 2);

    // boundary case
    if (pair_index >= total_elements) {
        return;
    }

    // decode pair index to 4d coordinates
    // difference is, is that D is split into D / 2
    int pair_in_head = pair_index % (D / 2);
    int s = (pair_index / (D / 2)) % S;
    int h = (pair_index / ((D / 2) * S)) % H;
    int b = pair_index / ((D / 2) * S * H);

    // calculate memory indices for both individual
    // elements that make up the pair
    int base_idx = b * (H * S * D) + h * (S * D) + s * D;
    // the first and second element
    int x_idx = base_idx + pair_in_head; // value: x in pair
    int y_idx = base_idx + pair_in_head + (D/2) // x_{i+D/2}
    
    // load cos/sin values (one lookup per pair instead of two)
    // which is s * half (D/2) and + the pair_in_head
    float cos_val = cos_cached[s * (D/2) + pair_in_head];
    float sin_val = sin_cached[s * (D/2) + pair_in_head];

    // load both element indices into x and y, which will
    // be used in output calc
    float x = input[x_idx];
    float y = input[y_idx];

    // aaply 2D rotation matrix to the pair, the actual
    // rope calcuilation
    output[x_idx] = x * cos_val - y * sin_val;
    output[y_idx] = x * sin_val + y * cos_val;
}

// shared mem implementation
__global__ shared_rope(const float* input, float* output,
const float* cos_cached, const float* sin_cached, int B, int S, int D, int H) {
    
    // shared memory intialization for cos/sin caching
    // initialize shared memory array
    extern __shared__ float shared_mem[];
    // find out different in shared mem for sin/cos caching
    float* shared_cos = shared_mem;
    float* shared_sin = &shared_mem[blockDim.x];

    // initilaize pair index, total pairs, and thread idx
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = B * H * S * (D / 2);
    int tid = threadIdx.x;

    // cooperative loading of cos/sin values into shared
    // memory
    if (pair_index < total_pairs) {
        // fully understand pair_in_head calcultion
        int pair_in_head = pair_index % (D / 2);
        int s = (pair_index / (D / 2)) % 2;

        shared_cos[tid] = cos_cached(s * (D/2) + pair_in_head);
        shared_sin[tid] = sin_cached(s * (D/2) + pair_in_head);
    }

    // sync threads
    __syncthreads();

    // edge case if pair index if greater/= to total pairs
    if (pair_index >= total_pairs) {
        return;
    }

    // decode coordinates (same)
    int pair_in_head
    int s
    int h
    int b

    // calculate memory indices
    int base_index
    int x_idx
    int y _idx


    // use cached values from shared memory
    float cos_val
    float sin_val

    // load and rotate pair
    float x
    float y

    output[x_idx]
    output[y_idx]


}

void rope_forward() {
    // if the length of the sequence is less than the
    // threshold use the simple kernel
    // otherwise, memory bound scenario then use
    // optimized kernel
    // launch said kernel with blockdim and gridim
}