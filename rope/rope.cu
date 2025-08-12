#include <cuda_runtime.h>

/*
Workflow:
1. optimized rope kernel
2. test simple
3. shared rope kernel
4. test
5. finalize forward launch function
*/

// First kernel for simple rope implementation
// parameters, input tensor, output tesnor, cos_cached, sin_cached,
// B for batch size, H for number of heads, S for sequence length,
// D is dimension of each head in attention module
// in total B, H, S, D represent the number dimensions of input processed
// by model, or the total elements

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
        int s = (pair_index / (D / 2)) % S;

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
    int pair_in_head = pair_index % (D / 2);
    int s = (pair_index / (D / 2)) % S;
    int h = (pair_index / ((D / 2) * S)) % H;
    int b = pair_index / ((D / 2) * S * H);

    // calculate memory indices
    int base_index = b * (H * S * D) + h * (S * D) + s * D; // somehow similar to coordinates indives
    int x_idx = base_index + pair_in_head;
    int y_idx = base_index + pair_in_head + (D / 2);


    // use cached values from shared memory, and load them
    // into these float values with passing in tid
    float cos_val = shared_cos[tid];
    float sin_val = shared_sin[tid];

    // load and rotate pair, x_idx and y_idx allows to reotate input pair
    float x = input[x_idx];
    float y = input[y_idx];

    // calculation to actually roate pair
    output[x_idx] = x * cos_val - y * sin_val;
    output[y_idx] = x * sin_val + y * cos_val; 
}

void rope_forward(const float* input, float* output, const float* cos_cached,
const float* sin_cached, int B, int H, int S, int D) {
    // if the length of the sequence is less than the
    // threshold use the simple kernel
    // otherwise, memory bound scenario then use
    // optimized kernel
    // launch said kernel with blockdim and gridim

    // define total pairs again
    int total_pairs = B * H * S * (D / 2);

    // choose block size based on problem size
    // total_pairs when 256 vs 512
    // understand why these numbers specifically
    int block_size;
    if (total_pairs < 1024) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    int grid_size = (total_pairs + block_size - 1) / block_size;

    // specify dimensions of grid of thread blocks 
    // and dimensions of each thread block in terms of threads
    dim3 block(block_size);
    dim3 grid(grid_size);

    // use shared memory for larger problems/text
    if (S > 512 && block_size >= 256) {
        size_t shared_mem_size = 2 * block_size * sizeof(float);
        // launch kernel
        shared_rope<<<grid, block, shared_mem_size>>>(input, output,
        cos_cached, sin_cached, B, H, S, D);
    } else {
        rope_optimized_kernel<<<grid, block>>>(input, output, cos_cached,
        sin_cached, B, H, S, D);
    }
    // error checking for kernel launch (find out if better to do in binding or cuda file)
}