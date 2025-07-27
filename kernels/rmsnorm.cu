#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>


// rms norm kernel, declare kernel, and then figure
// out other functions needed

// Need to implement using parallel reduction + shared memory

// USe warp level reduction as well

// kernel for large dimensions
// use of __restrict__ tells compiler that memory location of this pointer
// wont be accessed by any others
// aacoridng to formula:
// input = [B. D]
// weight = [D]
// output = [B, D]
__global__ void RMSNorm(const float* __restrict__ input, const float* __restrict__ weight,
float* __restrict__ output, int B, int D, float eps = 1e-6f) {

    // parallel reduction implementation: for each array in the batch,
    // assign a thread block consisting of a fixed number of threads to compute 
    // the sum of elements in the array

    // Sequential Addressing -> parallel reduction -> rmsnorm -> benchmarked in lit llamA

    // each block processes one seqeuence in batch
    // edge case, find out how this relates to rms formula
    int batch_index = threadIdx.x;
    if (batch_index > B) {
        return;
    }

    // declare shared memory array for reduction
    extern __shared__ float sdata[];

    // declare individual thread id, will pass this into
    // shared mem array
    int tid = threadIdx.x;
    // store blockDim.x in block size variable
    int block_size = blockDim.x;


    // calculate offset for this batch
    const float *input_ptr = blockIdx.x;
    // calculation for thread id
    int i = blockIdx.x * blockDim.x + threadIdx.x
    
    // compute sum of squares using parallel reduction

    // each thread processes multiple elements if D > block_size (dim)

    // store the partial sum in shared memory

    // implement parallel reducition using sequential addressing

    for (int s = blockDim.x / 2; s >= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // syncthreads to avoid race conditions (conflictions)
        __syncthreads();
    }


    // write result from shared mem back to global mem
    // as long as individual thread id == 0

    // should pass blockIdx.x to be populated
    // into global mem output array
    if (tid == 0) {
        g_output[blockIdx.x] = sdata[0];
    }

    // compute rms normalization and apply weight
    // understand this calculation:
    // output_ptr[i] = input_ptr[i] * rms_inv * weight[i];
    // and why i start at tidm go till D, and increment i with blocksize
}

// Warp reduction kernel, to optimize smaller dimensions
// declare kernel launch as a forward function for pytorch integration (binding.cpp file later)

// same parmeteters

__global__ void Warp_RMSNorm(const float* __restrict__ input, const float* __restrict__ weight,
float* __restrict__ output, int B, int D) {
    // each block processes one seqeuence in batch

    // initialize tid, warp_id, lane_id using / and % 32 calculations, then num_Warps

    // calculate input_ptr and output_ptr

    // compute sum of squares (refer to formula)

    // warp level reduction within each warp, utilize pragma unroll

    // store warp results in shared memory, up to 32 warps

    // final reduction across warps, by using first warp

    // broadcast result using rms calculation 

    // apply normlaization and scaling
}

void rmsnorm_forward_cuda(float *g_input, float *g_output, float* weight, int B, int D) {
    // optimize block size selection, <= 1024 vs not

    // dim3 for grid and block

    // if D <= 4096, use warp so call that kernel, else use sequential kernel

    // cuda error checking for launching kernels

    // benchmarking check
}