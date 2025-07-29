#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>
#define FULL_MASK 0xffffffff

// Implement using rms calculartion + sequential addressing to store calc + 
// kernel for large dimensions
// accoriding to formula:
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
    if (blockIdx.x >= B) {
        return;
    }

    // declare shared memory array for reduction
    extern __shared__ float sdata[];

    // declare individual thread id, will pass this into
    // shared mem array
    int tid = threadIdx.x;

    // calculate offset for this batch
    const float* input_ptr = input + blockIdx.x * D;
    float* output_ptr = output + blockIdx.x * D;
    
    // compute sum of squares using parallel reduction
    float sum = 0.0f;

    // each thread processes multiple elements if D > block_size (dim)
    // square of sums computation
    for (int i = tid; i < D; i += blockDim.x) {
        float val = input_ptr[i];
        sum += val * val;
    }

    // store the partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // implement parallel reducition using sequential addressing

    for (int s = blockDim.x / 2; s >= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // syncthreads to avoid race conditions (conflictions)
        __syncthreads();
    }


    // broadcast final sum to all threads
    // used to create rms_inv variable
    if (tid == 0) {
        float mean_sq = sdata[0] / D; // mean of squares
        rms_inv = rsqrt(mean_sq + eps); // 1/sqrt(mean_sq + eps)
    }

    // compute rms normalization and apply weight
    // understand this calculation:
    // output_ptr[i] = input_ptr[i] * rms_inv * weight[i];
    // and why i start at tidm go till D, and increment i with blocksize
    for (int i = tid; i < D; i += block_size) {
        output_ptr[i] = input_ptr[i] * rms_inv * weight[i];
    }

}

// Warp optimized kernel, to optimize smaller dimensions

__global__ void Warp_RMSNorm(const float* __restrict__ input, const float* __restrict__ weight,
float* __restrict__ output, int B, int D) {
    // each block processes one seqeuence in batch
    // refers to index of a block, as batch is a block
    int batch_index = blockIdx.x;
    if (batch_index >= B) {
        return;
    }

    // initialize tid, warp_id, lane_id using / and % 32 calculations
    // what is warp_id, lane_id, and num_warps?
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (blockDim.x + 31) / 32;

    // calculate input_ptr and output_ptr
    const float* input_ptr = input + blockIdx.x * D;
    float* output_ptr = output + blockIdx.x * D;

    // compute sum of squares, same formula used for rms norm
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        // pass in this index, into input_ptr calculation, find out
        // how that calculation relevant to here
        float val = input_ptr[i];
        sum += val * val;
    }

    // warp level reduction within each warp, utilize pragma unroll
    // use pragma unroll to reduce branch divergence and maximize efficiency
    // of gpu hardware by reducing loop overhead

    // start offset, divide by 2, decrease until 0 reached

    // perform tree reduction to compute sum of sum variable held
    // by each thread in a warp

    // understand warp architecture
    #pragma unroll
    for (int offset = 16; i > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // __shfl_down_sync utilizes the warp memory as it will get the value
    // of the sum variable from the thread at lane X + offset of the same 
    // warp. This data exchange is performed between registers and more 
    // efficient than shared memory.

    // store warp results in shared memory, up to 32 warps
    // support up to 32 warps which is 1024 threads
    __shared__ float wrap_sums[32];
    // find out why the lane_id needs to be 0, in order to load sum
    // into shared memory
    // find out what warp_sums is 
    if (lane_id == 0) {
        warp_sums[wrap_id] = sum;
    }

    __syncthreads();

    // final reduction across all warps
    // load the warp sums into the first warp
    if (lane_id < num_warps) {
        sum = warp_sums[lane_id];
    } else {
        sum = 0.0f;

        // reduce across warps using shuffle, diverge 
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
    }

    // broadcast result using mean_sq and rms_inv again

    // it mean for a tid to be 0: it means only first thread
    // in block executes this
    // what does rms_inv is the inverse of rms norm formula
    // and makes it easy to calculate the final value
    __shared__ float rms_inv;
    if (tid == 0) {
        // squared mean is the squared sum over dimensions
        float mean_sq = sum / D;
        // inverse of rmsnorm is the squared mean + eps scaling values
        // using rsqrtf and inverse of rmsnorm is much faster than
        // regular calculation

        // Calculates 1 / sqrt(mean_sq + eps) using the fast intrinsic (inverse)
        rms_inv = rsqrtf(mean_sq + eps);
    }

    __syncthreads();

    // apply normlaization and scaling, utilizing input pointer, rms_inv, and weight


    // with rms_inv all threads in the block can use rms_inv for final output
    // calculation

    // iterate through hidden dimensions
    // find out why i += blockDim.x, adding length of block every iteration
    // thought to check: start from first thread of a block, iteration through
    // the entire block, move to next block or next thread in block hence blockDim.x (length of block)?
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        // all threads in block computed for 
        output_ptr[i] = intput_ptr[i] * rms_inv * weight[i];
    }

}

void rmsnorm_forward_cuda(float *g_input, float *g_output, float* weight, int B, int D) {
    // optimize block size selection, <= 1024 vs not

    // dim3 for grid and block

    // if D <= 4096, use warp so call that kernel, else use sequential kernel

    // cuda error checking for launching kernels

    // benchmarking check
}