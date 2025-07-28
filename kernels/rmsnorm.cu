#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>

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
    const float* input_ptr = input + batch_index * D;
    float* output_ptr = output + batch_index * D;
    
    // compute sum of squares using parallel reduction
    float sum = 0.0f;

    // each thread processes multiple elements if D > block_size (dim)
    // square of sums computation
    for (int i = tid; i < D; i += block_size) {
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