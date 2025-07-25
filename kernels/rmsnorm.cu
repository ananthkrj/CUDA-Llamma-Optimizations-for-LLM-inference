#include <cuda_runtime.h>


// rms norm kernel, declare kernel, and then figure
// out other functions needed

// Need to implement using parallel reduction + shared memory

__global__ void NMSNorm(float* g_input, float* g_output) {

    // parallel reduction implementation: for each array in the batch,
    // assign a thread block consisting of a fixed number of threads to compute 
    // the sum of elements in the array

    // Sequential addressing is more efficient than interleaved
    // addressingwhen it comes to implementing nmsnorm and 
    // other functions. Find out why on 25th

    // Prioritizing sequential memory access, divergence through
    // careful code design are ket to achieving effixient parallel
    // reduction in rmsnorm

    // sequential memory -> divergence -> parallel reduction -> optimized rmsnorm

    // declare shared memory array
    extern __shared__ int = sdata[];

    // declare individual thread id, will pass this into
    // shared mem array
    int tid = threadIdx.x;
    // calculation for thread id
    int i = blockIdx.x * blockDim.x + threadIdx.x
    // transfer global memory to shared memory
    sdata[tid] = g_input[i];
    __syncthreads();
}


// wrapper function that will allocate device memory, copy
// from host to device, launch cuda kernel using gridDim and
// blockDIm
// copy results back to host from device, and free memory
// host memory should be the parameters of the wrapper launcher
int main() {

}

