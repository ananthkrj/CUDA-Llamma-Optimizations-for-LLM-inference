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

    // Prioritize sequential memory access, divergence through
    // careful code design are keu to achieving effixient parallel
    // reduction in rmsnorm

    // Sequential Addressing -> parallel reduction -> rmsnorm -> benchmarked in lit llamA

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

    // implement reduction in shared memory
    // use reverse loop and thread ID- based indexing

    // set integer s to be midpoint of number threads in
    // x axis of a block

    // go backwards until first thread in x axis is reached

    for (int s = blockDim.x / 2; s >= 1) {
        // populate shared thread id with summing 
        // of s variable with thread id as long
        // as the thread id is less than midpoint
        // of mid of x axis of block
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
}


// Goal: Find out what a wrapper function is, why I need it, if i need
// to use pytorch name in header, and if i need to connect to the
// binding

// cpp code in binding file will call the host side wrapper function
// defined here

// find out what the parameters of wrapper functions should be
int main() {

    // allocate input device memory


    // allocate output device memory

    // copy from host to device

    // allocalate block and grid dimensions

    // launch the kernel

    // copy from device back to host

    // free device memory

}

