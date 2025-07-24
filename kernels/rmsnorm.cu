#include <cuda_runtime.h>


// rms norm kernel, declare kernel, and then figure
// out other functions needed

// Need to implement using parallel reduction + shared memory

__global__ void NMSNorm(float* input, float* output) {

    // decl
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
}


// wrapper function that will allocate device memory, copy
// from host to device, launch cuda kernel using gridDim and
// blockDIm
// copy results back to host from device, and free memory
// host memory should be the parameters of the wrapper launcher
int main() {

}

