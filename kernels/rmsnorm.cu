#include <cuda_runtime.h>


// rms norm kernel, declare kernel, and then figure
// out other functions needed


// wrapper function that will allocate device memory, copy
// from host to device, launch cuda kernel using gridDim and
// blockDIm
// copy results back to host from device, and free memory
// host memory should be the parameters of the wrapper launcher

