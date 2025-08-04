# CUDA-Llamma-Optimizations-for-LLM-inference
high performance RMSNorm + KV Cache, and Flash attention Kernels for LLM interface

High level value:
- RMS norm in a normalization technique used in transformers to stabilize training and improve performance, 
particlularly in llms such as llaMa (like this repo)

- RMS norm is a technique in which more kernels built off 
rms norm like flash attention are used to optimize llms

Paper to read + go off:
https://arxiv.org/pdf/1910.07467

Notes:
RMS normalization - Root mean square normalization
- Rmn
- figure out cuda concepts needed to implement RMSsnorm
- Removes re-centering operation and regularizes sum inputs
with rms alone (potential sum acalcilation)
- Core challenge lies in calculating the sum of squares for
the RMS value across a large input vector in parallel
- understand calculation in research paper
- find out what the rmsnorm operation is defined as
- need to implement a parallel reduction algorithm using shared
memory. Threads within a block sum up partial squared of the input.
Then write results of these threads to shared memory.
- syncthreads 

which stems from:
https://arxiv.org/pdf/1706.03762

kernels: Cuda files
bindings: Pytorch extension wrappers
benchmarks: performance comparison vs baseline pytorch operations

Key Concepts:
- Before both optimization techniques, need to compute square of sums
value, then load this sum into the shared memory array
- Use Sequential Addressing Parallele reduction to optimize dimensions
that are greater than 4096. 
- Use Warp Level Reduction in second kernel to optimize dimensions less
than or equal to 4096

# Workflow
**Always define planned workflow in beginning**
RMSnorm Implementation
1. rmsnorm.cu, rmsnorm_binding.cpp, rmsnorm_layer.py
2. test_rmsnorm.py
3. benchmark_standalone.py

KVcache Implementation
1. kvcache.cu, kvcache_binding.cpp, kvcache_layer.py
2. test_rmsnorm.py


