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


Steps after reading:
- Start writing a CUDA RMSNorm for `float32`, then `half`.
- Find out how to implement warp-wide reductions.
- Only refer to shared memory optimization guides once basic kernel is working
- Then implement shared memory

kernels: Cuda files
bindings: Pytorch extension wrappers
benchmarks: performance comparison vs baseline pytorch operations

Key Concepts:
- Wil use shared memory and fused bias + activation to optimize rmsNorm
- Use Warp Level Reduction in addition to Sequential Addressing to fully optimize smaller 
dimensions in transformer models