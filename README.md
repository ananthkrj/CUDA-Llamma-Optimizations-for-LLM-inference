# CUDA-Llamma-Optimizations-for-LLM-inference
high performance RMSNorm + KV Cache, and Flash attention Kernels for LLM interface

Paper to read + go off:
https://arxiv.org/pdf/1910.07467

Steps after reading:
- Start writing a CUDA RMSNorm for `float32`, then `half`.
- Find out how to implement warp-wide reductions.
- Only refer to shared memory optimization guides once basic kernel is working
- Then implement shared memory