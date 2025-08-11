# CUDA-Llamma-Optimizations-for-LLM-inference

**High performance RMSNorm + Rotary Position Embdeddings (ROPE) for LLM inference**

High level value:
- RMS norm in a normalization technique used in transformers to stabilize training and improve performance, 
- Use Rotary Position Embeddings (ROPE) as they are used in almost every layer of LLaMA. 
- ROPE rotates coordinate pairs based on position, this will result in a large speedup for positional encoding and better cache utilozation


**Paper to read + go off:**
RMSNorm paper:
https://arxiv.org/pdf/1910.07467

ROPE paper:
https://arxiv.org/pdf/2104.09864

**RMSNorm Notes:**
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
Key Concepts:
- Before both optimization techniques, need to compute square of sums
value, then load this sum into the shared memory array
- Use Sequential Addressing Parallele reduction to optimize dimensions
that are greater than 4096. 
- Use Warp Level Reduction in second kernel to optimize dimensions less
than or equal to 4096

ROPE Notes:
- What does Rope do:
- How to apply Rope to optimize the llm layers:
- What performance optimizations will this result in:
- Understand the calculations that goes into calculations for pair_index,
total pairs, and decoding coordinates, base_index


