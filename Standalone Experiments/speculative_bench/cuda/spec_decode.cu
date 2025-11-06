#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

__device__ __forceinline__ int true_token(int i, int vocab) {
    // Deterministic "ground truth" token generator (LCG-like)
    uint32_t x = (1103515245u * (uint32_t)i + 12345u);
    return (int)(x % (uint32_t)vocab);
}

__device__ __forceinline__ int lcg_rand(uint32_t seed, uint32_t idx) {
    uint32_t x = seed ^ (idx * 747796405u + 2891336453u);
    x ^= x >> 17; x *= 0xed5ad4bbU;
    x ^= x >> 11; x *= 0xac4c1b51U;
    x ^= x >> 15; x *= 0x31848babU;
    x ^= x >> 14;
    return (int)x;
}

__device__ __forceinline__ void burn_flops(int iters) {
    // Doing some FMA work to consume GPU time (simulating matmuls)
    float a = 1.0001f, b = 0.9999f, c = 0.1234f;
    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, b, c);
        b = fmaf(b, c, a);
        c = fmaf(c, a, b);
    }
    if (a + b + c == 0.0f) { asm volatile(""); }
}

__global__ void draft_kernel(int *proposed, int draft_n, int i0, int vocab,
                             int workload_iters, unsigned int seed, int noise_ppm) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= draft_n) return;
    // Simulate "draft compute"
    burn_flops(workload_iters);

    int tok = true_token(i0 + idx, vocab);

    // Inject occasional noise (mismatch) based on ppm
    if (noise_ppm > 0) {
        int r = abs(lcg_rand(seed, (uint32_t)(i0 + idx))) % 1000000; // 0..999999
        if (r < noise_ppm) {
            tok = (tok + 1) % vocab; // flip token
        }
    }
    proposed[idx] = tok;
}

__global__ void verify_kernel(const int *proposed, int *accepted_len, int draft_n,
                              int i0, int vocab, int workload_iters) {
    // Single thread reduction for prefix-match (draft_n is small like 4–16)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Simulate "verifier compute" once per call
        burn_flops(workload_iters);

        int acc = 0;
        for (int j = 0; j < draft_n; ++j) {
            int gt = true_token(i0 + j, vocab);
            if (proposed[j] == gt) acc++;
            else break;
        }
        *accepted_len = acc;
    }
}

__global__ void baseline_kernel(int *out_token, int i, int vocab, int workload_iters) {
    burn_flops(workload_iters);
    *out_token = true_token(i, vocab);
}

void sd_init() {
    cudaFree(0); // lazy context init
}

void sd_draft(int *d_proposed, int draft_n, int i0, int vocab, int workload_iters,
              unsigned int seed, int noise_ppm) {
    int threads = 128;
    int blocks = (draft_n + threads - 1) / threads;
    draft_kernel<<<blocks, threads>>>(d_proposed, draft_n, i0, vocab, workload_iters, seed, noise_ppm);
    cudaDeviceSynchronize();
}

int sd_verify(const int *d_proposed, int draft_n, int i0, int vocab, int workload_iters) {
    int *d_acc = nullptr; int h_acc = 0;
    cudaMalloc(&d_acc, sizeof(int));
    verify_kernel<<<1, 32>>>(d_proposed, d_acc, draft_n, i0, vocab, workload_iters);
    cudaMemcpy(&h_acc, d_acc, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_acc);
    cudaDeviceSynchronize();
    return h_acc;
}

int sd_baseline_step(int i, int vocab, int workload_iters) {
    int *d_tok = nullptr; int h_tok = 0;
    cudaMalloc(&d_tok, sizeof(int));
    baseline_kernel<<<1, 32>>>(d_tok, i, vocab, workload_iters);
    cudaMemcpy(&h_tok, d_tok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_tok);
    cudaDeviceSynchronize();
    return h_tok;
}

} // extern "C"
