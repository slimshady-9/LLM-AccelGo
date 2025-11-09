#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// Global memory buffer to simulate HBM traffic
static float* g_mem = nullptr;
static int    g_mem_len = 0;

// proposed tokens device buffer
static int* g_proposed = nullptr;
static int  g_proposed_cap = 0;

static void ensure_proposed_capacity(int n) {
    if (n <= 0) return;
    if (g_proposed && g_proposed_cap >= n) return;
    if (g_proposed) { cudaFree(g_proposed); g_proposed = nullptr; g_proposed_cap = 0; }
    cudaMalloc(&g_proposed, sizeof(int) * (size_t)n);
    g_proposed_cap = n;
}

static void ensure_mem_buffer(int len){
    if (len <= 0) return;
    if (g_mem && g_mem_len >= len) return;
    if (g_mem) { cudaFree(g_mem); g_mem = nullptr; g_mem_len = 0; }
    cudaMalloc(&g_mem, sizeof(float) * (size_t)len);
    g_mem_len = len;
}

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b){
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b); return ms;
}

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

__device__ __forceinline__ void burn_memory(int iters, float* buf, int len){
    if (!buf || len <= 0) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = 0.f;
    for (int k = 0; k < iters; ++k) {
        int i = (idx + k * stride) % len;
        float v = buf[i];
        s = fmaf(v, 1.0001f, s + 0.1234f);
        buf[i] = s;
    }
    if (s == 123456.0f) asm volatile("");
}

__global__ void draft_kernel(int *proposed, int draft_n, int i0, int vocab,
                             int work_flops, int mem_iters, float* mem, int mem_len,
                             unsigned int seed, int noise_ppm) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= draft_n) return;
    // Simulate "draft compute"
    burn_flops(work_flops);
    burn_memory(mem_iters, mem, mem_len);

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
                              int i0, int vocab, int work_flops, int mem_iters, float* mem, int mem_len) {
    // Single thread reduction for prefix-match (draft_n is small like 4–16)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Simulate "verifier compute" once per call
        burn_flops(work_flops);
        burn_memory(mem_iters, mem, mem_len);

        int acc = 0;
        for (int j = 0; j < draft_n; ++j) {
            int gt = true_token(i0 + j, vocab);
            if (proposed[j] == gt) acc++;
            else break;
        }
        *accepted_len = acc;
    }
}

__global__ void baseline_kernel(int *out_token, int i, int vocab, int work_flops, int mem_iters, float* mem, int mem_len) {
    burn_flops(work_flops);
    burn_memory(mem_iters, mem, mem_len);
    *out_token = true_token(i, vocab);
}

void sd_init() {
    cudaFree(0); // lazy context init
}

void sd_draft(int *d_proposed, int draft_n, int i0, int vocab, int workload_iters,
              unsigned int seed, int noise_ppm) {
    // backward-compatible: no mem pressure
    int threads = 128;
    int blocks = (draft_n + threads - 1) / threads;
    ensure_proposed_capacity(draft_n);
    draft_kernel<<<blocks, threads>>>(g_proposed, draft_n, i0, vocab, workload_iters, 0, nullptr, 0, seed, noise_ppm);
    cudaDeviceSynchronize();
}

int sd_verify(const int *d_proposed, int draft_n, int i0, int vocab, int workload_iters) {
    int *d_acc = nullptr; int h_acc = 0;
    cudaMalloc(&d_acc, sizeof(int));
    ensure_proposed_capacity(draft_n);
    verify_kernel<<<1, 32>>>(g_proposed, d_acc, draft_n, i0, vocab, workload_iters, 0, nullptr, 0);
    cudaMemcpy(&h_acc, d_acc, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_acc);
    cudaDeviceSynchronize();
    return h_acc;
}

int sd_baseline_step(int i, int vocab, int workload_iters) {
    int *d_tok = nullptr; int h_tok = 0;
    cudaMalloc(&d_tok, sizeof(int));
    baseline_kernel<<<1, 32>>>(d_tok, i, vocab, workload_iters, 0, nullptr, 0);
    cudaMemcpy(&h_tok, d_tok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_tok);
    cudaDeviceSynchronize();
    return h_tok;
}

// Timed wrappers using CUDA events. Return kernel elapsed ms in ms_out.
void sd_draft_timed(int draft_n, int i0, int vocab, int work_flops, int mem_iters, int mem_len,
                    unsigned int seed, int noise_ppm, float* ms_out) {
    ensure_proposed_capacity(draft_n);
    ensure_mem_buffer(mem_len);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int threads = 128; int blocks = (draft_n + threads - 1) / threads;
    cudaEventRecord(s);
    draft_kernel<<<blocks, threads>>>(g_proposed, draft_n, i0, vocab, work_flops, mem_iters, g_mem, mem_len, seed, noise_ppm);
    cudaEventRecord(e); cudaEventSynchronize(e);
    *ms_out = elapsed_ms(s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
}

int sd_verify_timed(int draft_n, int i0, int vocab, int work_flops, int mem_iters, int mem_len,
                     float* ms_out) {
    ensure_proposed_capacity(draft_n);
    ensure_mem_buffer(mem_len);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int *d_acc = nullptr; int h_acc = 0; cudaMalloc(&d_acc, sizeof(int));
    cudaEventRecord(s);
    verify_kernel<<<1, 32>>>(g_proposed, d_acc, draft_n, i0, vocab, work_flops, mem_iters, g_mem, mem_len);
    cudaEventRecord(e); cudaEventSynchronize(e);
    *ms_out = elapsed_ms(s, e);
    cudaMemcpy(&h_acc, d_acc, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_acc); cudaEventDestroy(s); cudaEventDestroy(e);
    return h_acc;
}

int sd_baseline_step_timed(int i, int vocab, int work_flops, int mem_iters, int mem_len,
                            float* ms_out) {
    ensure_mem_buffer(mem_len);
    int *d_tok = nullptr; int h_tok = 0; cudaMalloc(&d_tok, sizeof(int));
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    baseline_kernel<<<1, 32>>>(d_tok, i, vocab, work_flops, mem_iters, g_mem, mem_len);
    cudaEventRecord(e); cudaEventSynchronize(e);
    *ms_out = elapsed_ms(s, e);
    cudaMemcpy(&h_tok, d_tok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_tok); cudaEventDestroy(s); cudaEventDestroy(e);
    return h_tok;
}

} // extern "C"
