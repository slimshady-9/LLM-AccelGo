#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Initialize CUDA (optional placeholder for future use)
void sd_init();

// Draft kernel: propose up to draft_n tokens starting at step=i0.
// Fills device buffer d_proposed (length draft_n).
// noise_ppm injects mismatches per million tokens (e.g., 5000 = 0.5%).
void sd_draft(int *d_proposed, int draft_n, int i0, int vocab, int workload_iters, unsigned int seed, int noise_ppm);

// Verify kernel: compares proposed tokens with "ground truth" sequence and
// returns the longest accepted prefix length (0..draft_n).
int sd_verify(const int *d_proposed, int draft_n, int i0, int vocab, int workload_iters);

// Baseline single-step: compute 1 token (simulated), returns token (unused).
int sd_baseline_step(int i, int vocab, int workload_iters);

#ifdef __cplusplus
}
#endif
