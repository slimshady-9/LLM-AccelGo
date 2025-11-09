synthetic_bench — README

Overview
--------
This folder contains a compact benchmarking harness that supports two complementary modes:

1) Pure synthetic Go harness (no CUDA): fast, portable exploration of speculative decoding behavior. It simulates per-token compute and memory costs with tunable knobs and can sample an empirical acceptance histogram.

2) Optional CUDA-backed harness: the same folder contains a CUDA implementation of simple "draft/verify/baseline" kernels (in `cuda/`). You can build a shared library from these sources and link a Go harness (cgo) to exercise real GPU timings using CUDA events.

Why this layout
----------------
Putting a CUDA implementation alongside a pure-Go synthetic harness makes experiments reproducible and easy to iterate:
- Use the pure-Go harness to sweep acceptance distributions and knob values quickly on any machine.
- Build the CUDA lib and run a CUDA-backed harness on a GPU-equipped machine to measure real kernel timings and memory pressure effects.
- Keep acceptance measurement (Python) in the same folder so the measured distribution can be used by either harness.

Files in this folder
--------------------
- `go.mod` — go module for the synthetic harness.
- `main.go` — Pure-Go synthetic benchmark. By default it:
  - Simulates `baseline_step`, `draft_kernel`, and `verify_kernel` latencies as simple functions of `workflops` and `workmem` knobs.
  - Optionally loads an acceptance histogram (`accept_hist_k4.json`) produced by `measure_acceptance.py` and samples accepted-prefix lengths from it.
  - Prints ms/token (baseline & speculative), accepted totals, per-kernel averages, and can export a small CSV.
  - Flags include `--total`, `--draftN`, `--workflops`, `--workmem`, `--measured`, `--hist`, `--flopScale`, `--memScale`, and `--csv`.

- `measure_acceptance.py` — Python script that collects an empirical acceptance histogram for K (draft length). It runs a small number of prompts through a draft model (TinyLlama) and a verifier (Phi-2) on GPU and writes `accept_hist_k4.json`. Use this to ground acceptance behavior in reality.

- `cuda/spec_decode.cu` and `cuda/spec_decode.h` — Minimal CUDA implementation of the kernels used by the synthetic harness:
  - `draft_kernel`, `verify_kernel`, `baseline_kernel` exercise compute (`burn_flops`) and HBM traffic (`burn_memory`).
  - Timed wrappers `sd_draft_timed`, `sd_verify_timed`, `sd_baseline_step_timed` wrap kernel launches with CUDA events and return elapsed ms via a `float* ms_out` parameter.
  - Device/global helpers: `g_mem` (HBM buffer) and `g_proposed` (device proposed-token buffer) with `ensure_*` helpers.

How to use the folder
---------------------
1) Collect a measured acceptance histogram (optional, recommended for realistic runs):

```powershell
# activate environment with torch/transformers and GPU support
conda activate llmaccel
cd '...\speculative_bench\synthetic_bench'
python measure_acceptance.py
# this writes accept_hist_k4.json
Get-Content .\accept_hist_k4.json
```

2) Run the pure-Go synthetic harness (no CUDA required):

```powershell
cd '...\speculative_bench\synthetic_bench'
go run . --total 512 --draftN 4 --workflops 50000 --workmem 8 --measured --hist accept_hist_k4.json --csv synth_out.csv
```

Tune `--flopScale` and `--memScale` to map the synthetic units to real ms if you want the synthetic outputs to approximate your GPU timings.

3) Build the CUDA shared library (to use real GPU kernels)

On Windows with nvcc/MSVC you will need to adapt the command for your environment. Example (adjust CUDA toolkit and MSVC setup if needed):

```powershell
cd '...\speculative_bench\synthetic_bench\cuda'
# produce libspecdecode.dll; adjust compiler flags for your toolchain
nvcc -shared -Xcompiler "/MD" -o libspecdecode.dll spec_decode.cu -lcudart
```

On Linux:

```bash
nvcc -std=c++14 -shared -Xlinker -soname,libspecdecode.so -o libspecdecode.so spec_decode.cu -lcudart
```

Need a single command that rebuilds the `.so`, exports `LD_LIBRARY_PATH`, and runs the Go harness? From this folder:

```bash
bash ./run_cuda_once.sh --total 512 --draftN 4 --workflops 50000 --workmem 8
```

The script compiles `cuda/libspecdecode.so`, prepends `$(pwd)/cuda` to `LD_LIBRARY_PATH`, and forwards any extra flags to `go run .`.

If you just want a one-liner that builds the `.so`, exports `LD_LIBRARY_PATH`, and runs the Go harness, use:

```bash
cd Standalone\ Experiments/speculative_bench/synthetic_bench && bash build_cuda_run_go.sh
```

4) Link a Go harness with cgo to call the timed wrappers

The current `main.go` is a pure-Go synthetic harness. If you want a Go program that calls the CUDA timed wrappers (`sd_draft_timed`, `sd_verify_timed`, `sd_baseline_step_timed`) I can add a cgo-based `main_cuda.go` that:
- uses `// #cgo LDFLAGS: -L${SRCDIR}/cuda -lspecdecode -lcudart` to link the built lib in `cuda/`;
- calls the timed wrappers and prints per-kernel and end-to-end ms/token.

5) Running vLLM HTTP benchmarks

This folder is focused on synthetic + CUDA kernel-level experiments. To run full HTTP server benchmarks use `benchmark_vllms.py` in the parent `speculative_bench/` folder (it issues POSTs to /v1/completions and reports avg/median/p95, completion TPS and can write CSV). Use it on running vLLM servers (e.g., ports 8000/8001 for Phi pair).

Notes and recommended workflow
------------------------------
- Collect acceptance histogram with `measure_acceptance.py` (TinyLlama -> Phi-2) on a GPU-equipped machine.
- Calibrate the synthetic harness by running smaller CUDA kernels and measuring draft/verify/ms on your GPU, then set `--flopScale` and `--memScale` so the synthetic ms roughly match real kernel timings. This makes sweeping acceptance and knob space useful and realistic.
- When publishing results, capture `nvidia-smi` during the load test and save the CSV or screenshot. Example logging command (PowerShell):

```powershell
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > nvidia-smi-log.csv
```

What I can add next
-------------------
- A cgo-based `main_cuda.go` that calls the timed wrappers directly (so the folder contains both a synthetic run mode and a real CUDA run mode).
- A PowerShell wrapper to build the CUDA lib, run the CUDA harness, collect `nvidia-smi` logs, and run the `benchmark_vllms.py` HTTP client for comparison.

Tell me if you want me to add the cgo harness and/or the PowerShell build+run wrapper and I will implement it in this folder.
