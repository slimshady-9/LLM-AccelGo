#!/usr/bin/env python3
import time, json, argparse, requests, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Config ----------
DEFAULT_PORTS = [8000, 8001, 8002, 8003]
N_RUNS = 10               # per-port latency runs
MAX_TOKENS = 160          # generation length
PROMPT = "Summarize speculative decoding and FlashDecoding++ in 5 sentences."
TIMEOUT = 120             # seconds per request
CONC_JOBS = 16            # concurrency for load test
LOAD_MAX_TOKENS = 200
LOAD_PROMPT = "Write a paragraph about GPUs reading faster with FlashDecoding++."
# ----------------------------

def one_call(port: int, prompt: str, max_tokens: int):
    url = f"http://localhost:{port}/v1/completions"
    t0 = time.time()
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": "any",  # vLLM ignores this field; uses server's model
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }),
        timeout=TIMEOUT,
    )
    t1 = time.time()
    try:
        d = r.json()
    except Exception:
        d = {}
    usage = d.get("usage", {})
    total = usage.get("total_tokens", 0)
    return (t1 - t0) * 1000.0, total, r.status_code  # ms, tokens, status

def latency_bench(port: int, runs: int, prompt: str, max_tokens: int):
    lats, toks, statuses = [], [], []
    # warmup (not counted)
    try:
        one_call(port, prompt, max_tokens)
    except Exception:
        pass

    for _ in range(runs):
        try:
            ms, total, status = one_call(port, prompt, max_tokens)
            lats.append(ms); toks.append(total); statuses.append(status)
        except Exception:
            lats.append(float("inf")); toks.append(0); statuses.append(0)

    # filter out failed runs (status != 200 or inf)
    ok = [(ms, tk) for ms, tk, st in zip(lats, toks, statuses) if st == 200 and ms != float("inf")]
    if not ok:
        return float("inf"), 0.0, 0.0, sum(1 for s in statuses if s != 200)

    ms_vals = [x[0] for x in ok]
    tk_vals = [x[1] for x in ok]
    avg_ms = statistics.mean(ms_vals)
    avg_toks = statistics.mean(tk_vals)
    tps = avg_toks / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
    failures = len(lats) - len(ok)
    return avg_ms, avg_toks, tps, failures

def load_once(port: int, prompt: str, max_tokens: int):
    try:
        ms, total, status = one_call(port, prompt, max_tokens)
        return status, ms, total
    except Exception:
        return 0, float("inf"), 0

def load_test(port: int, jobs: int, prompt: str, max_tokens: int):
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(load_once, port, prompt, max_tokens) for _ in range(jobs)]
        results = [f.result() for f in as_completed(futs)]
    ok = [r for r in results if r[0] == 200 and r[1] != float("inf")]
    fail = len(results) - len(ok)
    if not ok:
        return 0, float("inf"), 0, fail
    lats = [r[1] for r in ok]
    toks = [r[2] for r in ok]
    return len(ok), statistics.mean(lats), sum(toks), fail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", type=int, default=DEFAULT_PORTS,
                    help="Server ports to test (baseline/speculative pairs).")
    ap.add_argument("--runs", type=int, default=N_RUNS, help="Latency runs per port.")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Generation length for latency test.")
    ap.add_argument("--prompt", type=str, default=PROMPT, help="Prompt for latency test.")
    ap.add_argument("--concurrency", type=int, default=CONC_JOBS, help="Parallel requests for load test.")
    ap.add_argument("--load_max_tokens", type=int, default=LOAD_MAX_TOKENS, help="Generation length for load test.")
    ap.add_argument("--load_prompt", type=str, default=LOAD_PROMPT, help="Prompt for load test.")
    args = ap.parse_args()

    print("\n=== Single-client latency / tokens-per-second ===")
    rows = []
    for p in args.ports:
        ms, toks, tps, fail = latency_bench(p, args.runs, args.prompt, args.max_tokens)
        rows.append((p, ms, toks, tps, fail))

    print(f"\n| Port | Avg Latency (ms) | Avg Total Tokens | Throughput (tok/s) | Failures |")
    print(f"|-----:|-----------------:|-----------------:|-------------------:|---------:|")
    for p, ms, toks, tps, fail in rows:
        ms_str = "inf" if ms == float("inf") else f"{ms:.1f}"
        tps_str = f"{tps:.2f}" if tps != 0 else "0.00"
        print(f"| {p:>4} | {ms_str:>16} | {toks:>17.1f} | {tps_str:>19} | {fail:>8} |")

    print("\n=== Concurrency load test (continuous batching observable) ===")
    rows2 = []
    for p in args.ports:
        ok, avg_ms, total_toks, fail = load_test(p, args.concurrency, args.load_prompt, args.load_max_tokens)
        rows2.append((p, ok, avg_ms, total_toks, fail))

    print(f"\n| Port | OK Requests | Avg Latency per Req (ms) | Total Tokens Returned | Failures |")
    print(f"|-----:|------------:|-------------------------:|----------------------:|---------:|")
    for p, ok, avg_ms, total_toks, fail in rows2:
        ms_str = "inf" if avg_ms == float("inf") else f"{avg_ms:.1f}"
        print(f"| {p:>4} | {ok:>11} | {ms_str:>25} | {total_toks:>22} | {fail:>8} |")

    print("\nTips:")
    print("- Run servers on 8000/8001 (Phi-2 baseline/speculative) and 8002/8003 (Llama3-8B baseline/speculative).")
    print("- Increase --runs or --concurrency for more stable stats.")
    print("- Keep an eye on `watch -n 0.5 nvidia-smi` during the load test.")

if __name__ == "__main__":
    main()
