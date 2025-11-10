#!/usr/bin/env python3
import sys
import time, json, argparse, requests, statistics, csv, math
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

def one_call(port: int, prompt: str, max_tokens: int, model: str | None = None):
    url = f"http://localhost:{port}/v1/completions"
    payload_model = model if model is not None else "any"
    t0 = time.time()
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": payload_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
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

def latency_bench(port: int, runs: int, prompt: str, max_tokens: int, model: str | None = None):
    lats, toks, statuses = [], [], []
    # warmup (not counted)
    try:
        one_call(port, prompt, max_tokens, model)
    except Exception:
        pass

    for _ in range(runs):
        try:
            ms, total, status = one_call(port, prompt, max_tokens, model)
            lats.append(ms); toks.append(total); statuses.append(status)
        except Exception:
            lats.append(float("inf")); toks.append(0); statuses.append(0)

    # filter out failed runs (status != 200 or inf)
    ok = [(ms, tk) for ms, tk, st in zip(lats, toks, statuses) if st == 200 and ms != float("inf")]
    failures = len(lats) - len(ok)
    if not ok:
        return {"avg_ms": float("inf"), "median_ms": float("inf"), "p95_ms": float("inf"),
                "avg_toks": 0.0, "token_tps": 0.0, "completion_tps": 0.0, "failures": failures}

    ms_vals = sorted([x[0] for x in ok])
    tk_vals = [x[1] for x in ok]
    avg_ms = statistics.mean(ms_vals)
    median_ms = statistics.median(ms_vals)
    # p95 manual
    idx95 = math.ceil(0.95 * len(ms_vals)) - 1
    idx95 = max(0, min(len(ms_vals) - 1, idx95))
    p95_ms = ms_vals[idx95]
    avg_toks = statistics.mean(tk_vals)
    token_tps = avg_toks / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
    completion_tps = len(ms_vals) / (sum(ms_vals) / 1000.0) if sum(ms_vals) > 0 else 0.0
    return {"avg_ms": avg_ms, "median_ms": median_ms, "p95_ms": p95_ms,
            "avg_toks": avg_toks, "token_tps": token_tps, "completion_tps": completion_tps, "failures": failures}

def load_once(port: int, prompt: str, max_tokens: int, model: str | None = None):
    try:
        ms, total, status = one_call(port, prompt, max_tokens, model)
        return status, ms, total
    except Exception:
        return 0, float("inf"), 0

def load_test(port: int, jobs: int, prompt: str, max_tokens: int, model: str | None = None):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(load_once, port, prompt, max_tokens, model) for _ in range(jobs)]
        results = [f.result() for f in as_completed(futs)]
    wall_sec = time.perf_counter() - start
    ok = [r for r in results if r[0] == 200 and r[1] != float("inf")]
    fail = len(results) - len(ok)
    if not ok:
        return {"ok": 0, "avg_ms": float("inf"), "median_ms": float("inf"), "p95_ms": float("inf"), "total_toks": 0, "completion_tps": 0.0, "fail": fail}
    lats = sorted([r[1] for r in ok])
    toks = [r[2] for r in ok]
    avg_ms = statistics.mean(lats)
    median_ms = statistics.median(lats)
    idx95 = math.ceil(0.95 * len(lats)) - 1
    idx95 = max(0, min(len(lats) - 1, idx95))
    p95_ms = lats[idx95]
    total_toks = sum(toks)
    wall_sec = wall_sec if wall_sec > 0 else 1e-9
    completion_tps = len(lats) / wall_sec
    return {"ok": len(lats), "avg_ms": avg_ms, "median_ms": median_ms, "p95_ms": p95_ms, "total_toks": total_toks, "completion_tps": completion_tps, "fail": fail}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", type=int, default=DEFAULT_PORTS,
                    help="Server ports to test (baseline/speculative pairs). Provide 2 or 4 ports; tool also works with any even/odd number.")
    ap.add_argument("--models", nargs="+", type=str, default=None,
                    help="Optional model labels matching provided ports (e.g. 'phi2_baseline phi2_spec'). If provided, must match number of ports.")
    ap.add_argument("--runs", type=int, default=N_RUNS, help="Latency runs per port.")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Generation length for latency test.")
    ap.add_argument("--prompt", type=str, default=PROMPT, help="Prompt for latency test.")
    ap.add_argument("--concurrency", type=int, default=CONC_JOBS, help="Parallel requests for load test.")
    ap.add_argument("--load_max_tokens", type=int, default=LOAD_MAX_TOKENS, help="Generation length for load test.")
    ap.add_argument("--load_prompt", type=str, default=LOAD_PROMPT, help="Prompt for load test.")
    ap.add_argument("--wait_timeout", type=int, default=60, help="Seconds to wait for each server to become ready.")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output file path for per-port stats.")
    args = ap.parse_args()
    ports = args.ports
    models = args.models
    if models:
        if len(models) != len(ports):
            print(f"Error: --models length ({len(models)}) does not match --ports length ({len(ports)})", file=sys.stderr)
            sys.exit(1)
    else:
        # create a slot per-port so downstream indexing is simple
        models = [None] * len(ports)

    # Group ports into (baseline,spec) pairs for easier running of 2 servers at a time.
    pairs = [ports[i:i+2] for i in range(0, len(ports), 2)]

    def wait_ready(port: int, timeout: int, model: str | None = None) -> bool:
        # Try common health endpoints, fall back to a small POST probe
        deadline = time.time() + timeout
        urls = [f"http://localhost:{port}/v1/health", f"http://localhost:{port}/health", f"http://localhost:{port}/ready"]
        probe_payload = json.dumps({"model": (model if model is not None else "any"), "prompt": "ready?", "max_tokens": 1, "temperature": 0.0})
        headers = {"Content-Type": "application/json"}
        while time.time() < deadline:
            for u in urls:
                try:
                    r = requests.get(u, timeout=3)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
            # fallback POST probe
            try:
                r = requests.post(f"http://localhost:{port}/v1/completions", headers=headers, data=probe_payload, timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    # Wait for ports to be ready (useful when starting servers pairwise)
    for idx, p in enumerate(ports):
        ok = wait_ready(p, args.wait_timeout, models[idx])
        if not ok:
            print(f"Warning: port {p} did not become ready within {args.wait_timeout}s")

    print("\n=== Single-client latency / tokens-per-second ===")
    rows = []
    for idx, p in enumerate(args.ports):
        stats = latency_bench(p, args.runs, args.prompt, args.max_tokens, models[idx])
        rows.append((p, stats))

    print(f"\n| Port | Avg Latency (ms) | Median (ms) | p95 (ms) | Avg Tokens | CompTPS | Failures |")
    print(f"|-----:|-----------------:|-----------:|--------:|-----------:|--------:|---------:|")
    for p, stats in rows:
        avg_ms = stats["avg_ms"]
        median_ms = stats["median_ms"]
        p95_ms = stats["p95_ms"]
        avg_toks = stats["avg_toks"]
        completion_tps = stats.get("completion_tps", 0.0)
        failures = stats.get("failures", 0)
        ms_str = "inf" if avg_ms == float("inf") else f"{avg_ms:.1f}"
        tps_str = f"{completion_tps:.2f}" if completion_tps != 0 else "0.00"
        label = ''
        if models:
            try:
                idx = ports.index(p)
                label = models[idx]
            except Exception:
                label = ''
        if label:
            print(f"| {p:>4} ({label}) | {ms_str:>16} | {median_ms:>10.1f} | {p95_ms:>7.1f} | {avg_toks:>9.1f} | {tps_str:>6} | {failures:>8} |")
        else:
            print(f"| {p:>4} | {ms_str:>16} | {median_ms:>10.1f} | {p95_ms:>7.1f} | {avg_toks:>9.1f} | {tps_str:>6} | {failures:>8} |")

    print("\n=== Concurrency load test (continuous batching observable) ===")
    rows2 = []
    for idx, p in enumerate(args.ports):
        stats2 = load_test(p, args.concurrency, args.load_prompt, args.load_max_tokens, models[idx])
        rows2.append((p, stats2))

    print(f"\n| Port | OK Requests | Median (ms) | p95 (ms) | Total Tokens Returned | CompTPS | Failures |")
    print(f"|-----:|------------:|------------:|--------:|----------------------:|--------:|---------:|")
    for p, stats2 in rows2:
        ok = stats2.get("ok", 0)
        median_ms2 = stats2.get("median_ms", float("inf"))
        p95_ms2 = stats2.get("p95_ms", float("inf"))
        total_toks = stats2.get("total_toks", 0)
        completion_tps2 = stats2.get("completion_tps", 0.0)
        fail = stats2.get("fail", 0)
        ms_str = "inf" if median_ms2 == float("inf") else f"{median_ms2:.1f}"
        label = ''
        if models:
            try:
                idx = ports.index(p)
                label = models[idx]
            except Exception:
                label = ''
        if label:
            print(f"| {p:>4} ({label}) | {ok:>11} | {ms_str:>10} | {p95_ms2:>7.1f} | {total_toks:>22} | {completion_tps2:>7.2f} | {fail:>8} |")
        else:
            print(f"| {p:>4} | {ok:>11} | {ms_str:>10} | {p95_ms2:>7.1f} | {total_toks:>22} | {completion_tps2:>7.2f} | {fail:>8} |")

    # Pair-level summary (if ports were started as baseline/spec pairs)
    print("\n=== Pair summaries (baseline -> speculative) ===")
    # Build a map from port to its stats from first table
    stats_map = {p: stats for (p, stats) in rows}
    for pair in pairs:
        if len(pair) == 1:
            p0 = pair[0]
            print(f"Pair: {p0} (single) - no pair summary available")
            continue
        p0, p1 = pair[0], pair[1]
        s0 = stats_map.get(p0, {})
        s1 = stats_map.get(p1, {})
        ms0 = s0.get("avg_ms", float('inf'))
        ms1 = s1.get("avg_ms", float('inf'))
        ct0 = s0.get("completion_tps", 0.0)
        ct1 = s1.get("completion_tps", 0.0)
        ms_ratio = None
        tps_ratio = None
        if ms0 != float('inf') and ms1 != float('inf') and ms1 != 0:
            ms_ratio = ms0 / ms1
        if ct0 != 0 and ct1 != 0:
            tps_ratio = ct1 / ct0
        label0 = models[ports.index(p0)] if models and models[ports.index(p0)] else str(p0)
        label1 = models[ports.index(p1)] if models and models[ports.index(p1)] else str(p1)
        ms_ratio_str = f"{ms_ratio:.2f}" if ms_ratio is not None else "n/a"
        tps_ratio_str = f"{tps_ratio:.2f}" if tps_ratio is not None else "n/a"
        base_ms_str = f"{ms0:.1f}" if ms0 != float('inf') else 'inf'
        spec_ms_str = f"{ms1:.1f}" if ms1 != float('inf') else 'inf'
        print(f"Pair: {p0}({label0}) -> {p1}({label1}) | base {base_ms_str} ms -> spec {spec_ms_str} ms | ms-speedup(base/spec) = {ms_ratio_str} | tps(base->spec) {ct0:.2f} -> {ct1:.2f} | tps-speedup = {tps_ratio_str}")

    # Optional CSV export
    if args.csv:
        try:
            with open(args.csv, 'w', newline='') as cf:
                w = csv.writer(cf)
                w.writerow(["port", "label", "avg_ms", "median_ms", "p95_ms", "avg_toks", "token_tps", "completion_tps", "failures"])
                for p, stats in rows:
                    label = models[ports.index(p)] if models and models[ports.index(p)] else ''
                    w.writerow([p, label, stats.get('avg_ms'), stats.get('median_ms'), stats.get('p95_ms'), stats.get('avg_toks'), stats.get('token_tps'), stats.get('completion_tps'), stats.get('failures')])
            print(f"Wrote CSV to {args.csv}")
        except Exception as e:
            print(f"Failed to write CSV {args.csv}: {e}")

    print("\nTips:")
    print("- Run servers on 8000/8001 (Phi-2 baseline/speculative) and 8002/8003 (Llama3-8B baseline/speculative).")
    print("- Increase --runs or --concurrency for more stable stats.")
    print("- Keep an eye on `watch -n 0.5 nvidia-smi` during the load test.")

if __name__ == "__main__":
    main()
