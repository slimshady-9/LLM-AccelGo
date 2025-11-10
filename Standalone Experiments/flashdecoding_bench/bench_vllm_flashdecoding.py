import argparse
import time
import json
import requests
import os
import csv
from typing import Dict


# Defaults
DEFAULT_PORTS = [8000, 8001]
DEFAULT_MODEL = os.environ.get("BENCH_MODEL", "microsoft/phi-2")
PROMPT = "Explain FlashAttention vs standard attention in one paragraph."
MAX_TOKENS = 128


def parse_model_map(s: str) -> Dict[int, str]:
    """Parse a model-map string like '8000:phi-2,8001:TinyLlama/..' into a dict."""
    m = {}
    if not s:
        return m
    parts = s.split(",")
    for part in parts:
        if not part:
            continue
        if ":" not in part:
            continue
        port_str, model = part.split(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            continue
        m[port] = model
    return m


def make_prompt(num_words: int) -> str:
    """Create a prompt with approximately num_words tokens."""
    return "word " * max(1, num_words)


def main():
    p = argparse.ArgumentParser(description="Latency/throughput benchmark for vLLM/OpenAI-style servers, with CSV logging and context sweep.")
    p.add_argument("--ports", nargs="+", type=int, default=DEFAULT_PORTS, help="Ports to test (baseline, flash++)")
    p.add_argument("--model", type=str, required=True, help="Model name to test (e.g. microsoft/phi-2)")
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[32, 512, 2048, 8192], help="Prompt lengths (words) to sweep")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--csv", type=str, default="bench_results.csv", help="CSV file to append results to")
    args = p.parse_args()

    flash_map = {8000: False, 8001: True}

    csv_fields = ["timestamp", "model", "port", "flashdecoding", "prompt_len", "latency_ms", "status_code", "prompt_tokens", "completion_tokens", "total_tokens", "ttft_ms", "tpot_ms", "tps", "e2el_ms"]
    csv_path = args.csv
    csv_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if not csv_exists:
            writer.writeheader()

        for prompt_len in args.prompt_lens:
            prompt = make_prompt(prompt_len)
            for port in args.ports:
                url = f"http://127.0.0.1:{port}/v1/completions"
                payload = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens}
                print(f"\nTesting port {port} (model={args.model}, flashdecoding={'ON' if flash_map.get(port, False) else 'OFF'}, prompt_len={prompt_len}) ...")
                t0 = time.time()
                try:
                    r = requests.post(url, headers={"Content-Type": "application/json"},
                                      data=json.dumps(payload), timeout=60)
                except requests.exceptions.RequestException as e:
                    print("Request failed:", e)
                    continue
                ms = (time.time() - t0) * 1000
                print(f"HTTP {r.status_code}, {ms:.1f} ms")
                try:
                    data = r.json()
                    usage = data.get("usage") if isinstance(data, dict) else None
                    prompt_tokens = usage.get("prompt_tokens") if usage else None
                    completion_tokens = usage.get("completion_tokens") if usage else None
                    total_tokens = usage.get("total_tokens") if usage else None

                    # TTFT: If server provides, use it; else estimate as total latency for first token
                    ttft_ms = None
                    if "debug" in data and isinstance(data["debug"], dict):
                        for k in ["first_token_ms", "ttft_ms", "first_token_timestamp_ms"]:
                            if k in data["debug"]:
                                try:
                                    ttft_ms = float(data["debug"][k])
                                    break
                                except Exception:
                                    pass
                    if ttft_ms is None:
                        ttft_ms = ms if completion_tokens == 1 else None

                    # TPOT: Time per output token (decode phase)
                    tpot_ms = None
                    if completion_tokens and completion_tokens > 0:
                        if ttft_ms is not None and ms > ttft_ms:
                            tpot_ms = (ms - ttft_ms) / completion_tokens
                        else:
                            tpot_ms = ms / completion_tokens

                    # TPS: tokens per second
                    tps = None
                    if completion_tokens and ms > 0:
                        tps = (completion_tokens / ms) * 1000

                    # E2EL: End-to-end latency
                    e2el_ms = ms

                    writer.writerow({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": args.model,
                        "port": port,
                        "flashdecoding": flash_map.get(port, False),
                        "prompt_len": prompt_len,
                        "latency_ms": f"{ms:.1f}",
                        "status_code": r.status_code,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "ttft_ms": f"{ttft_ms:.1f}" if ttft_ms is not None else "",
                        "tpot_ms": f"{tpot_ms:.2f}" if tpot_ms is not None else "",
                        "tps": f"{tps:.2f}" if tps is not None else "",
                        "e2el_ms": f"{e2el_ms:.1f}"
                    })
                    print("usage:", usage)
                    print(f"TTFT: {ttft_ms:.1f} ms" if ttft_ms is not None else "TTFT: n/a")
                    print(f"TPOT: {tpot_ms:.2f} ms" if tpot_ms is not None else "TPOT: n/a")
                    print(f"TPS: {tps:.2f}" if tps is not None else "TPS: n/a")
                    print(f"E2EL: {e2el_ms:.1f} ms")
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        text = None
                        if isinstance(choice, dict):
                            text = choice.get("text") or (choice.get("message") or {}).get("content")
                        if text:
                            print("text:", str(text)[:240].replace("\n", " "))
                        else:
                            print("response (no text):", data)
                    else:
                        print("response:", data)
                except Exception:
                    print("raw:", r.text[:400])


if __name__ == "__main__":
    main()
