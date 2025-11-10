import time
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import prefix_accept
import seaborn as sns
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)
DEVICE = "cuda"
DTYPE  = torch.float16
N_NEW_DEFAULT = 64
TRIALS_DEFAULT = 2

# Config: adjust as needed
DRAFT_ID  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VERIFY_ID = "microsoft/phi-2"
PROMPTS = [
    "Explain speculative decoding in simple terms.",
    "Summarize the main ideas behind attention mechanisms in transformers.",
    "Write a short, friendly introduction to how speculative decoding speeds up generation."
]
K_VALUES = [4, 8]
N_NEW = N_NEW_DEFAULT
TRIALS = TRIALS_DEFAULT

OUT_DIR = os.path.dirname(__file__)
RESULTS_CSV = os.path.join(OUT_DIR, "bench_results.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "bench_summary.csv")
PLOT_PNG = os.path.join(OUT_DIR, "bench_plot.png")

print("Loading tokenizers and models (this may take a while)...")

tok_d = AutoTokenizer.from_pretrained(DRAFT_ID, use_fast=True)
tok_v = AutoTokenizer.from_pretrained(VERIFY_ID, use_fast=True)

draft  = AutoModelForCausalLM.from_pretrained(DRAFT_ID,  torch_dtype=DTYPE).to(DEVICE).eval()
verify = AutoModelForCausalLM.from_pretrained(VERIFY_ID, torch_dtype=DTYPE).to(DEVICE).eval()

@torch.inference_mode()
def baseline_ms_per_tok(ctx_v, n_new=N_NEW):
    cur = ctx_v.clone()
    t0 = time.time()
    for _ in range(n_new):
        logits = verify(cur).logits[:, -1, :]
        nxt = logits.argmax(dim=-1)
        cur = torch.cat([cur, nxt.unsqueeze(0)], 1)
    return (time.time() - t0) * 1000.0 / n_new

@torch.inference_mode()
def speculative_ms_per_tok(ctx_v, ctx_d, K, n_new=N_NEW):
    cur_v = ctx_v.clone()
    cur_d = ctx_d.clone()
    made = 0
    t0 = time.time()
    while made < n_new:
        # 1) draft proposes K tokens (greedy)
        prop_d = []
        td = cur_d
        for _ in range(K):
            l = draft(td).logits[:, -1, :]
            nxt = l.argmax(dim=-1)
            prop_d.append(int(nxt.item()))
            td = torch.cat([td, nxt.unsqueeze(0)], 1)
        prop_d = torch.tensor(prop_d, device=DEVICE, dtype=torch.long)

        # 2) bridge to verifier vocab (decode→encode)
        text = tok_d.decode(prop_d.tolist())
        prop_v = tok_v(text, return_tensors="pt").input_ids.to(DEVICE)[0]
        if prop_v.numel() == 0:
            continue

        # 3) verifier once on (context + proposal)
        vin = torch.cat([cur_v, prop_v.unsqueeze(0)], 1)
        logits_tail = verify(vin).logits[0, -prop_v.size(0):, :].contiguous()

        # 4) CUDA acceptance
        p = prefix_accept.accept_len(logits_tail, prop_v)
        if p <= 0:
            p = 1

        # 5) advance contexts by p
        cur_v = torch.cat([cur_v, prop_v[:p].unsqueeze(0)], 1)
        cur_d = torch.cat([cur_d, prop_d[:p].unsqueeze(0)], 1)
        made += p
    return (time.time() - t0) * 1000.0 / n_new


def run_bench(prompts=PROMPTS, k_values=K_VALUES, trials=TRIALS, n_new=N_NEW):
    # Prepare CSV header
    header = ["prompt", "K", "trial", "n_new", "baseline_ms_per_tok", "speculative_ms_per_tok"]
    rows = []

    for prompt in prompts:
        ctx_d = tok_d(prompt, return_tensors="pt").input_ids.to(DEVICE)
        ctx_v = tok_v(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Warmup once
        print(f"Warmup for prompt (len {ctx_v.size(1)}): {prompt[:60]!r}...")
        _ = baseline_ms_per_tok(ctx_v, n_new=8)
        _ = speculative_ms_per_tok(ctx_v, ctx_d, K=4, n_new=8)

        for K in k_values:
            for t in range(1, trials + 1):
                print(f"Running: prompt='{prompt[:40]}...', K={K}, trial={t}/{trials}")
                b = baseline_ms_per_tok(ctx_v, n_new=n_new)
                s = speculative_ms_per_tok(ctx_v, ctx_d, K=K, n_new=n_new)
                rows.append([prompt, K, t, n_new, b, s])
                # write incremental results to disk
                with open(RESULTS_CSV, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)

    # Summarize
    summary = {}
    for r in rows:
        key = (r[0], r[1])
        baseline, spec = r[4], r[5]
        if key not in summary:
            summary[key] = {"baseline": [], "spec": []}
        summary[key]["baseline"].append(baseline)
        summary[key]["spec"].append(spec)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "K", "n_runs", "baseline_mean", "baseline_std", "spec_mean", "spec_std", "spec_over_baseline"])
        for (prompt, K), vals in summary.items():
            import statistics
            bm = statistics.mean(vals["baseline"])
            bs = statistics.stdev(vals["baseline"]) if len(vals["baseline"])>1 else 0.0
            sm = statistics.mean(vals["spec"])
            ss = statistics.stdev(vals["spec"]) if len(vals["spec"])>1 else 0.0
            ratio = sm / bm if bm>0 else float('inf')
            writer.writerow([prompt, K, len(vals["baseline"]), bm, bs, sm, ss, ratio])

    # Plot
    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_CSV)
        # create a column for K as str for plotting
        df['K'] = df['K'].astype(str)
        df_m = df.melt(id_vars=["prompt","K","trial","n_new"], value_vars=["baseline_ms_per_tok","speculative_ms_per_tok"], var_name='mode', value_name='ms_per_tok')
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_m, x='prompt', y='ms_per_tok', hue='mode', ci='sd')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(PLOT_PNG)
        print(f"Saved plot to {PLOT_PNG}")
    except Exception as e:
        print("Plotting failed:", e)

    print("Bench complete. Results:", RESULTS_CSV, SUMMARY_CSV)


if __name__ == '__main__':
    run_bench()
