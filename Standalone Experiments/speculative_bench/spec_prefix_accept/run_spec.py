import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import prefix_accept  # built via setup.py
torch.set_grad_enabled(False)

DEVICE = "cuda"
DTYPE  = torch.float16    # switch to float32 if you hit dtype issues
K      = 4                # draft proposal length
N_NEW  = 128              # tokens to generate

DRAFT_ID  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VERIFY_ID = "microsoft/phi-2"
PROMPT    = "Explain speculative decoding in simple terms."

print("Loading tokenizers...")
tok_d = AutoTokenizer.from_pretrained(DRAFT_ID, use_fast=True)
tok_v = AutoTokenizer.from_pretrained(VERIFY_ID, use_fast=True)

print("Loading models on GPU...")
draft  = AutoModelForCausalLM.from_pretrained(DRAFT_ID,  torch_dtype=DTYPE).to(DEVICE).eval()
verify = AutoModelForCausalLM.from_pretrained(VERIFY_ID, torch_dtype=DTYPE).to(DEVICE).eval()

ctx_d = tok_d(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
ctx_v = tok_v(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

@torch.inference_mode()
def baseline_ms_per_tok():
    cur = ctx_v.clone()
    t0 = time.time()
    for _ in range(N_NEW):
        logits = verify(cur).logits[:, -1, :]          # [1,V]
        nxt = logits.argmax(dim=-1)                    # [1]
        cur = torch.cat([cur, nxt.unsqueeze(0)], 1)
    return (time.time() - t0) * 1000.0 / N_NEW

@torch.inference_mode()
def speculative_ms_per_tok():
    cur_v = ctx_v.clone()
    cur_d = ctx_d.clone()
    made = 0
    t0 = time.time()
    while made < N_NEW:
        # 1) draft proposes K tokens (greedy)
        prop_d = []
        td = cur_d
        for _ in range(K):
            l = draft(td).logits[:, -1, :]             # [1,Vd]
            nxt = l.argmax(dim=-1)                     # [1]
            prop_d.append(int(nxt.item()))
            td = torch.cat([td, nxt.unsqueeze(0)], 1)
        prop_d = torch.tensor(prop_d, device=DEVICE, dtype=torch.long)  # [K]

        # 2) bridge to verifier vocab (decode→encode)
        text = tok_d.decode(prop_d.tolist())
        prop_v = tok_v(text, return_tensors="pt").input_ids.to(DEVICE)[0]   # [K’]
        if prop_v.numel() == 0:  # rare tokenization oddity
            continue

        # 3) verifier once on (context + proposal)
        vin = torch.cat([cur_v, prop_v.unsqueeze(0)], 1)   # [1, Tv+K’]
        logits_tail = verify(vin).logits[0, -prop_v.size(0):, :].contiguous()  # [K’,V]

        # 4) CUDA acceptance
        p = prefix_accept.accept_len(logits_tail, prop_v)
        if p <= 0: p = 1

        # 5) advance contexts by p
        cur_v = torch.cat([cur_v, prop_v[:p].unsqueeze(0)], 1)
        cur_d = torch.cat([cur_d, prop_d[:p].unsqueeze(0)], 1)
        made += p
    return (time.time() - t0) * 1000.0 / N_NEW

if __name__ == "__main__":
    print("Running baseline (verifier-only)...")
    b = baseline_ms_per_tok()
    print(f"Baseline:     {b:.2f} ms/token")

    print("Running speculative...")
    s = speculative_ms_per_tok()
    print(f"Speculative:  {s:.2f} ms/token  ({b/s:.2f}x faster)")
