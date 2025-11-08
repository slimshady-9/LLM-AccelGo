import json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DRAFT_ID   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VERIFY_ID  = "microsoft/phi-2"
DEVICE     = "cuda"
K          = 4                 # proposal length
N_SAMPLES  = 200               # prompts
MAXLEN     = 128

draft_tok  = AutoTokenizer.from_pretrained(DRAFT_ID)
verify_tok = AutoTokenizer.from_pretrained(VERIFY_ID)
draft      = AutoModelForCausalLM.from_pretrained(DRAFT_ID, torch_dtype=torch.float16).to(DEVICE).eval()
verify     = AutoModelForCausalLM.from_pretrained(VERIFY_ID, torch_dtype=torch.float16).to(DEVICE).eval()

prompts = [
    "Explain speculative decoding in one paragraph.",
    "Write two lines of a haiku about the ocean.",
    "What is FlashDecoding++ and why is it fast?",
    "Summarize the benefits of GPU shared memory in transformers.",
    "Describe how attention works in simple words.",
]

def accept_prefix(ctx_ids):
    with torch.no_grad():
        # DRAFT: generate k tokens (greedy)
        out = draft.generate(ctx_ids, max_new_tokens=K, do_sample=False, pad_token_id=draft_tok.eos_token_id)
    prop = out[0, -K:]  # proposal tokens
    # VERIFY: get logits for each proposed position
    with torch.no_grad():
        v_in = torch.cat([ctx_ids.to(DEVICE), prop.unsqueeze(0).to(DEVICE)], dim=1)
        logits = verify(v_in).logits[0, -K:]  # [K, vocab]
    # compare argmax with proposal
    acc = 0
    for j in range(K):
        vtok = int(logits[j].argmax().item())
        if vtok == int(prop[j].item()):
            acc += 1
        else:
            break
    return acc

hist = {str(i): 0 for i in range(K+1)}
for _ in range(N_SAMPLES):
    prompt = random.choice(prompts)
    ctx = verify_tok(prompt, return_tensors="pt").input_ids[:, :MAXLEN].to(DEVICE)
    a = accept_prefix(ctx)
    hist[str(a)] += 1

with open("accept_hist_k4.json", "w") as f:
    json.dump({"k": K, "hist": hist, "samples": N_SAMPLES}, f, indent=2)

print("Saved accept_hist_k4.json:", hist)
