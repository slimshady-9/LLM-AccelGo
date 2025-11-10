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

def accept_prefix(prompt_text):
    """Generate K tokens with draft model from prompt_text, then verify whether
    the verify model's greedy argmax matches the proposed text when encoded
    with the verify tokenizer.

    Returns the number of accepted tokens (0..K).
    """
    # Tokenize separately for each model (truncate to MAXLEN for context)
    d_ctx = draft_tok(prompt_text, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids.to(DEVICE)
    v_ctx = verify_tok(prompt_text, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids.to(DEVICE)

    # DRAFT: generate K tokens (greedy)
    with torch.no_grad():
        out = draft.generate(d_ctx, max_new_tokens=K, do_sample=False, pad_token_id=draft_tok.eos_token_id)
    d_prop = out[0, -K:]  # draft proposal tokens (ids in draft vocab)

    # Decode the draft proposal tokens to text, then re-tokenize with verify tokenizer
    prop_text = draft_tok.decode(d_prop, skip_special_tokens=True)
    # If decode yields empty text, nothing to verify
    if len(prop_text.strip()) == 0:
        return 0
    v_prop = verify_tok(prop_text, return_tensors="pt").input_ids.to(DEVICE)

    # Only verify up to K tokens to keep histogram buckets consistent
    v_prop_len = v_prop.shape[1]
    verify_check_len = min(K, v_prop_len)
    if verify_check_len == 0:
        return 0

    # Prepare input for verify model: context + the verify-encoded proposal
    with torch.no_grad():
        v_in = torch.cat([v_ctx, v_prop], dim=1)
        logits = verify(v_in).logits[0, -v_prop_len:]  # logits for the proposed positions

    # Compare verify model's argmax tokens with the verify-encoded proposal tokens
    acc = 0
    for j in range(verify_check_len):
        vtok = int(logits[j].argmax().item())
        if vtok == int(v_prop[0, j].item()):
            acc += 1
        else:
            break
    return acc

hist = {str(i): 0 for i in range(K+1)}
for _ in range(N_SAMPLES):
    prompt = random.choice(prompts)
    a = accept_prefix(prompt)
    hist[str(a)] += 1

with open("accept_hist_k4.json", "w") as f:
    json.dump({"k": K, "hist": hist, "samples": N_SAMPLES}, f, indent=2)

print("Saved accept_hist_k4.json:", hist)
