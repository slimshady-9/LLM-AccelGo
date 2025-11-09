"""
Minimal demo for the CUDA prefix acceptance op without external model deps.

Build first:
  python setup.py build_ext --inplace

Then run:
  python demo_speculative.py
"""

import torch

try:
    import prefix_accept  # pybind module
    have_pybind = True
except Exception:
    have_pybind = False


def run_case(dtype=torch.float32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, V = 5, 10

    # Choose a desired accepted prefix length p (0..K)
    desired_p = 3

    # Create proposal tokens [K]
    prop = torch.tensor([1, 3, 7, 5, 2], dtype=torch.long, device=device)

    # Build logits so that argmax equals prop for rows < desired_p,
    # and mismatches at desired_p (unless desired_p == K)
    logits = torch.zeros((K, V), device=device, dtype=dtype)
    for j in range(K):
        if j < desired_p or desired_p == K:
            logits[j, int(prop[j].item())] = 1.0
        else:
            wrong = (int(prop[j].item()) + 1) % V
            logits[j, wrong] = 2.0

    logits = logits.contiguous()

    # Ensure proposal lives on CUDA for the custom op
    prop_cuda = prop.to(device)

    # Path 1: call via pybind module function
    p1 = None
    if have_pybind:
        p1 = prefix_accept.accept_len(logits, prop_cuda)

    # Path 2: call via torch.ops registry
    # Note: this requires the extension to be loaded (importing prefix_accept loads it).
    p2 = None
    try:
        p2 = torch.ops.prefixacc.accept_len(logits, prop_cuda)
    except Exception:
        pass

    print(f"dtype={dtype}, desired={desired_p}, pybind={p1}, torch.ops={p2}")


if __name__ == "__main__":
    run_case(torch.float32)
    if torch.cuda.is_available():
        run_case(torch.float16)

