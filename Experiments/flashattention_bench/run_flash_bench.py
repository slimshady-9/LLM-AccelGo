import torch

device = "cuda"
dtype = torch.float16  # or torch.bfloat16 on supported GPUs

# Create sample inputs (sequence length = 2048, hidden size = 1024)
B,L,H,D = 1,2048,16,64
q = torch.randn(B,L,H,D, device="cuda", dtype=torch.float16)
k = torch.randn(B,L,H,D, device="cuda", dtype=torch.float16)
v = torch.randn(B,L,H,D, device="cuda", dtype=torch.float16)
from flash_attn import flash_attn_qkvpacked_func

# Stack query (q), key (k), and value (v) tensors along a new dimension
# to create the packed input expected by FlashAttention
qkv = torch.stack([q, k, v], dim=2)

# Ensure all previous CUDA operations are finished before starting the timer
torch.cuda.synchronize()

# Record the start time for benchmarking
t0 = time.time()

# Run the FlashAttention kernel (no dropout applied)
_ = flash_attn_qkvpacked_func(qkv, dropout_p=0.0)

# Wait for all CUDA operations to complete before stopping the timer
torch.cuda.synchronize()

# Compute the elapsed execution time for the FlashAttention call
t_fa = time.time() - t0

#Compare it to naïve standard attention:

def vanilla_attention(q, k, v):
    """
    Computes standard (non-Flash) scaled dot-product attention.

    Args:
        q, k, v: Tensors of shape [B, L, H, D]
            - B: batch size
            - L: sequence length
            - H: number of heads
            - D: head dimension

    Returns:
        Tensor of shape [B, L, H, D] — the attention output.
    """

    # Rearrange tensors to [B, H, L, D] for easier batched matrix operations
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Compute the scaling factor (to prevent large dot products)
    scale = q.size(-1) ** 0.5

    # Compute attention scores: [B, H, L, L]
    attn = torch.matmul(qh, kh.transpose(-2, -1)) / scale

    # Apply softmax along the sequence dimension to get attention weights
    attn = F.softmax(attn, dim=-1)

    # Multiply attention weights by values and return to [B, L, H, D]
    out = torch.matmul(attn, vh).transpose(1, 2)

    return out


# --- Benchmark vanilla attention ---
# Ensure all CUDA operations complete before starting timer
torch.cuda.synchronize()

# Record start time
t0 = time.time()

# Run vanilla attention (for timing only)
_ = vanilla_attention(q, k, v)

# Wait for CUDA completion before stopping timer
torch.cuda.synchronize()

# Compute total execution time
t_van = time.time() - t0
print(f"vanilla: {t_van:.4f}s   flash-attn: {t_fa:.4f}s   speedup: {t_van/t_fa:.2f}x")