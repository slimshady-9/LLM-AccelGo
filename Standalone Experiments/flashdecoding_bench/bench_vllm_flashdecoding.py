import time, json, requests

PORTS = [8000, 8001]   # 8000 = baseline, 8001 = FlashDecoding++
PROMPT = "Explain FlashDecoding++ in one paragraph."
MAX_TOKENS = 128

for p in PORTS:
    url = f"http://localhost:{p}/v1/completions"
    print(f"\nTesting port {p} ...")
    t0 = time.time()
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "microsoft/phi-2",
                         "prompt": PROMPT,
                         "max_tokens": MAX_TOKENS})
    )
    elapsed = (time.time() - t0) * 1000
    print(f"Latency: {elapsed:.1f} ms")
    try:
        print("Usage:", r.json().get("usage"))
    except Exception as e:
        print("Response parse error:", e)
