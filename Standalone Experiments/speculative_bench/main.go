package main

/*
#cgo LDFLAGS: -L${SRCDIR}/cuda -lspecdecode -lcudart
#include "cuda/spec_decode.h"
*/
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

func baseline(totalSteps, vocab, workload int) float64 {
	start := time.Now()
	for i := 0; i < totalSteps; i++ {
		C.sd_baseline_step(C.int(i), C.int(vocab), C.int(workload))
	}
	elapsed := time.Since(start).Seconds() * 1000 // ms
	return elapsed / float64(totalSteps)
}

func speculative(totalSteps, draftN, vocab, workload, noisePPM int, seed uint32) (msPerTok float64, accepted int) {
	// device buffer for proposed tokens
	dProposed := (*C.int)(C.malloc(C.size_t(draftN) * C.size_t(unsafe.Sizeof(C.int(0)))))
	// NOTE: We're passing host pointer; kernels write to device memory by API,
	// but for simplicity we let CUDA allocate internally in sd_draft (we used device pointer).
	// To be precise, allocate via cudaMalloc in C and expose pointer; for demo, this works because
	// sd_draft/verify operate entirely on device memory they manage. If you want zero-copy,
	// add a cudaMalloc wrapper. For clarity, we won't deref this in Go.

	defer C.free(unsafe.Pointer(dProposed))

	i := 0
	start := time.Now()
	accepted = 0
	for i < totalSteps {
		// propose up to draftN tokens
		C.sd_draft((*C.int)(nil), C.int(draftN), C.int(i), C.int(vocab), C.int(workload), C.uint(seed), C.int(noisePPM))
		// verify and get prefix length accepted
		acc := int(C.sd_verify((*C.int)(nil), C.int(draftN), C.int(i), C.int(vocab), C.int(workload)))
		if acc <= 0 {
			// ensure progress (worst case)
			acc = 1
		}
		i += acc
		accepted += acc
	}
	elapsed := time.Since(start).Seconds() * 1000
	return elapsed / float64(totalSteps), accepted
}

func main() {
	// Params
	total := 512          // total tokens to "generate"
	vocab := 32000        // dummy vocab size
	workload := 50000     // GPU work per token (tune up/down based on GPU)
	draftN := 4           // draft proposes 4 tokens at a time
	noisePPM := 3000      // ~0.3% mismatch rate
	seed := uint32(12345) // draft randomness

	// Initialize CUDA
	C.sd_init()

	// Baseline: verify 1 token per step
	base := baseline(total, vocab, workload)
	fmt.Printf("Baseline:       %.2f ms/token\n", base)

	// Speculative: propose draftN tokens, accept prefix
	spec, accepted := speculative(total, draftN, vocab, workload, noisePPM, seed)
	fmt.Printf("Speculative:    %.2f ms/token (%.2fx faster)\n", spec, base/spec)
	fmt.Printf("Accepted total: %d of %d steps (draftN=%d, noise=%.2f%%)\n",
		accepted, total, draftN, float64(noisePPM)/10000.0)
}
