package main

/*
#cgo LDFLAGS: -L${SRCDIR}/cuda -lspecdecode -lcudart
#include "cuda/spec_decode.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"time"
)

type Hist struct{ K int `json:"k"`; Hist map[string]int `json:"hist"`; Samples int `json:"samples"` }
var accCDF []int
var accVals []int

func loadAcceptanceHist(path string) error {
	b, err := os.ReadFile(path)
	if err != nil { return err }
	var h Hist
	if err := json.Unmarshal(b, &h); err != nil { return err }
	accVals = make([]int, 0, len(h.Hist))
	for i := 0; i <= h.K; i++ { accVals = append(accVals, i) }
	accCDF = make([]int, len(accVals))
	cum := 0
	for i, v := range accVals {
		cum += h.Hist[fmt.Sprintf("%d", v)]
		accCDF[i] = cum
	}
	if cum == 0 { accCDF[len(accCDF)-1] = 1 }
	return nil
}

func sampleAccepted() int {
	if len(accCDF) == 0 { return 1 }
	r := rand.Intn(accCDF[len(accCDF)-1])
	for i, c := range accCDF {
		if r < c { return accVals[i] }
	}
	return accVals[len(accVals)-1]
}

func baseline(totalSteps, vocab, workload int) float64 {
	start := time.Now()
	for i := 0; i < totalSteps; i++ {
		C.sd_baseline_step(C.int(i), C.int(vocab), C.int(workload))
	}
	elapsed := time.Since(start).Seconds() * 1000 // ms
	return elapsed / float64(totalSteps)
}

func speculative(totalSteps, draftN, vocab, workloadFlops, workloadMem, memLen, noisePPM int, seed uint32, useMeasuredAcceptance bool) (msPerTok float64, accepted int, avgDraftMs float64, avgVerifyMs float64, draftMsPerTok float64, verifyMsPerTok float64) {
	i := 0
	start := time.Now()
	accepted = 0
	var totalDraftMs, totalVerifyMs float64
	rounds := 0
	for i < totalSteps {
		var draftMs, verifyMs C.float
		if !useMeasuredAcceptance {
			// old path: run kernels for proposal+verify and get real accepted prefix
			C.sd_draft_timed(C.int(draftN), C.int(i), C.int(vocab), C.int(workloadFlops), C.int(workloadMem), C.int(memLen), C.uint(seed), C.int(noisePPM), &draftMs)
			acc := int(C.sd_verify_timed(C.int(draftN), C.int(i), C.int(vocab), C.int(workloadFlops), C.int(workloadMem), C.int(memLen), &verifyMs))
			if acc <= 0 { acc = 1 }
			i += acc
			accepted += acc
			_ = draftMs; _ = verifyMs
		} else {
			// measured path: still pay draft+verify compute cost, but acceptance comes from real hist
			C.sd_draft_timed(C.int(draftN), C.int(i), C.int(vocab), C.int(workloadFlops), C.int(workloadMem), C.int(memLen), C.uint(seed), 0, &draftMs)
			_ = C.sd_verify_timed(C.int(draftN), C.int(i), C.int(vocab), C.int(workloadFlops), C.int(workloadMem), C.int(memLen), &verifyMs)
			acc := sampleAccepted()
			if acc <= 0 { acc = 1 }
			if acc > draftN { acc = draftN }
			i += acc
			accepted += acc
			_ = draftMs; _ = verifyMs
		}
		totalDraftMs += float64(draftMs)
		totalVerifyMs += float64(verifyMs)
		rounds++
	}
	elapsed := time.Since(start).Seconds() * 1000
	if rounds > 0 {
		avgDraftMs = totalDraftMs / float64(rounds)
		avgVerifyMs = totalVerifyMs / float64(rounds)
	}
	if totalSteps > 0 {
		scale := float64(totalSteps)
		draftMsPerTok = totalDraftMs / scale
		verifyMsPerTok = totalVerifyMs / scale
	}
	return elapsed / float64(totalSteps), accepted, avgDraftMs, avgVerifyMs, draftMsPerTok, verifyMsPerTok
}

func main() {
	// Params
	total := 512            // total tokens to "generate"
	vocab := 32000          // dummy vocab size
	workloadFlops := 50000  // GPU work per token (tune up/down based on GPU)
	workloadMem := 8        // mem iters (burn-memory loop count)
	memLen := 1 << 20       // ~1M floats (~4MB) HBM buffer; tune to 16<<20 for larger
	draftN := 4             // draft proposes 4 tokens at a time (match measured K)
	noisePPM := 3000        // ~0.3% mismatch rate (only used in synthetic path)
	seed := uint32(12345)   // draft randomness
	useMeasuredAcceptance := true // toggle to use measured acceptance histogram

	// Initialize CUDA
	C.sd_init()
	// try to load measured acceptance histogram if present
	_ = loadAcceptanceHist("accept_hist_k4.json")

	// timed baseline: use baseline_timed to include mem pressure
	baseMs := 0.0
	var bms C.float
	for i := 0; i < total; i++ {
		_ = C.sd_baseline_step_timed(C.int(i), C.int(vocab), C.int(workloadFlops), C.int(workloadMem), C.int(memLen), &bms)
		baseMs += float64(bms)
	}
	base := baseMs / float64(total)
	fmt.Printf("Baseline:       %.2f ms/token\n", base)

	// Speculative: propose draftN tokens, accept prefix
	spec, accepted, draftAvg, verifyAvg, draftPerTok, verifyPerTok := speculative(total, draftN, vocab, workloadFlops, workloadMem, memLen, noisePPM, seed, useMeasuredAcceptance)
	fmt.Printf("Speculative:    %.2f ms/token (%.2fx faster)\n", spec, base/spec)
	fmt.Printf("Kernel avg ms:  draft %.2f (%.2f ms/token) / verify %.2f (%.2f ms/token)\n",
		draftAvg, draftPerTok, verifyAvg, verifyPerTok)
	fmt.Printf("Accepted total: %d of %d steps (draftN=%d)\n", accepted, total, draftN)
}
