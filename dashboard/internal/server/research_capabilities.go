package server

// UI inventory for the RWKV-community research additions. The implementations
// and primary paper/repository citations live beside each Python module and in
// README.md; this panel distinguishes runnable dashboard paths from library
// reference paths instead of implying every experimental oracle is production-ready.

import (
	"net/http"
	"strings"

	"github.com/starfederation/datastar-go/datastar"
)

func (s *Server) handleResearchCapabilities(w http.ResponseWriter, r *http.Request) {
	type capability struct{ name, status, entry, source string }
	rows := []capability{
		{"balanced conversion state", "experiment lever", "rwkv_pretrain --balance-state", "Recursal QRWKV7"},
		{"recurrent state adapters", "library oracle", "rwkv_lab.state_tuning", "rwkv-rlhf / OpenMOSE"},
		{"paged recurrent serving", "library oracle", "rwkv_lab.recurrent_serving", "AUXStar/RWKV-Server"},
		{"triangular delta inversion", "qualification suite", "rwkv_lab.triangular_delta", "arXiv:2605.21325"},
		{"offline sleep consolidation", "library oracle", "OnlineAssociativeMemory.sleep_consolidate", "arXiv:2605.26099"},
		{"Reasoning Cache", "library oracle", "rwkv_lab.reasoning_cache", "arXiv:2602.03773"},
		{"triplet-block diffusion", "experimental head", "rwkv_lab.diffusion_rwkv", "arXiv:2605.25969"},
		{"HiLS hybrid attention", "experimental oracle", "rwkv_lab.hils_attention", "arXiv:2607.02980"},
		{"decoder evaluation matrix", "evaluation harness", "rwkv_lab.decoding_eval", "arXiv:2402.06925 / RWKV Discord"},
		{"state-offset tuning", "LM experiment lever", "rwkv_pretrain --state-offset 1", "arXiv:2503.03499"},
		{"routed recurrent state bank", "library oracle", "rwkv_lab.state_bank", "RWKV Discord proposal"},
		{"byte-aware / SuperBPE track", "experiment builder", "rwkv_lab.tokenizer_experiments", "SuperBPE / RWKV Discord"},
		{"guarded adapter consolidation", "promotion-gated controller", "rwkv_lab.adapter_consolidation", "RWKV Discord proposal"},
		{"typed decoding policies", "library oracle", "rwkv_lab.decoding_policy", "RWKV Discord proposal"},
	}
	var b strings.Builder
	b.WriteString(`<div id="research-capabilities-body" class="exp-body"><table class="exp-tbl"><tr class="exp-hd"><td>capability</td><td>readiness</td><td>entry point</td><td>source</td></tr>`)
	for _, row := range rows {
		b.WriteString(`<tr><td>` + esc(row.name) + `</td><td><span class="dim">` + esc(row.status) +
			`</span></td><td><code>` + esc(row.entry) + `</code></td><td>` + esc(row.source) + `</td></tr>`)
	}
	b.WriteString(`</table><p class="dim">All capabilities are off by default. “Oracle” means a correctness-first reference path; hardware adoption still requires a production qualification receipt.</p></div>`)
	_ = datastar.NewSSE(w, r).PatchElements(b.String())
}
