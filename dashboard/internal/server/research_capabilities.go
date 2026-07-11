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
		{"StateX state expansion", "transformation oracle", "rwkv_lab.state_expansion", "arXiv:2509.22630"},
		{"supervised memory training", "training objective", "rwkv_lab.supervised_memory", "arXiv:2606.06479"},
		{"routing-free MoE", "LM experiment lever", "rwkv_pretrain --routing-free-moe 1", "arXiv:2604.00801"},
		{"dense-to-sparse transfer", "calibration / distillation", "rwkv_lab.sparse_transfer", "arXiv:2605.16928"},
		{"external kernel candidates", "qualification gate", "rwkv_lab.kernel_candidates", "Mirage / arXiv:2405.05751"},
		{"ROSA backend registry", "qualification gate", "rwkv_lab.rosa_backends", "ROSA-Tuning / ROSA-FPGA"},
		{"action-conditioned JEPA", "diagnostic objective", "rwkv_lab.llm_jepa", "arXiv:2606.27014"},
		{"Key-Value Means", "memory oracle", "rwkv_lab.key_value_means", "arXiv:2605.09877"},
		{"neural procedural memory", "steering oracle", "rwkv_lab.procedural_memory", "arXiv:2606.29824"},
		{"Mamba-3 recurrence", "architecture ablations", "rwkv_lab.mamba3_recurrence", "arXiv:2603.15569"},
		{"M2RNN matrix state", "hybrid-layer oracle", "rwkv_lab.m2rnn", "arXiv:2603.14360"},
		{"Compositional Muon", "optimizer oracle", "rwkv_lab.compositional_muon", "tilde-research"},
		{"distillation expert merge", "conversion oracle", "rwkv_lab.distillation_merge", "arXiv:2603.15590"},
		{"guarded test-time training", "rollback-gated transaction", "rwkv_lab.test_time_training", "TTT-E2E"},
		{"ROSA+ fallback", "statistical oracle", "rwkv_lab.rosa_plus", "bcml-labs/rosa-plus"},
		{"tool-use length generalization", "curriculum / evaluation", "rwkv_lab.tool_length_generalization", "arXiv:2510.14826"},
		{"energy-based refinement", "latent refinement oracle", "rwkv_lab.energy_refinement", "arXiv:2507.02092"},
		{"compressed convolutional attention", "attention oracle", "rwkv_lab.cca_attention", "arXiv:2510.04476"},
		{"data-filter regime audit", "evaluation gate", "rwkv_lab.data_filter_audit", "arXiv:2605.19407 / Apple"},
		{"runtime backend matrix", "qualification gate", "rwkv_lab.runtime_backends", "Albatross / vLLM / TT / JAX"},
		{"native RWKV megakernel backend", "compiled + qualification-gated", "rwkv_lab.megakernel", "Megakernels / TileRT"},
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
