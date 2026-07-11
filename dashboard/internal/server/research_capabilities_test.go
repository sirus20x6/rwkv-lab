package server

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResearchCapabilityInventoryExposesAllReadinessRows(t *testing.T) {
	s := &Server{}
	recorder := httptest.NewRecorder()
	s.handleResearchCapabilities(recorder, httptest.NewRequest("GET", "/api/research-capabilities", nil))
	body := recorder.Body.String()
	for _, expected := range []string{"balanced conversion state", "recurrent state adapters",
		"paged recurrent serving", "triangular delta inversion", "offline sleep consolidation",
		"Reasoning Cache", "triplet-block diffusion", "HiLS hybrid attention",
		"decoder evaluation matrix", "state-offset tuning", "routed recurrent state bank",
		"byte-aware / SuperBPE track", "guarded adapter consolidation", "typed decoding policies",
		"StateX state expansion", "supervised memory training", "routing-free MoE",
		"dense-to-sparse transfer", "external kernel candidates", "ROSA backend registry",
		"action-conditioned JEPA", "Key-Value Means", "neural procedural memory",
		"Mamba-3 recurrence", "M2RNN matrix state", "Compositional Muon",
		"distillation expert merge", "guarded test-time training", "ROSA+ fallback",
		"tool-use length generalization", "energy-based refinement",
		"compressed convolutional attention", "data-filter regime audit",
		"runtime backend matrix", "full execution-plan qualification"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("capability panel missing %q", expected)
		}
	}
}
