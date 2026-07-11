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
		"byte-aware / SuperBPE track", "guarded adapter consolidation", "typed decoding policies"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("capability panel missing %q", expected)
		}
	}
}
