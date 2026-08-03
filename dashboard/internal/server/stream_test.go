package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestMageFlowLoopFallbackRendersLiveUtilization(t *testing.T) {
	runDir := t.TempDir()
	line := `{"kind":"train","step":17,"domain":"animation",` +
		`"loop/mean_gate_rms":0.03125,"loop/mean_expected_loops":2.4,` +
		`"loop/maximum_loops_per_block":3,` +
		`"loop/executed_refinement_block_calls":24,` +
		`"loop/maximum_refinement_block_calls":30,` +
		`"loop/factor_count":4,` +
		`"loop/executed_refinement_factor_calls":72,` +
		`"loop/maximum_refinement_factor_calls":120,` +
		`"loop/total_image_tokens":2000,` +
		`"loop/active_tread_image_tokens":1000,` +
		`"loop/original_backbone_blocks_executed":12,` +
		`"loop/expert_blocks_executed":3}` + "\n"
	if err := os.WriteFile(
		filepath.Join(runDir, "train.jsonl"), []byte(line), 0o600,
	); err != nil {
		t.Fatal(err)
	}

	loop, ok := readLoopRW(runDir)
	if !ok {
		t.Fatal("MageFlow train-log fallback was not detected")
	}
	if !loop.AggregateOnly || loop.ModelFamily != "mageflow" ||
		loop.NLayers != 15 || loop.TreadActiveFraction != 0.5 {
		t.Fatalf("unexpected MageFlow fallback: %+v", loop)
	}
	html := renderLoopRW(loop)
	for _, want := range []string{
		"loop utilization · MageFlow learned depth",
		"resident animation",
		"expected 2.40/3 passes",
		"refinements 24/30",
		"factors 72/120",
		"TREAD active 50%",
		"gate RMS",
	} {
		if !strings.Contains(html, want) {
			t.Fatalf("missing %q in %s", want, html)
		}
	}
}

func TestMageFlowDetailedLoopUsesBlockLabelsAndDepth(t *testing.T) {
	html := renderLoopRW(LoopRW{
		Step: 20, LoopCount: 3, NLayers: 15,
		ModelFamily: "mageflow", ResidentDomain: "photo",
		MeanExpectedLoops: 2.25, ExecutedRefinementCalls: 30,
		MaximumRefinementCalls: 30, TreadActiveFraction: 0.5,
		GateMode: "factored-head-channel",
		Layers: []LoopRWLayer{{
			Layer: 12, Label: "E0", MaxRW: 0.1,
			RW: []float64{0.05, 0.1}, ExpectedLoops: 2.25,
		}},
	})
	if !strings.Contains(html, ">E0<") ||
		!strings.Contains(html, "0.1000 · 2.25x") {
		t.Fatalf("missing MageFlow block utilization detail: %s", html)
	}
}

func TestStreamPatchStateSuppressesIdenticalAndThrottlesLargeChanges(t *testing.T) {
	state := newStreamPatchState()
	now := time.Unix(100, 0)
	if !state.shouldPatch("loop", "large-a", now, 0) {
		t.Fatal("initial region was not patched")
	}
	if state.shouldPatch("loop", "large-a", now.Add(time.Second), 0) {
		t.Fatal("identical region was patched again")
	}
	if !state.shouldPatch("loop", "large-b", now.Add(time.Second), 0) {
		t.Fatal("changed unthrottled region was not patched")
	}
	if !state.shouldPatch("runs", "step-1", now, 5*time.Second) {
		t.Fatal("initial throttled region was not patched")
	}
	if state.shouldPatch("runs", "step-2", now.Add(time.Second), 5*time.Second) {
		t.Fatal("large region ignored its throttle")
	}
	if !state.shouldPatch("runs", "step-3", now.Add(5*time.Second), 5*time.Second) {
		t.Fatal("latest large region was not patched after throttle elapsed")
	}
}

func TestLoopSplitRendersCompactUtilizationSummary(t *testing.T) {
	var output strings.Builder
	renderLoopRWSplit(&output, &LoopRWSplit{
		Heads: 40, Channels: 2560, ChPerHead: 64, ChannelBuckets: 64,
		HeadAbs:    [][]float64{{0, 0.1, 0.2}},
		ChannelAbs: [][]float64{{0, 0, 0.3, 0.1}},
	})
	html := output.String()
	if !strings.Contains(html, "p1 heads · mean 0.1000 · max 0.2000 · active 67%") ||
		!strings.Contains(html, "p1 channels · mean 0.1000 · max 0.3000 · active 50%") {
		t.Fatalf("missing compact utilization statistics: %s", html)
	}
	if len(html) > 1000 {
		t.Fatalf("split summary unexpectedly expanded to %d bytes", len(html))
	}
}
