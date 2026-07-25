package server

import (
	"strings"
	"testing"
	"time"
)

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
