package ingest

import (
	"encoding/json"
	"testing"
)

func TestHierarchicalMetricsBecomeChartableIdentifiers(t *testing.T) {
	ev, ok := parseLine([]byte(`{"kind":"train","step":7,"loss":1.2,"recon/sam":0.4,"relation/moon_8":0.2}`), 123)
	if !ok || ev.Kind != kindTrain {
		t.Fatal("compressor train row was not parsed")
	}
	var extra map[string]any
	if err := json.Unmarshal([]byte(ev.Train.Extra), &extra); err != nil {
		t.Fatal(err)
	}
	if extra["recon_sam"] != 0.4 || extra["relation_moon_8"] != 0.2 {
		t.Fatalf("hierarchical metrics were not normalized: %#v", extra)
	}
	if _, unsafe := extra["recon/sam"]; unsafe {
		t.Fatalf("unsafe metric key remained: %#v", extra)
	}
}
