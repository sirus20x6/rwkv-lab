package sysmon

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLivenessUsesFreshStatusHeartbeatDuringLongEvaluation(t *testing.T) {
	root := t.TempDir()
	run := filepath.Join(root, "vision")
	if err := os.MkdirAll(run, 0o755); err != nil {
		t.Fatal(err)
	}
	log := filepath.Join(run, "train.jsonl")
	status := filepath.Join(run, "status.json")
	if err := os.WriteFile(log, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(status, []byte(`{"state":"evaluating"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	old := now.Add(-20 * time.Minute)
	if err := os.Chtimes(log, old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(status, now, now); err != nil {
		t.Fatal(err)
	}

	age, state := liveness(root, "vision", float64(now.UnixNano())/1e9)
	if age == nil || *age > 1 || state != "healthy" {
		t.Fatalf("fresh eval heartbeat was not used: age=%v state=%q", age, state)
	}
}
