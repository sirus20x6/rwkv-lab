package sysmon

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMageFlowExpertTrainerIsRecognized(t *testing.T) {
	const script = "mage_flow_expert_train.py"
	if !AllowedScript(script) {
		t.Fatal("Mage-Flow expert trainer is not allowlisted")
	}
	module, ok := ModuleForScript(script)
	if !ok || module != "rwkv_lab.mage_flow_expert_train" {
		t.Fatalf("unexpected module mapping: module=%q ok=%v", module, ok)
	}
	if got := matchedScript(
		[]string{"python", "-m", "rwkv_lab.mage_flow_expert_train", "train"},
	); got != script {
		t.Fatalf("module process was not recognized: %q", got)
	}
}

func TestMLATrainersRemainObservableButCannotUseLegacyLaunchAuthority(t *testing.T) {
	for _, script := range []string{"train_mla.py", "train_mla_engram.py"} {
		if AllowedScript(script) {
			t.Fatalf("%s retained legacy dashboard launch authority", script)
		}
		if _, ok := ModuleForScript(script); ok {
			t.Fatalf("%s retained a legacy launch module", script)
		}
	}
	if got := matchedScript([]string{"python", "-m", "rwkv_lab.train_mla"}); got != "train_mla.py" {
		t.Fatalf("legacy MLA process is no longer observable: %q", got)
	}
}

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
