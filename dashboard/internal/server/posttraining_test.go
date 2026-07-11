package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPosttrainingDatasetDiscoveryAndRunPaths(t *testing.T) {
	root := t.TempDir()
	datasets := filepath.Join(root, "datasets")
	runs := filepath.Join(root, "runs")
	if err := os.MkdirAll(filepath.Join(datasets, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(datasets, "nested", "train.jsonl"), []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(datasets, "ignore.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(runs, "candidate"), 0o755); err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(runs, "candidate", "ckpt.pt")
	if err := os.WriteFile(checkpoint, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RepoRoot: root, RunsDir: runs}}
	paths := s.posttrainDatasets()
	if len(paths) != 1 || paths[0] != "datasets/nested/train.jsonl" {
		t.Fatalf("unexpected datasets: %#v", paths)
	}
	if got, err := s.posttrainRunCheckpoint("candidate"); err != nil || got != checkpoint {
		t.Fatalf("checkpoint: got %q, %v", got, err)
	}
	for _, bad := range []string{"../escape", "nested/run", ""} {
		if _, err := s.posttrainRunCheckpoint(bad); err == nil {
			t.Fatalf("expected invalid run %q to fail", bad)
		}
	}
}

func TestPosttrainingCampaignAndAdapterLineageDiscovery(t *testing.T) {
	root := t.TempDir()
	runs := filepath.Join(root, "runs")
	campaignDir := filepath.Join(runs, "posttrain")
	loopDir := filepath.Join(runs, "recursive")
	if err := os.MkdirAll(campaignDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(loopDir, 0o755); err != nil {
		t.Fatal(err)
	}
	campaign := map[string]any{"schema": "rwkv-lab.posttrain-campaign.v1", "status": "complete",
		"created_ts": 12.0, "objectives": []string{"dpo"}, "seeds": []int{0, 1},
		"confirmation_seeds": []int{100, 101}, "comparisons": map[string]any{"dpo": map[string]any{
			"confirm": map[string]any{"n": 2, "delta": 0.2, "ci_low": 0.1, "ci_high": 0.3}}},
		"promotion_receipts": []map[string]any{{"objective": "dpo", "eligible": true}}}
	data, _ := json.Marshal(campaign)
	if err := os.WriteFile(filepath.Join(campaignDir, "posttrain-campaign.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	loop := []byte(`{"schema":"rwkv-lab.adapter-recursive-loop.v1","status":"complete","current_checkpoint":"parent.pt","iterations":[{"accepted":true}]}`)
	if err := os.WriteFile(filepath.Join(loopDir, "adapter-loop.json"), loop, 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RepoRoot: root, RunsDir: runs}}
	campaigns, loops := s.readPosttrainCampaigns()
	if len(campaigns) != 1 || campaigns[0].Comparisons["dpo"]["confirm"].CILow != 0.1 {
		t.Fatalf("campaigns: %#v", campaigns)
	}
	if len(loops) != 1 || len(loops[0].Iterations) != 1 || !loops[0].Iterations[0].Accepted {
		t.Fatalf("loops: %#v", loops)
	}
	recorder := httptest.NewRecorder()
	s.handlePosttraining(recorder, httptest.NewRequest("GET", "/api/posttraining", nil))
	body := recorder.Body.String()
	if !strings.Contains(body, "ptCampaignRank") || !strings.Contains(body, "ptCampaignOffload") ||
		!strings.Contains(body, "ptCampaignBootstrap") {
		t.Fatalf("advanced post-training controls missing: %s", body)
	}
}
