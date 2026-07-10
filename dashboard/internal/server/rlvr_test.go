package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBoundedInt(t *testing.T) {
	if got, err := boundedInt("", 8, 2, 64); err != nil || got != "8" {
		t.Fatalf("default: got %q, %v", got, err)
	}
	if _, err := boundedInt("not-a-number", 8, 2, 64); err == nil {
		t.Fatal("expected malformed integer to fail")
	}
	if _, err := boundedInt("65", 8, 2, 64); err == nil {
		t.Fatal("expected out-of-range integer to fail")
	}
}

func TestPathUnderRepo(t *testing.T) {
	root := t.TempDir()
	runs := filepath.Join(root, "runs")
	if err := os.Mkdir(runs, 0o755); err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(runs, "parent.pt")
	if err := os.WriteFile(checkpoint, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RepoRoot: root, RunsDir: runs}}
	if got, err := s.pathUnderRepo("runs/parent.pt", true); err != nil || got != checkpoint {
		t.Fatalf("inside path: got %q, %v", got, err)
	}
	if _, err := s.pathUnderRepo(filepath.Join(root, "..", "escape.pt"), false); err == nil {
		t.Fatal("expected path outside repository to fail")
	}
}

func TestReadRLVRCampaigns(t *testing.T) {
	root := t.TempDir()
	runs := filepath.Join(root, "runs")
	dir := filepath.Join(runs, "comparison")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := rlvrCampaign{
		Status: "complete", Algorithms: []string{"gspo"}, Seeds: []int{0, 1, 2},
		Created: 2, Summary: map[string]rlvrSummary{"gspo": {Runs: 3, Promotions: 1}},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "campaign.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RepoRoot: root, RunsDir: runs}}
	rows := s.readRLVRCampaigns()
	if len(rows) != 1 || rows[0].Path != "comparison" || rows[0].Summary["gspo"].Runs != 3 {
		t.Fatalf("unexpected campaign rows: %#v", rows)
	}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest("GET", "/api/rlvr", nil)
	s.handleRLVR(recorder, request)
	if body := recorder.Body.String(); !strings.Contains(body, "comparison") || !strings.Contains(body, "gspo") {
		t.Fatalf("campaign missing from rendered panel: %s", body)
	}
}
