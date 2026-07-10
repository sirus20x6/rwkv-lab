package server

import (
	"os"
	"path/filepath"
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
