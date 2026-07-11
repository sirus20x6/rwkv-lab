package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestQualificationReceiptDiscoveryAndPanel(t *testing.T) {
	root := t.TempDir()
	runs := filepath.Join(root, "runs")
	dir := filepath.Join(runs, "qualification")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := `{"schema":"rwkv-lab.production-kernel-qualification.v1",` +
		`"environment":{"device_name":"test GPU"},"adopted":["rosa"],` +
		`"reports":{"rosa":{"available":true,"exact":true,"speedup":1.5,"adopted":true}},` +
		`"metrics":{"peak_memory_bytes":10},"regression_gate":{"passed":true}}`
	if err := os.WriteFile(filepath.Join(dir, "kernel.json"), []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RepoRoot: root, RunsDir: runs}}
	receipts := s.qualificationReceipts()
	if len(receipts) != 1 || receipts[0].Report.Reports["rosa"]["speedup"] != 1.5 {
		t.Fatalf("unexpected receipts: %#v", receipts)
	}
	recorder := httptest.NewRecorder()
	s.handleQualification(recorder, httptest.NewRequest("GET", "/api/qualification", nil))
	body := recorder.Body.String()
	for _, expected := range []string{"qualDevice", "qualBaseline", "baseline passed", "1.50x", "test GPU"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("missing %q in panel: %s", expected, body)
		}
	}
}
