package server

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadBestUsesManifestCheckpoint(t *testing.T) {
	run := t.TempDir()
	best := filepath.Join(run, "best")
	if err := os.Mkdir(best, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(best, "ckpt.pt"), []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(best, "winner.pt"), []byte("new"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(best, "best.json"),
		[]byte(`{"step":200,"ppl":2.5,"checkpoint":"winner.pt"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got := readBest(run)
	if !got.Exists || got.Step != 200 || got.PPL != 2.5 {
		t.Fatalf("unexpected best: %+v", got)
	}
}

func TestReadBestDoesNotFallbackWhenManifestTargetIsMissing(t *testing.T) {
	run := t.TempDir()
	best := filepath.Join(run, "best")
	if err := os.Mkdir(best, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(best, "ckpt.pt"), []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(best, "best.json"),
		[]byte(`{"step":200,"ppl":2.5,"checkpoint":"missing.pt"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := readBest(run); got.Exists {
		t.Fatalf("mismatched legacy alias was advertised: %+v", got)
	}
}

func TestReadBestRejectsManifestAndTemporaryCheckpointTargets(t *testing.T) {
	for _, target := range []string{"best.json", "winner.pt.tmp", "winner.bin"} {
		t.Run(target, func(t *testing.T) {
			run := t.TempDir()
			best := filepath.Join(run, "best")
			if err := os.Mkdir(best, 0o755); err != nil {
				t.Fatal(err)
			}
			if target != "best.json" {
				if err := os.WriteFile(filepath.Join(best, target), []byte("not a published checkpoint"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			manifest := `{"step":200,"ppl":2.5,"checkpoint":"` + target + `"}`
			if err := os.WriteFile(filepath.Join(best, "best.json"), []byte(manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			if got := readBest(run); got.Exists {
				t.Fatalf("non-.pt checkpoint target was advertised: %+v", got)
			}
		})
	}
}

func TestReadBestRejectsSymlinkCheckpointTargets(t *testing.T) {
	for _, manifest := range []string{
		`{"step":200,"ppl":2.5}`,
		`{"step":200,"ppl":2.5,"checkpoint":"winner.pt"}`,
	} {
		t.Run(manifest, func(t *testing.T) {
			run := t.TempDir()
			best := filepath.Join(run, "best")
			if err := os.Mkdir(best, 0o755); err != nil {
				t.Fatal(err)
			}
			outside := filepath.Join(run, "outside.pt")
			if err := os.WriteFile(outside, []byte("outside"), 0o600); err != nil {
				t.Fatal(err)
			}
			target := "ckpt.pt"
			if strings.Contains(manifest, "winner.pt") {
				target = "winner.pt"
			}
			if err := os.Symlink(outside, filepath.Join(best, target)); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(best, "best.json"), []byte(manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			if got := readBest(run); got.Exists {
				t.Fatalf("symlink checkpoint was advertised: %+v", got)
			}
		})
	}
}

func TestReadBestRejectsSymlinkBestDirectory(t *testing.T) {
	run := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "ckpt.pt"), []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "best.json"),
		[]byte(`{"step":200,"ppl":2.5}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(run, "best")); err != nil {
		t.Fatal(err)
	}
	if got := readBest(run); got.Exists {
		t.Fatalf("symlink best directory was advertised: %+v", got)
	}
}

func TestReadBestRejectsStructurallyInvalidManifest(t *testing.T) {
	tests := []string{
		`{}`,
		`[]`,
		`{"step":10}`,
		`{"step":-1,"ppl":3}`,
		`{"step":10.5,"ppl":3}`,
		`{"step":10,"loss":-0.1}`,
		`{"step":10,"ppl":0}`,
		`{"step":10,"ppl":1e999}`,
		`{"step":10,"loss":true}`,
		`{"step":10,"ppl":3,"checkpoint":null}`,
		`{"step":10,"ppl":3,"checkpoint":"../outside.pt"}`,
	}
	for _, manifest := range tests {
		t.Run(manifest, func(t *testing.T) {
			run := t.TempDir()
			best := filepath.Join(run, "best")
			if err := os.Mkdir(best, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(best, "ckpt.pt"), []byte("old"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(best, "best.json"), []byte(manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			if got := readBest(run); got.Exists {
				t.Fatalf("invalid manifest was advertised: %+v", got)
			}
		})
	}
}

func TestReadBestAcceptsLegacyAndLossOnlyManifestShapes(t *testing.T) {
	for name, manifest := range map[string]string{
		"legacy":    `{"step":17600,"loss":0.9245,"ppl":2.5206}`,
		"loss-only": `{"step":12,"loss":2}`,
	} {
		t.Run(name, func(t *testing.T) {
			run := t.TempDir()
			best := filepath.Join(run, "best")
			if err := os.Mkdir(best, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(best, "ckpt.pt"), []byte("winner"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(best, "best.json"), []byte(manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			got := readBest(run)
			if !got.Exists || got.Step < 0 || !isFinitePositive(got.PPL) {
				t.Fatalf("valid manifest was rejected: %+v", got)
			}
			if name == "loss-only" && math.Abs(got.PPL-math.Exp(2)) > 1e-12 {
				t.Fatalf("loss-only PPL = %g, want %g", got.PPL, math.Exp(2))
			}
		})
	}
}

func TestReadBestDetectsDurableEvalContractReset(t *testing.T) {
	for _, marker := range []string{
		"best.before-explicit-resume-step-100",
		"best.before-loop-reset-step-100.1",
		"best.before-text-limit-384-to-768",
	} {
		t.Run(marker, func(t *testing.T) {
			run := t.TempDir()
			if err := os.Mkdir(filepath.Join(run, marker), 0o755); err != nil {
				t.Fatal(err)
			}
			got := readBest(run)
			if !got.ContractReset || got.Exists {
				t.Fatalf("reset marker was not authoritative: %+v", got)
			}
		})
	}
}

func TestEvalContractReceiptCoversMissingBestAndFreshStart(t *testing.T) {
	run := t.TempDir()
	marker := filepath.Join(run, "best.before-loop-reset-step-100")
	if err := os.Mkdir(marker, 0o755); err != nil {
		t.Fatal(err)
	}
	receipt := filepath.Join(run, "eval_contract_reset.json")
	if err := os.WriteFile(receipt, []byte(
		`{"schema":1,"reset":true,"step":100,"reasons":["loop_reset"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := readBest(run); !got.ContractReset || got.Exists {
		t.Fatalf("reset receipt did not cover absent best: %+v", got)
	}

	if err := os.WriteFile(receipt, []byte(
		`{"schema":1,"reset":true,"step":0,"reasons":["fresh"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := readBest(run); !got.ContractReset {
		t.Fatalf("fresh receipt did not suppress old eval history: %+v", got)
	}
}

func TestMalformedEvalContractReceiptFailsClosed(t *testing.T) {
	run := t.TempDir()
	if err := os.WriteFile(filepath.Join(run, "eval_contract_reset.json"),
		[]byte(`{"reset":false}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := readBest(run); !got.ContractReset {
		t.Fatalf("malformed present receipt failed open: %+v", got)
	}
}

func isFinitePositive(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}
