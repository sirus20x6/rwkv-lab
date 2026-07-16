package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"trainboard/internal/convboard"
	"trainboard/internal/db"
)

func TestRunHeaderShowsLiveDatabaseBestWithoutArtifact(t *testing.T) {
	ppl, step := 7.642, int64(3500)
	html := renderRunHeader(db.RunSummary{
		Name: "vision", BestPPL: &ppl, BestPPLStep: &step,
	}, nil, BestInfo{}, 0)
	if !strings.Contains(html, "best eval ppl 7.642 @ step 3500") {
		t.Fatalf("header does not contain live best: %s", html)
	}
}

func TestRunHeaderShowsDurableCheckpointThatPrecedesItsEvalLog(t *testing.T) {
	oldPPL, oldStep := 7.642, int64(3500)
	html := renderRunHeader(db.RunSummary{
		Name: "vision", BestPPL: &oldPPL, BestPPLStep: &oldStep,
	}, nil, BestInfo{Exists: true, PPL: 6.25, Step: 3600}, 0)
	if !strings.Contains(html, "checkpoint ppl 6.250 @ step 3600 · restartable") {
		t.Fatalf("durable winner was hidden behind stale DB rollup: %s", html)
	}
}

func TestRunHeaderDoesNotReplaceBetterDatabaseEvalWithStaleCheckpoint(t *testing.T) {
	ppl, step := 5.5, int64(4000)
	html := renderRunHeader(db.RunSummary{
		Name: "vision", BestPPL: &ppl, BestPPLStep: &step,
	}, nil, BestInfo{Exists: true, PPL: 6.25, Step: 3600}, 0)
	if !strings.Contains(html, "best eval ppl 5.500 @ step 4000") ||
		strings.Contains(html, "checkpoint ppl 6.250") {
		t.Fatalf("stale checkpoint displaced better DB eval: %s", html)
	}
}

func TestRunHeaderSuppressesAbandonedBestUntilResetContractHasWinner(t *testing.T) {
	ppl, step := 5.5, int64(4000)
	html := renderRunHeader(db.RunSummary{
		Name: "vision", BestPPL: &ppl, BestPPLStep: &step,
	}, nil, BestInfo{ContractReset: true}, 0)
	if !strings.Contains(html, "eval contract reset · no winner yet") ||
		strings.Contains(html, "best eval ppl 5.500") {
		t.Fatalf("abandoned winner survived contract reset: %s", html)
	}
}

func TestRunHeaderUsesActiveResetContractWinnerOverLowerAbandonedPPL(t *testing.T) {
	ppl, step := 5.5, int64(4000)
	html := renderRunHeader(db.RunSummary{
		Name: "vision", BestPPL: &ppl, BestPPLStep: &step,
	}, nil, BestInfo{ContractReset: true, Exists: true, PPL: 6.25, Step: 4200}, 0)
	if !strings.Contains(html, "best eval ppl 6.250 @ step 4200 · restartable") ||
		strings.Contains(html, "best eval ppl 5.500") {
		t.Fatalf("abandoned DB minimum displaced active winner: %s", html)
	}
}

func TestEvalContractResetSanitizesHeadlineSurfacesWithoutWinner(t *testing.T) {
	ppl, top1, step := 5.5, 0.9, int64(4000)
	summary := db.RunSummary{
		Name: "vision", LatestPPL: &ppl, LatestTop1: &top1,
		BestPPL: &ppl, BestPPLStep: &step, BestTop1: &top1,
		HasHorizons: true,
	}
	applyEvalContractSummary(&summary, BestInfo{ContractReset: true})
	if summary.LatestPPL != nil || summary.LatestTop1 != nil ||
		summary.BestPPL != nil || summary.BestPPLStep != nil ||
		summary.BestTop1 != nil || summary.HasHorizons {
		t.Fatalf("abandoned summary eval claims survived: %+v", summary)
	}
	if html := renderRunList([]db.RunSummary{summary}, nil, 0); strings.Contains(html, "5.50") {
		t.Fatalf("sidebar retained abandoned best: %s", html)
	}
	if html := renderLeaderboard([]lbRow{{Name: summary.Name, BestPPL: summary.BestPPL,
		LastPPL: summary.LatestPPL, BestTop1: summary.BestTop1}}); strings.Contains(html, "5.500") {
		t.Fatalf("leaderboard retained abandoned best: %s", html)
	}
	if ppl := conversionPPLForSummary(t, summary); ppl != nil {
		t.Fatalf("conversion board retained abandoned best: %g", *ppl)
	}

	kpi := db.RunKPIs{PPL: &ppl, Top1: &top1, BestPPL: &ppl,
		BestPPLStep: &step, BestTop1: &top1, BestTop1Step: &step}
	applyEvalContractKPIs(&kpi, BestInfo{ContractReset: true})
	if kpi.PPL != nil || kpi.Top1 != nil || kpi.BestPPL != nil ||
		kpi.BestPPLStep != nil || kpi.BestTop1 != nil || kpi.BestTop1Step != nil {
		t.Fatalf("abandoned KPI eval claims survived: %+v", kpi)
	}
}

func TestEvalContractResetPropagatesActiveWinnerAcrossHeadlineSurfaces(t *testing.T) {
	oldPPL, latestPPL, oldTop1, oldStep := 5.5, 7.0, 0.95, int64(4000)
	best := BestInfo{ContractReset: true, Exists: true, PPL: 6.25, Step: 4200}
	summary := db.RunSummary{Name: "vision", LatestPPL: &latestPPL,
		BestPPL: &oldPPL, BestPPLStep: &oldStep, BestTop1: &oldTop1}
	applyEvalContractSummary(&summary, best)
	if summary.BestPPL == nil || *summary.BestPPL != 6.25 ||
		summary.BestPPLStep == nil || *summary.BestPPLStep != 4200 ||
		summary.BestTop1 != nil || summary.LatestPPL == nil || *summary.LatestPPL != 7.0 {
		t.Fatalf("active winner was not propagated: %+v", summary)
	}
	if html := renderRunList([]db.RunSummary{summary}, nil, 0); !strings.Contains(html, "6.25") || strings.Contains(html, "5.50") {
		t.Fatalf("sidebar did not use active winner: %s", html)
	}
	if html := renderLeaderboard([]lbRow{{Name: summary.Name, BestPPL: summary.BestPPL,
		LastPPL: summary.LatestPPL, BestTop1: summary.BestTop1}}); !strings.Contains(html, "6.250") || strings.Contains(html, "5.500") {
		t.Fatalf("leaderboard did not use active winner: %s", html)
	}
	if ppl := conversionPPLForSummary(t, summary); ppl == nil || *ppl != 6.25 {
		t.Fatalf("conversion board did not use active winner: %v", ppl)
	}

	kpi := db.RunKPIs{PPL: &latestPPL, BestPPL: &oldPPL, BestPPLStep: &oldStep,
		BestTop1: &oldTop1, BestTop1Step: &oldStep}
	applyEvalContractKPIs(&kpi, best)
	if kpi.BestPPL == nil || *kpi.BestPPL != 6.25 ||
		kpi.BestPPLStep == nil || *kpi.BestPPLStep != 4200 ||
		kpi.BestTop1 != nil || kpi.BestTop1Step != nil {
		t.Fatalf("KPI strip did not use active winner: %+v", kpi)
	}
}

func conversionPPLForSummary(t *testing.T, summary db.RunSummary) *float64 {
	t.Helper()
	root := t.TempDir()
	database, err := db.Open(filepath.Join(root, "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	runStep := filepath.Join(root, "runs", summary.Name, "step_000001")
	if err := os.MkdirAll(runStep, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runStep, "config.json"),
		[]byte(`{"config":{"train_rwkv8_layers":"0"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	layers := convboard.Scan(database, filepath.Join(root, "lib"),
		filepath.Join(root, "runs"), []db.RunSummary{summary}, nil, 1)
	if len(layers) != 1 {
		t.Fatalf("conversion scan returned %d layers", len(layers))
	}
	return layers[0].PPL
}
