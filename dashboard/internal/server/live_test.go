package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"trainboard/internal/db"
)

func TestRunLiveRefreshesLatestAndBestEvalWithoutRunSwitch(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	runs := t.TempDir()
	runDir := filepath.Join(runs, "vision")
	if err := os.Mkdir(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	rid, err := database.EnsureRun("vision", runDir, 1)
	if err != nil {
		t.Fatal(err)
	}
	writeEval := func(step int64, ppl float64, touched float64) {
		t.Helper()
		batch, err := database.Begin()
		if err != nil {
			t.Fatal(err)
		}
		if err := batch.Eval(rid, db.EvalRow{Step: step, PPL: &ppl}); err != nil {
			batch.Rollback()
			t.Fatal(err)
		}
		if err := batch.Commit(); err != nil {
			t.Fatal(err)
		}
		if err := database.TouchRun(rid, touched); err != nil {
			t.Fatal(err)
		}
	}
	fetch := func() runLivePayload {
		t.Helper()
		req := httptest.NewRequest("GET", "/api/runs/vision/live", nil)
		req.SetPathValue("name", "vision")
		res := httptest.NewRecorder()
		s := &Server{cfg: Config{RunsDir: runs}, db: database}
		s.handleRunLive(res, req)
		if res.Code != 200 {
			t.Fatalf("live response %d: %s", res.Code, res.Body.String())
		}
		if got := res.Header().Get("Cache-Control"); got != "no-store" {
			t.Fatalf("live cache policy = %q", got)
		}
		var payload runLivePayload
		if err := json.Unmarshal(res.Body.Bytes(), &payload); err != nil {
			t.Fatal(err)
		}
		return payload
	}

	writeEval(100, 8.7, 2)
	first := fetch()
	if !strings.Contains(first.KPIHTML, ">8.700<") ||
		!strings.Contains(first.HeaderHTML, "best eval ppl 8.700 @ step 100") {
		t.Fatalf("first live payload missed eval: %+v", first)
	}
	// Append another eval while keeping the same selected run. The next live
	// request must carry the new latest value and winner without any selection
	// handler or stream reconnection.
	writeEval(200, 7.2, 3)
	second := fetch()
	if !strings.Contains(second.KPIHTML, ">7.200<") ||
		!strings.Contains(second.KPIHTML, "@ step 200") ||
		!strings.Contains(second.HeaderHTML, "best eval ppl 7.200 @ step 200") {
		t.Fatalf("second live payload stayed stale: %+v", second)
	}
	if second.Version <= first.Version {
		t.Fatalf("live version did not advance: %d -> %d", first.Version, second.Version)
	}

	// A later regression updates latest PPL while preserving the best header.
	writeEval(300, 9.1, 4)
	third := fetch()
	if !strings.Contains(third.KPIHTML, ">9.100<") ||
		!strings.Contains(third.KPIHTML, ">7.200<") ||
		!strings.Contains(third.HeaderHTML, "best eval ppl 7.200 @ step 200") {
		t.Fatalf("latest/best split is stale or conflated: %+v", third)
	}
}

func TestRenderKPIsUsesStablePatchTarget(t *testing.T) {
	ppl, best := 9.1, 7.2
	step, bestStep := int64(300), int64(200)
	html := renderKPIs(db.RunKPIs{
		Step: &step, PPL: &ppl, BestPPL: &best, BestPPLStep: &bestStep,
	})
	for _, want := range []string{`id="run-kpis"`, ">9.100<", ">7.200<", "@ step 200"} {
		if !strings.Contains(html, want) {
			t.Fatalf("KPI patch missing %q: %s", want, html)
		}
	}
}
