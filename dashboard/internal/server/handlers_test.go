package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"trainboard/internal/db"
	"trainboard/internal/series"
)

func metric(v float64) *float64 { return &v }

func TestSeriesTipOverlapReturnsCorrectedRowWithoutCaching(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	rid, err := database.EnsureRun("vision", "/tmp/vision", 1)
	if err != nil {
		t.Fatal(err)
	}
	write := func(loss float64) {
		batch, beginErr := database.Begin()
		if beginErr != nil {
			t.Fatal(beginErr)
		}
		if rowErr := batch.Train(rid, db.TrainRow{Step: 5, Loss: metric(loss)}); rowErr != nil {
			t.Fatal(rowErr)
		}
		if commitErr := batch.Commit(); commitErr != nil {
			t.Fatal(commitErr)
		}
	}
	write(1)
	write(9) // same primary-key step: SQLite updates rather than appends

	s := &Server{cfg: Config{RunsDir: t.TempDir()}, db: database}
	req := httptest.NewRequest("GET", "/api/series/vision?train=loss&train_since=4", nil)
	req.SetPathValue("run", "vision")
	w := httptest.NewRecorder()
	s.handleSeries(w, req)
	if w.Code != 200 {
		t.Fatalf("series response %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("live series cache policy = %q", got)
	}
	var result series.Result
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	loss := result.Train.Cols["loss"]
	if len(result.Train.Step) != 1 || result.Train.Step[0] != 5 ||
		len(loss) != 1 || loss[0] == nil || *loss[0] != 9 {
		t.Fatalf("tip-overlap response missed corrected row: %+v", result.Train)
	}
}

func TestSeriesSuppressesAmbiguousBestMarkerAfterEvalContractReset(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	runs := t.TempDir()
	run := filepath.Join(runs, "vision")
	if err := os.Mkdir(run, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(run, "eval_contract_reset.json"), []byte(
		`{"schema":1,"reset":true,"step":100,"reasons":["loop_reset"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := database.EnsureRun("vision", run, 1); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RunsDir: runs}, db: database}
	req := httptest.NewRequest("GET", "/api/series/vision?eval=ppl", nil)
	req.SetPathValue("run", "vision")
	w := httptest.NewRecorder()
	s.handleSeries(w, req)
	if w.Code != 200 {
		t.Fatalf("series response %d: %s", w.Code, w.Body.String())
	}
	var result series.Result
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.SuppressBest {
		t.Fatal("reset series response still permits an ambiguous local best marker")
	}
}
