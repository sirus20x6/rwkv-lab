package alerts

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

func alertMetric(value float64) *float64 { return &value }

func writeAlertRows(t *testing.T, database *db.DB, runID int64, trains []db.TrainRow, evals []db.EvalRow) {
	t.Helper()
	batch, err := database.Begin()
	if err != nil {
		t.Fatal(err)
	}
	defer batch.Rollback()
	for _, row := range trains {
		if err := batch.Train(runID, row); err != nil {
			t.Fatal(err)
		}
	}
	for _, row := range evals {
		if err := batch.Eval(runID, row); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Commit(); err != nil {
		t.Fatal(err)
	}
}

func TestEvalContractResetScopesDetectorToCurrentRows(t *testing.T) {
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
	runID, err := database.EnsureRun("vision", runDir, 1)
	if err != nil {
		t.Fatal(err)
	}

	// These future-step rows belong to an abandoned branch and are deliberately
	// left in SQLite to model the watcher catch-up window.
	stale := make([]db.TrainRow, 0, 50)
	for step := int64(101); step <= 150; step++ {
		stale = append(stale, db.TrainRow{
			Step: step, Gnorm: alertMetric(2000), Loss: alertMetric(1), TS: 100,
		})
	}
	writeAlertRows(t, database, runID, stale, []db.EvalRow{
		{Step: 200, PPL: alertMetric(1.25), TS: 150},
	})

	receiptPath := filepath.Join(runDir, "eval_contract_reset.json")
	if err := os.WriteFile(receiptPath, []byte(
		`{"schema":1,"reset":true,"step":100,"reasons":["loop_reset"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	publication := time.Unix(200, 0)
	if err := os.Chtimes(receiptPath, publication, publication); err != nil {
		t.Fatal(err)
	}
	receipt, present, valid := readEvalContractReset(runDir)
	if !present || !valid || receipt.Step != 100 || receipt.PublishedTS != 200 {
		t.Fatalf("receipt = %+v, present=%v valid=%v", receipt, present, valid)
	}

	detector := &Detector{db: database, runsDir: runs}
	fallback, err := database.RecentTrainStats(runID, 50)
	if err != nil {
		t.Fatal(err)
	}
	proc := sysmon.Proc{RunName: "vision"}
	if stats, reset, ready, cause := detector.currentTrainStats(proc, runID, fallback); ready || reset == nil || stats.N != 0 || cause != "" {
		t.Fatalf("stale reset startup failed open: stats=%+v reset=%+v ready=%v cause=%q", stats, reset, ready, cause)
	}

	current := make([]db.TrainRow, 0, 10)
	for step := int64(101); step <= 110; step++ {
		current = append(current, db.TrainRow{
			Step: step, Gnorm: alertMetric(5), Loss: alertMetric(0.9), TS: 250,
		})
	}
	writeAlertRows(t, database, runID, current, []db.EvalRow{
		{Step: 101, PPL: alertMetric(9), TS: 250},
		{Step: 102, PPL: alertMetric(8), TS: 250},
		{Step: 103, PPL: alertMetric(7), TS: 250},
	})
	stats, reset, ready, _ := detector.currentTrainStats(proc, runID, fallback)
	if !ready || reset == nil || stats.N != 10 || stats.LastStep != 110 || stats.MaxGnorm != 5 {
		t.Fatalf("current contract train window = %+v reset=%+v ready=%v", stats, reset, ready)
	}
	evals, err := detector.evalStats(runID, reset)
	if err != nil {
		t.Fatal(err)
	}
	if evals.N != 3 || evals.LastStep != 103 || evals.LastPPL != 7 || evals.MinPPL != 7 {
		t.Fatalf("abandoned eval contaminated current contract: %+v", evals)
	}

	// The same receipt persists across a normal process restart. Rows from the
	// previous PID remain valid history for eval, but cannot trigger live health
	// actions until this PID has produced a train record.
	proc.StartedTS = 300
	if stats, _, ready, cause := detector.currentTrainStats(proc, runID, fallback); ready || stats.N != 0 || cause != "" {
		t.Fatalf("previous PID train rows were treated as live: %+v ready=%v cause=%q", stats, ready, cause)
	}
	writeAlertRows(t, database, runID, []db.TrainRow{
		{Step: 111, Gnorm: alertMetric(4), Loss: alertMetric(0.8), TS: 310},
	}, nil)
	if stats, _, ready, _ := detector.currentTrainStats(proc, runID, fallback); !ready || stats.N != 1 || stats.LastStep != 111 {
		t.Fatalf("current PID evidence did not release detector: %+v ready=%v", stats, ready)
	}
}

func TestMalformedEvalContractReceiptFailsDetectorClosed(t *testing.T) {
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
	runID, err := database.EnsureRun("vision", runDir, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "eval_contract_reset.json"),
		[]byte(`{"schema":1,"reset":false}`), 0o600); err != nil {
		t.Fatal(err)
	}
	fallback := db.TrainStats{N: 50, MaxGnorm: 2000, LastStep: 900, LastTS: 900}
	detector := &Detector{db: database, runsDir: runs}
	stats, reset, ready, cause := detector.currentTrainStats(
		sysmon.Proc{RunName: "vision"}, runID, fallback)
	if ready || reset != nil || stats.N != 0 {
		t.Fatalf("malformed receipt failed open: stats=%+v reset=%+v ready=%v", stats, reset, ready)
	}
	if cause == "" {
		t.Fatalf("malformed receipt produced no suspend cause")
	}
}

func alertKinds(t *testing.T, database *db.DB) map[string]int {
	t.Helper()
	alerts, err := database.ActiveAlerts(100)
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[string]int{}
	for _, a := range alerts {
		kinds[a.Kind]++
	}
	return kinds
}

func TestStallAlertFiresWithoutIngestedEvidence(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()

	// Process running, zero rows ingested since StartedTS (hang before the
	// first log append), log stale beyond the stall threshold.
	logAge := stallSeconds + 60
	proc := sysmon.Proc{RunName: "vision", StartedTS: 100, LogAgeS: &logAge}
	detector := &Detector{db: database, runsDir: t.TempDir(), lastRaised: map[string]float64{}}
	detector.checkStall(proc, 0)
	if kinds := alertKinds(t, database); kinds["stall"] != 1 {
		t.Fatalf("stall alert missing for evidence-less hung process: %v", kinds)
	}
}

func TestMonitoringSuspendedAlertIsOneShotAndClears(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()

	proc := sysmon.Proc{RunName: "vision"}
	detector := &Detector{db: database, runsDir: t.TempDir(), lastRaised: map[string]float64{}}

	// Entering the fail-closed state raises exactly one alert per cause.
	detector.noteMonitoringGate(proc, "eval_contract_reset.json present but malformed/unreadable")
	detector.noteMonitoringGate(proc, "eval_contract_reset.json present but malformed/unreadable")
	if kinds := alertKinds(t, database); kinds["monitoring_suspended"] != 1 {
		t.Fatalf("suspend alert not one-shot: %v", kinds)
	}
	// The ordinary evidence gate (cause "") clears the state silently…
	detector.noteMonitoringGate(proc, "")
	if kinds := alertKinds(t, database); kinds["monitoring_suspended"] != 1 {
		t.Fatalf("clearing raised a spurious alert: %v", kinds)
	}
	// …and a later re-entry alerts again (cooldown key was reset on clear).
	detector.noteMonitoringGate(proc, "train stats query failed: disk I/O error")
	if kinds := alertKinds(t, database); kinds["monitoring_suspended"] != 2 {
		t.Fatalf("re-entry after recovery did not re-alert: %v", kinds)
	}
}
