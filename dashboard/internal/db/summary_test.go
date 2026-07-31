package db

import (
	"database/sql"
	"errors"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func ptr(v float64) *float64 { return &v }

func TestTouchRunAlwaysAdvancesBrowserRevision(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	const epoch = 1_784_000_000.0
	rid, err := d.EnsureRun("r1", "/tmp/r1", epoch)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.TouchRun(rid, epoch); err != nil {
		t.Fatal(err)
	}
	var first float64
	if err := d.QueryRow(`SELECT last_update_ts FROM runs WHERE id=?`, rid).Scan(&first); err != nil {
		t.Fatal(err)
	}
	if err := d.TouchRun(rid, 10); err != nil {
		t.Fatal(err)
	}
	var second float64
	if err := d.QueryRow(`SELECT last_update_ts FROM runs WHERE id=?`, rid).Scan(&second); err != nil {
		t.Fatal(err)
	}
	if !(second > first) {
		t.Fatalf("restored older mtime did not advance revision: %f -> %f", first, second)
	}
	if int64(second*1000) <= int64(first*1000) {
		t.Fatalf("millisecond browser revision did not advance: %d -> %d",
			int64(first*1000), int64(second*1000))
	}
}

func TestBrowserRevisionClampsSubTenMillisecondMtimes(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	const epoch = 1_784_000_000.0
	rid, err := d.EnsureRun("r1", "/tmp/r1", epoch)
	if err != nil {
		t.Fatal(err)
	}

	if err := d.TouchRun(rid, epoch+0.0004); err != nil {
		t.Fatal(err)
	}
	var touched float64
	if err := d.QueryRow(`SELECT last_update_ts FROM runs WHERE id=?`, rid).Scan(&touched); err != nil {
		t.Fatal(err)
	}
	if touched-epoch < 0.009 {
		t.Fatalf("sub-millisecond TouchRun advanced only %.9fs", touched-epoch)
	}

	path := filepath.Join(t.TempDir(), "train.jsonl")
	if err := d.PublishCursor(rid, touched+0.0004, path, Cursor{
		Offset: 12, Size: 12, Mtime: touched + 0.0004,
	}); err != nil {
		t.Fatal(err)
	}
	var published float64
	if err := d.QueryRow(`SELECT last_update_ts FROM runs WHERE id=?`, rid).Scan(&published); err != nil {
		t.Fatal(err)
	}
	if published-touched < 0.009 {
		t.Fatalf("sub-millisecond PublishCursor advanced only %.9fs", published-touched)
	}
	if int64(published*1000) <= int64(touched*1000) {
		t.Fatalf("published browser revision did not advance: %d -> %d",
			int64(touched*1000), int64(published*1000))
	}
}

func TestPublishCursorDoesNotCommitWithoutRunRevision(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	const path = "/tmp/missing/train.jsonl"
	if err := d.PublishCursor(999, 10, path, Cursor{Offset: 12, Size: 12}); err == nil {
		t.Fatal("publishing a cursor for an unknown run unexpectedly succeeded")
	}
	if cursor, err := d.GetCursor(path); err != nil {
		t.Fatal(err)
	} else if cursor != (Cursor{}) {
		t.Fatalf("cursor committed without its matching run revision: %+v", cursor)
	}
}

func TestReplacementBatchNeverExposesTransientEmptyRun(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 10)
	if err != nil {
		t.Fatal(err)
	}
	old, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := old.Train(rid, TrainRow{Step: 1, Loss: ptr(1), TS: 10}); err != nil {
		t.Fatal(err)
	}
	if err := old.Commit(); err != nil {
		t.Fatal(err)
	}

	replacement, err := d.BeginReplacement(rid)
	if err != nil {
		t.Fatal(err)
	}
	if err := replacement.Train(rid, TrainRow{Step: 9, Loss: ptr(9), TS: 11}); err != nil {
		replacement.Rollback()
		t.Fatal(err)
	}
	// WAL readers remain on the complete committed generation while the reset
	// and replacement inserts are still in flight.
	var count, step int
	if err := d.QueryRow(`SELECT count(*),max(step) FROM train_events WHERE run_id=?`, rid).
		Scan(&count, &step); err != nil {
		replacement.Rollback()
		t.Fatal(err)
	}
	if count != 1 || step != 1 {
		replacement.Rollback()
		t.Fatalf("reader observed partial replacement: count=%d step=%d", count, step)
	}
	if err := replacement.CommitAndPublish(rid, 11, "/tmp/vision/train.jsonl",
		Cursor{Offset: 100, Size: 100, Mtime: 11, TailHash: "new", FileID: "1:2"}); err != nil {
		t.Fatal(err)
	}
	var generation int
	if err := d.QueryRow(`SELECT count(*),max(step),
		(SELECT event_generation FROM runs WHERE id=?)
		FROM train_events WHERE run_id=?`, rid, rid).Scan(&count, &step, &generation); err != nil {
		t.Fatal(err)
	}
	if count != 1 || step != 9 || generation != 1 {
		t.Fatalf("replacement did not commit as one generation: count=%d step=%d generation=%d",
			count, step, generation)
	}
}

func TestRunSummaryRollups(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("r1", "/tmp/r1", 10)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, TrainRow{
		Step: 1, Loss: ptr(2),
		Extra: `{"engram_inj_rms":0.005,"engram_recall_valid_rate":1}`,
		TS:    11,
	}); err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, TrainRow{
		Step: 2, Loss: ptr(1),
		Extra: `{"codec_rel":0.125,"engram_inj_rms":0,"engram_recall_valid_rate":0}`,
		TS:    12,
	}); err != nil {
		t.Fatal(err)
	}
	if err := b.Eval(rid, EvalRow{Step: 2, PPL: ptr(4), Top1: ptr(.2), Extra: `{"h4_top1":0.1}`, TS: 12}); err != nil {
		t.Fatal(err)
	}
	if err := b.Checkpoint(rid, CkptRow{Step: 2}); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}

	// Updating an existing primary-key row must not inflate counts.
	b, _ = d.Begin()
	if err := b.Eval(rid, EvalRow{Step: 2, PPL: ptr(3), Top1: ptr(.3), Extra: `{"h4_top1":0.2}`, TS: 13}); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}

	rows, err := d.RunSummaries(13)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d summaries", len(rows))
	}
	s := rows[0]
	if s.NTrain != 2 || s.NEval != 1 || s.NCkpt != 1 {
		t.Fatalf("bad counts: %+v", s)
	}
	if s.LatestStep == nil || *s.LatestStep != 2 || s.LatestPPL == nil || *s.LatestPPL != 3 {
		t.Fatalf("bad latest values: %+v", s)
	}
	if s.BestPPL == nil || *s.BestPPL != 3 || s.BestPPLStep == nil || *s.BestPPLStep != 2 || !s.HasHorizons {
		t.Fatalf("bad best/horizon: %+v", s)
	}
	single, found, err := d.RunSummaryByName("r1", 13)
	if err != nil || !found {
		t.Fatalf("single-run summary missing: found=%v err=%v", found, err)
	}
	if single.Name != s.Name || single.NTrain != s.NTrain || single.NEval != s.NEval ||
		single.LatestStep == nil || *single.LatestStep != *s.LatestStep ||
		single.LatestPPL == nil || *single.LatestPPL != *s.LatestPPL ||
		single.BestPPL == nil || *single.BestPPL != *s.BestPPL ||
		single.BestPPLStep == nil || *single.BestPPLStep != *s.BestPPLStep ||
		single.HasHorizons != s.HasHorizons || single.Status != s.Status {
		t.Fatalf("single-run summary differs from shared rollup: all=%+v single=%+v", s, single)
	}
	if _, found, err := d.RunSummaryByName("missing", 13); err != nil || found {
		t.Fatalf("missing single-run summary: found=%v err=%v", found, err)
	}
	// The browser calls this path every second. Guard against accidentally
	// turning its name lookup back into an all-runs scan.
	namePlan, err := d.Query(`EXPLAIN QUERY PLAN `+runSummarySelect+` WHERE r.name=?`, "r1")
	if err != nil {
		t.Fatal(err)
	}
	usedNameIndex := false
	for namePlan.Next() {
		var id, parent, unused int
		var detail string
		if err := namePlan.Scan(&id, &parent, &unused, &detail); err != nil {
			namePlan.Close()
			t.Fatal(err)
		}
		if strings.Contains(detail, "SEARCH r USING INDEX") && strings.Contains(detail, "name") {
			usedNameIndex = true
		}
	}
	if err := namePlan.Close(); err != nil {
		t.Fatal(err)
	}
	if !usedNameIndex {
		t.Fatal("single-run summary does not use the unique runs.name index")
	}
	codec, err := d.LatestCodecRelByRun()
	if err != nil || codec["r1"] == nil || *codec["r1"] != .125 {
		t.Fatalf("bad codec batch query: codec=%v err=%v", codec, err)
	}
	plan, err := d.Query(`EXPLAIN QUERY PLAN SELECT json_extract(extra_json,'$.codec_rel')
		FROM train_events INDEXED BY idx_train_codec_rel WHERE run_id=?
		  AND json_extract(extra_json,'$.codec_rel') IS NOT NULL
		ORDER BY step DESC LIMIT 1`, rid)
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	usedCodecIndex := false
	for plan.Next() {
		var id, parent, unused int
		var detail string
		if err := plan.Scan(&id, &parent, &unused, &detail); err != nil {
			t.Fatal(err)
		}
		if strings.Contains(detail, "idx_train_codec_rel") {
			usedCodecIndex = true
		}
	}
	if !usedCodecIndex {
		t.Fatal("latest codec lookup does not use its partial index")
	}
	health, err := d.RecentTrainStatsByName([]string{"r1", "missing"}, 50)
	if err != nil || health["r1"].RunID != rid || health["r1"].Stats.N != 2 ||
		health["r1"].Stats.LastStep != 2 || health["r1"].Stats.CodecRel == nil {
		t.Fatalf("bad batched health query: health=%v err=%v", health, err)
	}
	stats := health["r1"].Stats
	if len(stats.EngramInjWindow) != 1 || stats.EngramInjWindow[0] != .005 ||
		stats.EngramInjRMS == nil || *stats.EngramInjRMS != .005 {
		t.Fatalf("zero-recall row contaminated Engram health window: %+v", stats)
	}
}

// TestEngramWindowPopulatesWithoutARecallRateKey is the regression guard for
// the alert that was UNFIRABLE for every non-vision run. Only
// src/rwkv_lab/vision_train.py emits engram_recall_valid_rate; the Engram-LMB /
// GDN-conversion runs go through grokking_metrics.injection_stats(), which
// emits engram_inj_rms with no recall key at all. Gating the window on a
// present recall rate left those windows empty forever, so the detector could
// never fire for the exact run class the alert exists for.
func TestEngramWindowPopulatesWithoutARecallRateKey(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("convert_L3", "/tmp/convert_L3", 10)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	for step := 1; step <= 12; step++ {
		// injection_stats() shape: rosa + engram RMS, no recall statistics.
		if err := b.Train(rid, TrainRow{
			Step: int64(step), Loss: ptr(2),
			Extra: `{"rosa_inj_rms":0.02,"engram_inj_rms":1e-9}`,
			TS:    float64(10 + step),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}

	stats, err := d.RecentTrainStats(rid, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats.EngramInjWindow) != 12 || stats.EngramInjRMS == nil {
		t.Fatalf("recall-rate-free Engram rows produced an empty window: %+v", stats)
	}
	if len(stats.RosaInjWindow) != 12 {
		t.Fatalf("ROSA window regressed: %+v", stats)
	}
	batched, err := d.RecentTrainStatsByName([]string{"convert_L3"}, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(batched["convert_L3"].Stats.EngramInjWindow) != 12 {
		t.Fatalf("batched query dropped the recall-rate-free window: %+v",
			batched["convert_L3"].Stats)
	}
}

// TestNearZeroRecallRateIsNotUsableEngramEvidence pins the threshold fix:
// engram_recall_valid_rate is a FRACTION (count / valid.numel()), so a batch
// where a handful of thousands of positions recalled anything passes a bare >0
// gate while contributing a legitimately near-zero injection RMS -- biasing the
// window toward a false memory_path_dead.
func TestNearZeroRecallRateIsNotUsableEngramEvidence(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 10)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	rows := []string{
		`{"engram_inj_rms":0.5,"engram_recall_valid_rate":0.9}`,
		// 1 of 4096 positions recalled: passes >0, evidence about nothing.
		`{"engram_inj_rms":1e-9,"engram_recall_valid_rate":0.000244}`,
		`{"engram_inj_rms":0,"engram_recall_valid_rate":0}`,
	}
	for i, extra := range rows {
		if err := b.Train(rid, TrainRow{
			Step: int64(i + 1), Loss: ptr(2), Extra: extra, TS: float64(11 + i),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}
	stats, err := d.RecentTrainStats(rid, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats.EngramInjWindow) != 1 || stats.EngramInjWindow[0] != 0.5 {
		t.Fatalf("near-zero recall rows contaminated the window: %+v", stats.EngramInjWindow)
	}
}

// TestCodecRelWindowCarriesTheWholeWindow guards finding 9: codec_collapse is
// CRITICAL (it can SIGINT the run under auto-stop) and used to be judged from
// the newest row alone.
func TestCodecRelWindowCarriesTheWholeWindow(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("conv", "/tmp/conv", 10)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	for i, codec := range []float64{0.11, 0.12, 0.10, 0.9} { // newest row is the spike
		if err := b.Train(rid, TrainRow{
			Step: int64(i + 1), Loss: ptr(2),
			Extra: `{"codec_rel":` + strconv.FormatFloat(codec, 'f', -1, 64) + `}`,
			TS:    float64(11 + i),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}
	stats, err := d.RecentTrainStats(rid, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats.CodecRelWindow) != 4 {
		t.Fatalf("codec window = %v", stats.CodecRelWindow)
	}
	if stats.CodecRel == nil || *stats.CodecRel != 0.9 {
		t.Fatalf("latest codec_rel = %v", stats.CodecRel)
	}
	batched, err := d.RecentTrainStatsByName([]string{"conv"}, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(batched["conv"].Stats.CodecRelWindow) != 4 {
		t.Fatalf("batched codec window = %v", batched["conv"].Stats.CodecRelWindow)
	}
}

// TestRunKPIScanErrorsAreSurfaced guards finding 10: the KPI SELECTs carry a
// dozen-plus json_extract columns whose count must match their Scan list, and a
// discarded error there renders EVERY KPI as "—" with nothing in the log.
func TestRunKPIScanErrorsAreSurfaced(t *testing.T) {
	if err := kpiScan("latest eval", "r1", sql.ErrNoRows); err != nil {
		t.Fatalf("a run with no rows of this kind must not be an error: %v", err)
	}
	if err := kpiScan("latest eval", "r1", nil); err != nil {
		t.Fatalf("clean scan reported an error: %v", err)
	}
	arity := errors.New("sql: expected 15 destination arguments in Scan, not 14")
	err := kpiScan("latest eval", "r1", arity)
	if err == nil {
		t.Fatal("a scan arity mismatch was swallowed")
	}
	if !errors.Is(err, arity) || !strings.Contains(err.Error(), "latest eval") ||
		!strings.Contains(err.Error(), `"r1"`) {
		t.Fatalf("surfaced error lost its cause or context: %v", err)
	}

	// A run with no events at all must still succeed (pure ErrNoRows path).
	d, dbErr := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if dbErr != nil {
		t.Fatal(dbErr)
	}
	defer d.Close()
	if _, dbErr = d.EnsureRun("empty", "/tmp/empty", 10); dbErr != nil {
		t.Fatal(dbErr)
	}
	k, found, dbErr := d.RunKPIsByName("empty")
	if dbErr != nil || !found {
		t.Fatalf("empty run KPIs: found=%v err=%v", found, dbErr)
	}
	if k.Step != nil || k.PPL != nil {
		t.Fatalf("empty run reported values: %+v", k)
	}
}
