package db

import (
	"path/filepath"
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
	if err := b.Train(rid, TrainRow{Step: 1, Loss: ptr(2), TS: 11}); err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, TrainRow{Step: 2, Loss: ptr(1), Extra: `{"codec_rel":0.125}`, TS: 12}); err != nil {
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
}
