package db

import (
	"path/filepath"
	"testing"
)

func ptr(v float64) *float64 { return &v }

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
	if s.BestPPL == nil || *s.BestPPL != 3 || !s.HasHorizons {
		t.Fatalf("bad best/horizon: %+v", s)
	}
	codec, err := d.LatestCodecRelByRun()
	if err != nil || codec["r1"] == nil || *codec["r1"] != .125 {
		t.Fatalf("bad codec batch query: codec=%v err=%v", codec, err)
	}
}
