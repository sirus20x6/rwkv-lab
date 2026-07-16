package series

import (
	"path/filepath"
	"slices"
	"testing"

	"trainboard/internal/db"
)

func fptr(v float64) *float64 { return &v }

func TestFetchCursorsDoesNotSkipLateEvalAtTrainStep(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	for step := int64(1); step <= 3; step++ {
		if err := b.Train(rid, db.TrainRow{Step: step, Loss: fptr(float64(step))}); err != nil {
			t.Fatal(err)
		}
	}
	// This eval is appended after train step 2. A shared since=2 cursor loses it.
	if err := b.Eval(rid, db.EvalRow{Step: 2, Loss: fptr(1.5), PPL: fptr(4.5)}); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}

	got, err := FetchCursors(d, rid, []string{"loss"}, []string{"ppl"}, 2, -1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Train.Step) != 1 || got.Train.Step[0] != 3 {
		t.Fatalf("train increment = %v, want [3]", got.Train.Step)
	}
	if len(got.Eval.Step) != 1 || got.Eval.Step[0] != 2 {
		t.Fatalf("eval increment = %v, want [2]", got.Eval.Step)
	}
	if got.MaxTrainStep != 3 || got.MaxEvalStep != 2 || got.MaxStep != 3 {
		t.Fatalf("bad cursor tips: %+v", got)
	}
}

func TestIncrementalTipsRemainAuthoritativeAfterCursorRunsPastData(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, db.TrainRow{Step: 40, Loss: fptr(1)}); err != nil {
		t.Fatal(err)
	}
	if err := b.Eval(rid, db.EvalRow{Step: 30, PPL: fptr(4)}); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}

	got, err := FetchCursors(d, rid, []string{"loss"}, []string{"ppl"},
		100, 100, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Train.Step) != 0 || len(got.Eval.Step) != 0 {
		t.Fatalf("past-tip increment unexpectedly returned rows: %+v", got)
	}
	if got.MaxTrainStep != 40 || got.MaxEvalStep != 30 || got.MaxStep != 40 {
		t.Fatalf("tips describe the empty increment instead of the tables: %+v", got)
	}
}

func TestEmptyTableTipsAreDistinctFromStepZero(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 1)
	if err != nil {
		t.Fatal(err)
	}

	empty, err := FetchCursors(d, rid, []string{"loss"}, []string{"ppl"},
		0, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if empty.MaxTrainStep != -1 || empty.MaxEvalStep != -1 || empty.MaxStep != 0 {
		t.Fatalf("empty table tips are ambiguous: %+v", empty)
	}

	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, db.TrainRow{Step: 0, Loss: fptr(1)}); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}
	stepZero, err := FetchCursors(d, rid, []string{"loss"}, []string{"ppl"},
		-1, -1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if stepZero.MaxTrainStep != 0 || stepZero.MaxEvalStep != -1 ||
		stepZero.MaxStep != 0 || stepZero.Generation != 0 {
		t.Fatalf("step-zero tips do not describe the table: %+v", stepZero)
	}

	if err := d.ResetRunEvents(rid); err != nil {
		t.Fatal(err)
	}
	rewritten, err := FetchCursors(d, rid, []string{"loss"}, []string{"ppl"},
		0, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten.MaxTrainStep != -1 || rewritten.Generation != 1 {
		t.Fatalf("event reset generation was not exposed: %+v", rewritten)
	}
}

func TestCustomMetricCatalogAndFetchRejectNonNumericJSON(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, db.TrainRow{Step: 1, Extra: `{
		"numeric":1.5,"mixed":2,"flag":true,"label":"factored","nested":{"x":1}}`}); err != nil {
		t.Fatal(err)
	}
	if err := b.Train(rid, db.TrainRow{Step: 2, Extra: `{
		"numeric":2.5,"mixed":"not-a-number","flag":false,"values":[1,2]}`}); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(); err != nil {
		t.Fatal(err)
	}

	cat, err := Catalog(d, rid)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"numeric", "mixed", "flag"} {
		if !slices.Contains(cat["train"], key) {
			t.Fatalf("numeric custom key %q missing from catalog: %v", key, cat["train"])
		}
	}
	for _, key := range []string{"label", "nested", "values"} {
		if slices.Contains(cat["train"], key) {
			t.Fatalf("non-numeric custom key %q leaked into catalog: %v", key, cat["train"])
		}
	}

	// Duplicate requested fields previously appended twice into the same output
	// column, while a string in one row made scanning the entire response fail.
	got, err := FetchCursors(d, rid,
		[]string{"numeric", "mixed", "mixed", "flag"}, nil, -1, -1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Train.Step) != 2 || len(got.Train.Cols["mixed"]) != 2 {
		t.Fatalf("misaligned custom series: %+v", got.Train)
	}
	if got.Train.Cols["mixed"][0] == nil || *got.Train.Cols["mixed"][0] != 2 ||
		got.Train.Cols["mixed"][1] != nil {
		t.Fatalf("mixed numeric/string values were not safely nulled: %v", got.Train.Cols["mixed"])
	}
	if *got.Train.Cols["flag"][0] != 1 || *got.Train.Cols["flag"][1] != 0 {
		t.Fatalf("boolean metric did not map to 1/0: %v", got.Train.Cols["flag"])
	}
}
