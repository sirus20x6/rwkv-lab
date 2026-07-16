package db

import (
	"path/filepath"
	"testing"
)

func TestTimelineTimestampTieMapsToNewestStep(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "trainboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rid, err := d.EnsureRun("vision", "/tmp/vision", 1)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := d.Begin()
	if err != nil {
		t.Fatal(err)
	}
	// An initial ingester scan assigns the same filesystem mtime to every row.
	for _, step := range []int64{1, 50, 100} {
		if err := batch.Train(rid, TrainRow{Step: step, TS: 10}); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Commit(); err != nil {
		t.Fatal(err)
	}
	d.LogAction(11, "checkpoint", "vision", "{}", "requested", 0)
	timeline, err := d.GetTimeline("vision")
	if err != nil {
		t.Fatal(err)
	}
	if len(timeline.Events) != 1 || timeline.Events[0].Step != 100 {
		t.Fatalf("timestamp-only action mapped to arbitrary tied row: %+v", timeline.Events)
	}
}
