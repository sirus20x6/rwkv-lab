package ingest

import (
	"bytes"
	"context"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"trainboard/internal/db"
)

// Ingester incrementally tails every runs/*/train.jsonl into SQLite. It polls
// (no fsnotify dep) — stat'ing ~100 files per tick is negligible, and polling
// is robust to editor-style rewrites and NFS.
type Ingester struct {
	db       *db.DB
	runsDir  string
	interval time.Duration
}

// New builds an ingester. interval<=0 defaults to 1s.
func New(database *db.DB, runsDir string, interval time.Duration) *Ingester {
	if interval <= 0 {
		interval = time.Second
	}
	return &Ingester{db: database, runsDir: runsDir, interval: interval}
}

// Run does an initial full scan then polls until ctx is cancelled.
func (ig *Ingester) Run(ctx context.Context) {
	if n, err := ig.ScanOnce(); err != nil {
		log.Printf("[ingest] initial scan error: %v", err)
	} else if n > 0 {
		log.Printf("[ingest] initial scan ingested %d events", n)
	}
	t := time.NewTicker(ig.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if _, err := ig.ScanOnce(); err != nil {
				log.Printf("[ingest] scan error: %v", err)
			}
		}
	}
}

// ScanOnce ingests new bytes from every run's train.jsonl. Returns total events ingested.
func (ig *Ingester) ScanOnce() (int, error) {
	entries, err := os.ReadDir(ig.runsDir)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		runDir := filepath.Join(ig.runsDir, e.Name())
		jsonl := filepath.Join(runDir, "train.jsonl")
		n, err := ig.ingestFile(e.Name(), runDir, jsonl)
		if err != nil {
			log.Printf("[ingest] %s: %v", e.Name(), err)
			continue
		}
		total += n
	}
	return total, nil
}

func (ig *Ingester) ingestFile(name, runDir, jsonl string) (int, error) {
	st, err := os.Stat(jsonl)
	if err != nil {
		return 0, nil // no train.jsonl yet — not an error
	}
	size := st.Size()
	mtime := float64(st.ModTime().UnixNano()) / 1e9

	cur, err := ig.db.GetCursor(jsonl)
	if err != nil {
		return 0, err
	}
	// Truncation / rewrite (convert_train.py opens train.jsonl in "w" mode):
	// if the file shrank below where we'd read, start over. Upserts on
	// (run_id, step) make a full re-read idempotent.
	if size < cur.Offset {
		cur.Offset = 0
	}
	if cur.Offset == size {
		return 0, nil // nothing new
	}

	f, err := os.Open(jsonl)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	if _, err := f.Seek(cur.Offset, io.SeekStart); err != nil {
		return 0, err
	}
	chunk, err := io.ReadAll(f)
	if err != nil {
		return 0, err
	}
	// Only process up to the last complete line; a trailing partial line (writer
	// mid-append) waits for the next scan.
	lastNL := bytes.LastIndexByte(chunk, '\n')
	if lastNL < 0 {
		return 0, nil // no complete line yet
	}
	complete := chunk[:lastNL+1]

	runID, err := ig.db.EnsureRun(name, runDir, mtime)
	if err != nil {
		return 0, err
	}

	batch, err := ig.db.Begin()
	if err != nil {
		return 0, err
	}
	n := 0
	for _, line := range bytes.Split(complete, []byte{'\n'}) {
		ev, ok := parseLine(line, mtime)
		if !ok {
			continue
		}
		switch ev.Kind {
		case kindTrain:
			err = batch.Train(runID, ev.Train)
		case kindEval:
			err = batch.Eval(runID, ev.Eval)
		case kindCheckpoint:
			err = batch.Checkpoint(runID, ev.Ckpt)
		}
		if err != nil {
			batch.Rollback()
			return 0, err
		}
		n++
	}
	if err := batch.Commit(); err != nil {
		return 0, err
	}

	newOffset := cur.Offset + int64(lastNL+1)
	if err := ig.db.SaveCursor(jsonl, db.Cursor{Offset: newOffset, Size: size, Mtime: mtime}); err != nil {
		return 0, err
	}
	if err := ig.db.TouchRun(runID, mtime); err != nil {
		return 0, err
	}
	return n, nil
}
