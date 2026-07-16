package ingest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"trainboard/internal/db"
)

const cursorFingerprintBytes int64 = 4096

func tailHashAt(file *os.File, offset int64) (string, error) {
	if offset <= 0 {
		return "", nil
	}
	start := offset - cursorFingerprintBytes
	if start < 0 {
		start = 0
	}
	payload := make([]byte, offset-start)
	if _, err := file.ReadAt(payload, start); err != nil {
		return "", err
	}
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:]), nil
}

func fileIdentity(info os.FileInfo) string {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return ""
	}
	return strconv.FormatUint(uint64(stat.Dev), 16) + ":" +
		strconv.FormatUint(stat.Ino, 16)
}

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
	fileID := fileIdentity(st)

	cur, err := ig.db.GetCursor(jsonl)
	if err != nil {
		return 0, err
	}
	f, err := os.Open(jsonl)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	// Truncation / rewrite (convert_train.py opens train.jsonl in "w" mode).
	// Size alone is insufficient: a rewrite can regrow past the old offset
	// between polls. Verify the bytes immediately before the cursor as a file-
	// generation fingerprint; unchanged prefix means seeking remains valid.
	// A migrated cursor has no historical fingerprint. If its saved stat no
	// longer matches, conservatively replay the whole file once: blindly seeding
	// a hash from today's bytes could bless a rewrite that happened while the old
	// dashboard was offline and leave its prior SQLite rows permanently mixed in.
	legacyCursorChanged := cur.Offset > 0 && (cur.TailHash == "" || cur.FileID == "") &&
		(size != cur.Size || mtime != cur.Mtime)
	rewritten := size < cur.Offset || legacyCursorChanged ||
		(cur.FileID != "" && fileID != "" && cur.FileID != fileID)
	if !rewritten && cur.Offset > 0 && cur.TailHash != "" {
		actualTail, hashErr := tailHashAt(f, cur.Offset)
		if hashErr != nil {
			return 0, hashErr
		}
		rewritten = actualTail != cur.TailHash
	}
	var runID int64
	if rewritten {
		runID, err = ig.db.EnsureRun(name, runDir, mtime)
		if err != nil {
			return 0, err
		}
		// Commit the new file generation even when it is currently empty or
		// contains only a partial first line. Otherwise the old nonzero cursor
		// survives the early return below and causes a full reset every poll.
		cur = db.Cursor{Size: size, Mtime: mtime, FileID: fileID}
		// Publish the reset immediately, including when the replacement log is
		// empty or currently ends in a partial line. Otherwise open browsers can
		// retain abandoned future points until another complete event is written.
		// Cursor + revision are one commit so a crash cannot acknowledge this file
		// generation without also waking already-open browsers.
		if err := ig.db.ResetRunEventsAndPublish(
			runID, mtime, jsonl, cur); err != nil {
			return 0, err
		}
	}
	// Transparently seed fingerprints for cursors created by older schemas.
	if cur.Offset > 0 && (cur.TailHash == "" || cur.FileID == "") {
		cur.TailHash, err = tailHashAt(f, cur.Offset)
		if err != nil {
			return 0, err
		}
		cur.Size, cur.Mtime, cur.FileID = size, mtime, fileID
		if err := ig.db.SaveCursor(jsonl, cur); err != nil {
			return 0, err
		}
	}
	if cur.Offset == size {
		return 0, nil // nothing new
	}
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

	if runID == 0 {
		runID, err = ig.db.EnsureRun(name, runDir, mtime)
		if err != nil {
			return 0, err
		}
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
	tailHash, err := tailHashAt(f, newOffset)
	if err != nil {
		return 0, err
	}
	if err := ig.db.PublishCursor(runID, mtime, jsonl, db.Cursor{
		Offset: newOffset, Size: size, Mtime: mtime, TailHash: tailHash,
		FileID: fileID,
	}); err != nil {
		return 0, err
	}
	return n, nil
}
