package db

import (
	"database/sql"
	"path/filepath"
	"testing"
)

func TestCursorFingerprintColumnsMigrateOnExistingDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.db")
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`CREATE TABLE ingest_cursors (
		path TEXT PRIMARY KEY, offset INTEGER, size INTEGER, mtime REAL)`); err != nil {
		t.Fatal(err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}

	database, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	if err := database.SaveCursor("run/train.jsonl", Cursor{
		Offset: 12, Size: 12, Mtime: 1, TailHash: "tail", FileID: "dev:ino",
	}); err != nil {
		t.Fatal(err)
	}
	cursor, err := database.GetCursor("run/train.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if cursor.TailHash != "tail" || cursor.FileID != "dev:ino" {
		t.Fatalf("fingerprint columns did not migrate: %+v", cursor)
	}
}

func TestEventGenerationMigratesOnExistingRunsTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.db")
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`CREATE TABLE runs (
		id INTEGER PRIMARY KEY, name TEXT UNIQUE, path TEXT, created_ts REAL,
		last_update_ts REAL, status TEXT, max_steps INTEGER, config_json TEXT,
		notes TEXT DEFAULT '', tags_json TEXT DEFAULT '[]')`); err != nil {
		t.Fatal(err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}

	database, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	if _, err := database.Exec(`INSERT INTO runs(name,event_generation)
		VALUES('vision',0)`); err != nil {
		t.Fatal(err)
	}
	var generation int
	if err := database.QueryRow(`SELECT event_generation FROM runs
		WHERE name='vision'`).Scan(&generation); err != nil {
		t.Fatal(err)
	}
	if generation != 0 {
		t.Fatalf("unexpected migrated generation: %d", generation)
	}
}
