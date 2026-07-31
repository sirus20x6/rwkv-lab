// Package trainvm provides the dashboard's read-only view of the native
// TrainVM journal. It deliberately exposes no mutation methods: lifecycle
// authority remains in the C++ controller.
package trainvm

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"

	_ "modernc.org/sqlite"
)

type Reader struct {
	db *sql.DB
}

type Run struct {
	RunID            string `json:"run_id"`
	ExperimentName   string `json:"experiment_name"`
	PlanHash         string `json:"plan_hash"`
	DesiredState     string `json:"desired_state"`
	ObservedState    string `json:"observed_state"`
	CurrentNodeID    string `json:"current_node_id"`
	CurrentAttemptID string `json:"current_attempt_id"`
	RunRevision      uint64 `json:"run_revision"`
	OptimizerStep    uint64 `json:"optimizer_step"`
	LastHeartbeatNS  int64  `json:"last_heartbeat_ns"`
	LastEventSeq     uint64 `json:"last_event_sequence"`
	FailureSummary   string `json:"failure_summary"`
}

type Event struct {
	Sequence        uint64          `json:"sequence"`
	EventID         string          `json:"event_id"`
	RunID           string          `json:"run_id"`
	RunRevision     uint64          `json:"run_revision"`
	PlanRevision    uint64          `json:"plan_revision"`
	NodeID          string          `json:"node_id"`
	AttemptID       string          `json:"attempt_id"`
	WorkerSequence  uint64          `json:"worker_sequence"`
	EventType       string          `json:"event_type"`
	EventVersion    uint32          `json:"event_version"`
	WallTimeNS      int64           `json:"wall_time_ns"`
	MonotonicTimeNS uint64          `json:"monotonic_time_ns"`
	OptimizerStep   *uint64         `json:"optimizer_step,omitempty"`
	Payload         json.RawMessage `json:"payload"`
}

func Open(path string) (*Reader, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve TrainVM journal: %w", err)
	}
	uri := (&url.URL{Scheme: "file", Path: filepath.ToSlash(absolute)}).String()
	db, err := sql.Open("sqlite", uri+"?mode=ro&_pragma=query_only(ON)&_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, fmt.Errorf("open TrainVM journal: %w", err)
	}
	db.SetMaxOpenConns(4)
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("open TrainVM journal: %w", err)
	}
	return &Reader{db: db}, nil
}

func (r *Reader) Close() error { return r.db.Close() }

func (r *Reader) Runs(ctx context.Context) ([]Run, error) {
	rows, err := r.db.QueryContext(ctx, `
		SELECT run_id, experiment_name, plan_hash, desired_state, observed_state,
		       current_node_id, current_attempt_id, run_revision, optimizer_step,
		       last_heartbeat_ns, last_event_sequence, failure_summary
		FROM run_projection ORDER BY last_event_sequence DESC, run_id`)
	if err != nil {
		return nil, fmt.Errorf("list TrainVM runs: %w", err)
	}
	defer rows.Close()
	result := make([]Run, 0)
	for rows.Next() {
		var run Run
		if err := scanRun(rows, &run); err != nil {
			return nil, fmt.Errorf("scan TrainVM run: %w", err)
		}
		result = append(result, run)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list TrainVM runs: %w", err)
	}
	return result, nil
}

func (r *Reader) Run(ctx context.Context, runID string) (Run, bool, error) {
	row := r.db.QueryRowContext(ctx, `
		SELECT run_id, experiment_name, plan_hash, desired_state, observed_state,
		       current_node_id, current_attempt_id, run_revision, optimizer_step,
		       last_heartbeat_ns, last_event_sequence, failure_summary
		FROM run_projection WHERE run_id=?`, runID)
	var run Run
	if err := scanRun(row, &run); err != nil {
		if err == sql.ErrNoRows {
			return Run{}, false, nil
		}
		return Run{}, false, fmt.Errorf("read TrainVM run: %w", err)
	}
	return run, true, nil
}

type rowScanner interface {
	Scan(dest ...any) error
}

func scanRun(row rowScanner, run *Run) error {
	return row.Scan(&run.RunID, &run.ExperimentName, &run.PlanHash, &run.DesiredState,
		&run.ObservedState, &run.CurrentNodeID, &run.CurrentAttemptID, &run.RunRevision,
		&run.OptimizerStep, &run.LastHeartbeatNS, &run.LastEventSeq, &run.FailureSummary)
}

func (r *Reader) Timeline(ctx context.Context, runID string, after uint64, limit int) ([]Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 250
	}
	rows, err := r.db.QueryContext(ctx, `
		SELECT journal_sequence, event_id, run_id, run_revision, plan_revision,
		       node_id, attempt_id, worker_sequence, event_type, event_version,
		       wall_time_ns, monotonic_time_ns, optimizer_step, payload_json
		FROM events
		WHERE run_id=? AND journal_sequence>?
		ORDER BY journal_sequence LIMIT ?`, runID, after, limit)
	if err != nil {
		return nil, fmt.Errorf("read TrainVM timeline: %w", err)
	}
	defer rows.Close()
	result := make([]Event, 0)
	for rows.Next() {
		var event Event
		var step sql.NullInt64
		var payload string
		if err := rows.Scan(&event.Sequence, &event.EventID, &event.RunID, &event.RunRevision,
			&event.PlanRevision, &event.NodeID, &event.AttemptID, &event.WorkerSequence,
			&event.EventType, &event.EventVersion, &event.WallTimeNS, &event.MonotonicTimeNS,
			&step, &payload); err != nil {
			return nil, fmt.Errorf("scan TrainVM event: %w", err)
		}
		if step.Valid {
			value := uint64(step.Int64)
			event.OptimizerStep = &value
		}
		if !json.Valid([]byte(payload)) {
			return nil, fmt.Errorf("TrainVM event %q contains invalid payload JSON", event.EventID)
		}
		event.Payload = json.RawMessage(payload)
		result = append(result, event)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read TrainVM timeline: %w", err)
	}
	return result, nil
}
