// Package trainvm provides the dashboard's read-only view of the native
// TrainVM journal. It deliberately exposes no mutation methods: lifecycle
// authority remains in the C++ controller.
package trainvm

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"
)

type Reader struct {
	db *sql.DB
}

const compiledPlanMaximumBytes = 2 << 20

// ReadModel is the dashboard's bounded TrainVM projection API. Production
// wiring uses the native gRPC authority; Reader remains only as an explicit
// legacy compatibility adapter for offline journals and migration tests.
type ReadModel interface {
	JournalID(context.Context) (string, error)
	Runs(context.Context) ([]Run, error)
	Run(context.Context, string) (Run, bool, error)
	CompiledPlan(context.Context, string) (CompiledPlanView, bool, error)
	Events(context.Context, EventQuery) ([]Event, error)
	Timeline(context.Context, string, uint64, int) ([]Event, error)
	Controls(context.Context, string) (ControlView, bool, error)
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

type CompiledPlanView struct {
	JournalID     string          `json:"journal_id"`
	RunID         string          `json:"run_id"`
	RunRevision   uint64          `json:"run_revision"`
	PlanHash      string          `json:"plan_hash"`
	CanonicalPlan json.RawMessage `json:"canonical_plan"`
}

type ControlDescriptor struct {
	Type              string   `json:"type"`
	Default           any      `json:"default"`
	Minimum           *float64 `json:"minimum,omitempty"`
	Maximum           *float64 `json:"maximum,omitempty"`
	Values            []any    `json:"values,omitempty"`
	Apply             string   `json:"apply"`
	MutableAfterStart bool     `json:"mutable_after_start"`
	RequiresPause     bool     `json:"requires_pause,omitempty"`
	Description       string   `json:"description,omitempty"`
	Unit              string   `json:"unit,omitempty"`
}

type ControlCommandView struct {
	CommandID       string          `json:"command_id"`
	ControlRevision uint64          `json:"control_revision"`
	ApplyPoint      string          `json:"apply_point"`
	Assignments     json.RawMessage `json:"assignments"`
	Author          string          `json:"author"`
	Reason          string          `json:"reason"`
	Status          string          `json:"status"`
	EffectiveStep   *uint64         `json:"effective_step,omitempty"`
	EffectiveValues json.RawMessage `json:"effective_values"`
	Diagnostics     json.RawMessage `json:"diagnostics"`
}

type ControlView struct {
	Catalog                 map[string]ControlDescriptor `json:"catalog"`
	EffectiveValues         map[string]any               `json:"effective_values"`
	LatestRequestedRevision uint64                       `json:"latest_requested_revision"`
	LatestEffectiveRevision uint64                       `json:"latest_effective_revision"`
	Commands                []ControlCommandView         `json:"commands"`
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

func (r *Reader) JournalID(ctx context.Context) (string, error) {
	var identity string
	if err := r.db.QueryRowContext(ctx,
		"SELECT value FROM journal_meta WHERE key='journal_id'").Scan(&identity); err != nil {
		return "", fmt.Errorf("read TrainVM journal identity: %w", err)
	}
	if len(identity) != 32 {
		return "", fmt.Errorf("TrainVM journal identity is malformed")
	}
	return identity, nil
}

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

func (r *Reader) CompiledPlan(ctx context.Context, runID string) (CompiledPlanView, bool, error) {
	if strings.TrimSpace(runID) == "" || len(runID) > 256 {
		return CompiledPlanView{}, false, &ValidationError{Message: "bounded run ID is required"}
	}
	transaction, err := r.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return CompiledPlanView{}, false, fmt.Errorf("begin TrainVM compiled-plan snapshot: %w", err)
	}
	defer transaction.Rollback()
	var view CompiledPlanView
	var canonical string
	err = transaction.QueryRowContext(ctx, `
		SELECT journal.value, run_projection.run_id, run_projection.run_revision,
		       run_projection.plan_hash, compiled_plans.canonical_plan_json
		FROM run_projection
		JOIN compiled_plans ON compiled_plans.plan_hash=run_projection.plan_hash
		JOIN journal_meta AS journal ON journal.key='journal_id'
		WHERE run_projection.run_id=?`, runID).
		Scan(&view.JournalID, &view.RunID, &view.RunRevision, &view.PlanHash, &canonical)
	if err == sql.ErrNoRows {
		return CompiledPlanView{}, false, nil
	}
	if err != nil {
		return CompiledPlanView{}, false, fmt.Errorf("read TrainVM compiled plan: %w", err)
	}
	if len(view.JournalID) != 32 || view.RunID != runID || view.RunRevision == 0 ||
		len(canonical) == 0 || len(canonical) > compiledPlanMaximumBytes || !json.Valid([]byte(canonical)) {
		return CompiledPlanView{}, false, fmt.Errorf("TrainVM compiled-plan snapshot is malformed")
	}
	actualHash := fmt.Sprintf("%x", sha256.Sum256([]byte(canonical)))
	if actualHash != view.PlanHash {
		return CompiledPlanView{}, false, fmt.Errorf(
			"TrainVM persisted plan failed integrity verification: expected %s, got %s",
			view.PlanHash, actualHash)
	}
	view.CanonicalPlan = json.RawMessage(canonical)
	if err := transaction.Commit(); err != nil {
		return CompiledPlanView{}, false, fmt.Errorf("commit TrainVM compiled-plan snapshot: %w", err)
	}
	return view, true, nil
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
	return r.Events(ctx, EventQuery{RunID: runID, After: after, Limit: limit})
}

func (r *Reader) Events(ctx context.Context, input EventQuery) ([]Event, error) {
	query, err := normalizeEventQuery(input)
	if err != nil {
		return nil, err
	}
	statement := `
		SELECT journal_sequence, event_id, run_id, run_revision, plan_revision,
		       node_id, attempt_id, worker_sequence, event_type, event_version,
		       wall_time_ns, monotonic_time_ns, optimizer_step, payload_json
		FROM events
		WHERE run_id=? AND journal_sequence>?`
	arguments := []any{query.RunID, query.After}
	if len(query.EventTypes) != 0 {
		statement += " AND event_type IN (" + strings.TrimRight(strings.Repeat("?,", len(query.EventTypes)), ",") + ")"
		for _, eventType := range query.EventTypes {
			arguments = append(arguments, eventType)
		}
	}
	statement += " ORDER BY journal_sequence LIMIT ?"
	arguments = append(arguments, query.Limit)
	rows, err := r.db.QueryContext(ctx, statement, arguments...)
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

func (r *Reader) Controls(ctx context.Context, runID string) (ControlView, bool, error) {
	transaction, err := r.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return ControlView{}, false, fmt.Errorf("begin TrainVM control snapshot: %w", err)
	}
	defer transaction.Rollback()
	var planHash, canonical string
	err = transaction.QueryRowContext(ctx, `
		SELECT compiled_plans.plan_hash, compiled_plans.canonical_plan_json
		FROM run_projection
		JOIN compiled_plans ON compiled_plans.plan_hash=run_projection.plan_hash
		WHERE run_projection.run_id=?`, runID).Scan(&planHash, &canonical)
	if err == sql.ErrNoRows {
		return ControlView{}, false, nil
	}
	if err != nil {
		return ControlView{}, false, fmt.Errorf("read TrainVM control catalog: %w", err)
	}
	actualHash := fmt.Sprintf("%x", sha256.Sum256([]byte(canonical)))
	if actualHash != planHash {
		return ControlView{}, false, fmt.Errorf(
			"TrainVM persisted plan failed integrity verification: expected %s, got %s",
			planHash, actualHash)
	}
	var document struct {
		Spec struct {
			Controls struct {
				Catalog map[string]ControlDescriptor `json:"catalog"`
			} `json:"controls"`
		} `json:"spec"`
	}
	if err := json.Unmarshal([]byte(canonical), &document); err != nil {
		return ControlView{}, false, fmt.Errorf("decode TrainVM control catalog: %w", err)
	}
	view := ControlView{
		Catalog:         document.Spec.Controls.Catalog,
		EffectiveValues: make(map[string]any, len(document.Spec.Controls.Catalog)),
		Commands:        []ControlCommandView{},
	}
	for key, descriptor := range view.Catalog {
		view.EffectiveValues[key] = descriptor.Default
	}
	if err := transaction.QueryRowContext(ctx, `
		SELECT COALESCE(MAX(control_revision),0),
		       COALESCE(MAX(CASE WHEN status='applied' THEN control_revision ELSE 0 END),0)
		FROM control_commands WHERE run_id=?`, runID).
		Scan(&view.LatestRequestedRevision, &view.LatestEffectiveRevision); err != nil {
		return ControlView{}, false, fmt.Errorf("read TrainVM control revisions: %w", err)
	}
	applied, err := transaction.QueryContext(ctx, `
		SELECT effective_values_json FROM control_commands
		WHERE run_id=? AND status='applied' ORDER BY control_revision`, runID)
	if err != nil {
		return ControlView{}, false, fmt.Errorf("read TrainVM effective controls: %w", err)
	}
	for applied.Next() {
		var encoded string
		if err := applied.Scan(&encoded); err != nil {
			applied.Close()
			return ControlView{}, false, fmt.Errorf("scan TrainVM effective controls: %w", err)
		}
		var values map[string]any
		if err := json.Unmarshal([]byte(encoded), &values); err != nil {
			applied.Close()
			return ControlView{}, false, fmt.Errorf("decode TrainVM effective controls: %w", err)
		}
		for key, value := range values {
			view.EffectiveValues[key] = value
		}
	}
	if err := applied.Err(); err != nil {
		applied.Close()
		return ControlView{}, false, fmt.Errorf("read TrainVM effective controls: %w", err)
	}
	if err := applied.Close(); err != nil {
		return ControlView{}, false, fmt.Errorf("close TrainVM effective controls: %w", err)
	}
	rows, err := transaction.QueryContext(ctx, `
		SELECT command_id, control_revision, apply_point, assignments_json, author,
		       reason, status, effective_step, effective_values_json, diagnostics_json
		FROM control_commands WHERE run_id=? ORDER BY control_revision DESC LIMIT 50`, runID)
	if err != nil {
		return ControlView{}, false, fmt.Errorf("read TrainVM control commands: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var command ControlCommandView
		var step sql.NullInt64
		var assignments, effective, diagnostics string
		if err := rows.Scan(&command.CommandID, &command.ControlRevision, &command.ApplyPoint,
			&assignments, &command.Author, &command.Reason, &command.Status, &step,
			&effective, &diagnostics); err != nil {
			return ControlView{}, false, fmt.Errorf("scan TrainVM control command: %w", err)
		}
		if !json.Valid([]byte(assignments)) || !json.Valid([]byte(effective)) ||
			!json.Valid([]byte(diagnostics)) {
			return ControlView{}, false, fmt.Errorf("TrainVM control command contains invalid JSON")
		}
		if step.Valid {
			value := uint64(step.Int64)
			command.EffectiveStep = &value
		}
		command.Assignments = json.RawMessage(assignments)
		command.EffectiveValues = json.RawMessage(effective)
		command.Diagnostics = json.RawMessage(diagnostics)
		view.Commands = append(view.Commands, command)
	}
	if err := rows.Err(); err != nil {
		return ControlView{}, false, fmt.Errorf("read TrainVM control commands: %w", err)
	}
	if err := rows.Close(); err != nil {
		return ControlView{}, false, fmt.Errorf("close TrainVM control commands: %w", err)
	}
	if err := transaction.Commit(); err != nil {
		return ControlView{}, false, fmt.Errorf("commit TrainVM control snapshot: %w", err)
	}
	return view, true, nil
}
