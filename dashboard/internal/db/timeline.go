package db

import (
	"database/sql"
	"fmt"
	"sort"
)

// TimelineEvent is one point-in-time marker on a run's charts (a checkpoint,
// alert, applied control override, or operator action). Step is the x-position;
// for ts-only sources (actions, step-less alerts) it is mapped to the nearest
// train step server-side so the client can draw without a ts->step table.
type TimelineEvent struct {
	Step     int64   `json:"step"`
	Ts       float64 `json:"ts,omitempty"`
	Type     string  `json:"type"`               // checkpoint|alert|control|action
	Kind     string  `json:"kind"`               // reason / alert-kind / control-key / action-kind
	Severity string  `json:"severity,omitempty"` // alerts: warn|critical
	Label    string  `json:"label"`              // short chip text
	Detail   string  `json:"detail,omitempty"`   // hover/list detail
}

// Timeline bundles a run's markers, ordered by step then type.
type Timeline struct {
	Events []TimelineEvent `json:"events"`
}

// stepAtTs maps a wall-clock ts to the most recent train step at/just before it.
// Returns 0 if no train row precedes ts (marker lands at the chart origin).
func (d *DB) stepAtTs(runID int64, ts float64) int64 {
	var step sql.NullInt64
	_ = d.QueryRow(
		`SELECT step FROM train_events WHERE run_id=? AND ts<=?
		 ORDER BY ts DESC, step DESC LIMIT 1`,
		runID, ts).Scan(&step)
	if step.Valid {
		return step.Int64
	}
	return 0
}

// GetTimeline collects checkpoint/alert/control/action markers for one run.
func (d *DB) GetTimeline(runName string) (*Timeline, error) {
	runID, ok, err := d.RunID(runName)
	if err != nil {
		return nil, err
	}
	tl := &Timeline{Events: []TimelineEvent{}}
	if !ok {
		return tl, nil
	}

	// checkpoints (have step)
	if rows, err := d.Query(
		`SELECT step, COALESCE(reason,''), COALESCE(size_bytes,0)
		 FROM checkpoints WHERE run_id=? ORDER BY step`, runID); err == nil {
		for rows.Next() {
			var step, size int64
			var reason string
			if rows.Scan(&step, &reason, &size) == nil {
				e := TimelineEvent{Step: step, Type: "checkpoint", Kind: reason, Label: "ckpt"}
				if reason != "" {
					e.Detail = reason
				}
				if size > 0 {
					e.Detail = fmt.Sprintf("%s · %.1f GB", e.Detail, float64(size)/1e9)
				}
				tl.Events = append(tl.Events, e)
			}
		}
		rows.Close()
	}

	// alerts (step or ts). NOTE: ts->step mapping happens AFTER rows.Close() —
	// with a single-connection pool, calling stepAtTs (a nested query) while
	// these rows are still open self-deadlocks the entire DB.
	if rows, err := d.Query(
		`SELECT COALESCE(step,0), COALESCE(ts,0), kind, COALESCE(severity,''), COALESCE(message,'')
		 FROM alerts WHERE run_name=? ORDER BY ts`, runName); err == nil {
		start := len(tl.Events)
		for rows.Next() {
			var step int64
			var ts float64
			var kind, sev, msg string
			if rows.Scan(&step, &ts, &kind, &sev, &msg) == nil {
				tl.Events = append(tl.Events, TimelineEvent{
					Step: step, Ts: ts, Type: "alert", Kind: kind,
					Severity: sev, Label: kind, Detail: msg,
				})
			}
		}
		rows.Close()
		for i := start; i < len(tl.Events); i++ {
			if e := &tl.Events[i]; e.Step <= 0 && e.Ts > 0 {
				e.Step = d.stepAtTs(runID, e.Ts)
			}
		}
	}

	// applied control overrides (applied_step)
	if rows, err := d.Query(
		`SELECT key, value, applied_step, COALESCE(applied_ts,0)
		 FROM run_controls WHERE run_name=? AND applied_step IS NOT NULL ORDER BY applied_step`,
		runName); err == nil {
		for rows.Next() {
			var key string
			var val float64
			var step int64
			var ts float64
			if rows.Scan(&key, &val, &step, &ts) == nil {
				tl.Events = append(tl.Events, TimelineEvent{
					Step: step, Ts: ts, Type: "control", Kind: key,
					Label: key, Detail: fmt.Sprintf("%s = %g", key, val),
				})
			}
		}
		rows.Close()
	}

	// operator actions (ts only -> map to step after the rows are closed; see
	// the deadlock note on the alerts block above)
	if rows, err := d.Query(
		`SELECT COALESCE(ts,0), kind, COALESCE(result,'') FROM actions WHERE run_id=? ORDER BY ts`,
		runID); err == nil {
		start := len(tl.Events)
		for rows.Next() {
			var ts float64
			var kind, result string
			if rows.Scan(&ts, &kind, &result) == nil {
				tl.Events = append(tl.Events, TimelineEvent{
					Ts: ts, Type: "action", Kind: kind, Label: kind, Detail: result,
				})
			}
		}
		rows.Close()
		for i := start; i < len(tl.Events); i++ {
			tl.Events[i].Step = d.stepAtTs(runID, tl.Events[i].Ts)
		}
	}

	sort.SliceStable(tl.Events, func(i, j int) bool {
		if tl.Events[i].Step != tl.Events[j].Step {
			return tl.Events[i].Step < tl.Events[j].Step
		}
		return tl.Events[i].Type < tl.Events[j].Type
	})
	return tl, nil
}
