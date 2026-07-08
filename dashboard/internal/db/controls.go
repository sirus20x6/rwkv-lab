package db

import "database/sql"

// Control is one live hyperparameter override (desired value + apply ack).
type Control struct {
	Key         string   `json:"key"`
	Value       float64  `json:"value"`
	Generation  int64    `json:"generation"`
	RequestedTs float64  `json:"requested_ts"`
	AppliedStep *int64   `json:"applied_step"`
	AppliedTs   *float64 `json:"applied_ts"`
	Pending     bool     `json:"pending"` // applied_ts IS NULL → trainer hasn't picked it up yet
}

// SetControls atomically writes a set of overrides for a run. All keys in one
// call share a single new generation (so a multi-knob change is applied as a
// unit), and their apply-ack is cleared so they re-show as pending. This is the
// ACID multi-knob commit: the trainer's next poll sees all of them or none.
func (d *DB) SetControls(runName string, kv map[string]float64, ts float64) error {
	tx, err := d.DB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck — no-op after Commit

	var gen int64
	if err := tx.QueryRow(`SELECT COALESCE(max(generation),0) FROM run_controls WHERE run_name=?`, runName).
		Scan(&gen); err != nil {
		return err
	}
	gen++
	for k, v := range kv {
		if _, err := tx.Exec(
			`INSERT INTO run_controls(run_name,key,value,generation,requested_ts,applied_step,applied_ts)
			 VALUES(?,?,?,?,?,NULL,NULL)
			 ON CONFLICT(run_name,key) DO UPDATE SET value=excluded.value, generation=excluded.generation,
			   requested_ts=excluded.requested_ts, applied_step=NULL, applied_ts=NULL`,
			runName, k, v, gen, ts); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// GetControls returns all overrides for a run (for the live-tuning panel).
func (d *DB) GetControls(runName string) ([]Control, error) {
	rows, err := d.Query(
		`SELECT key, value, generation, COALESCE(requested_ts,0), applied_step, applied_ts
		 FROM run_controls WHERE run_name=? ORDER BY key`, runName)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Control
	for rows.Next() {
		var c Control
		var as sql.NullInt64
		var at sql.NullFloat64
		if err := rows.Scan(&c.Key, &c.Value, &c.Generation, &c.RequestedTs, &as, &at); err != nil {
			return nil, err
		}
		if as.Valid {
			c.AppliedStep = &as.Int64
		}
		if at.Valid {
			c.AppliedTs = &at.Float64
		}
		c.Pending = !at.Valid
		out = append(out, c)
	}
	return out, rows.Err()
}
