package db

import "database/sql"

// Alert is one health/divergence finding.
type Alert struct {
	ID       int64   `json:"id"`
	Ts       float64 `json:"ts"`
	RunName  string  `json:"run_name"`
	Kind     string  `json:"kind"`
	Severity string  `json:"severity"`
	Message  string  `json:"message"`
	Step     int64   `json:"step"`
}

// InsertAlert records an alert.
func (d *DB) InsertAlert(a Alert) (int64, error) {
	res, err := d.Exec(
		`INSERT INTO alerts(ts,run_name,kind,severity,message,step,acknowledged) VALUES(?,?,?,?,?,?,0)`,
		a.Ts, a.RunName, a.Kind, a.Severity, a.Message, a.Step)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// ActiveAlerts returns unacknowledged alerts, most recent first.
func (d *DB) ActiveAlerts(limit int) ([]Alert, error) {
	rows, err := d.Query(
		`SELECT id,ts,run_name,kind,severity,message,COALESCE(step,0)
		 FROM alerts WHERE acknowledged=0 ORDER BY ts DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Alert
	for rows.Next() {
		var a Alert
		if err := rows.Scan(&a.ID, &a.Ts, &a.RunName, &a.Kind, &a.Severity, &a.Message, &a.Step); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// AckAlert marks one alert acknowledged (id) or all (id<=0).
func (d *DB) AckAlert(id int64) error {
	if id <= 0 {
		_, err := d.Exec(`UPDATE alerts SET acknowledged=1 WHERE acknowledged=0`)
		return err
	}
	_, err := d.Exec(`UPDATE alerts SET acknowledged=1 WHERE id=?`, id)
	return err
}

// TrainStats summarizes the most recent train rows for divergence detection.
type TrainStats struct {
	N             int
	MaxGnorm      float64
	SkipFrac      float64
	LastTokPerSec float64
	MedTokPerSec  float64
	LastStep      int64
	LastLoss      float64  // newest row in the window
	OldestLoss    float64  // oldest row in the window (coarse train-trend check)
	CodecRel      *float64
	RosaInjRMS    *float64 // latest ROSA injection RMS (nil if the run has no ROSA)
	EngramInjRMS  *float64 // latest Engram injection RMS (nil if the run has no Engram)
}

// RecentTrainStats aggregates the last n train rows for a run.
func (d *DB) RecentTrainStats(runID int64, n int) (TrainStats, error) {
	rows, err := d.Query(
		`SELECT step, gnorm, skipped, tok_per_sec, loss,
		        json_extract(extra_json,'$.codec_rel'),
		        json_extract(extra_json,'$.rosa_inj_rms'),
		        json_extract(extra_json,'$.engram_inj_rms')
		 FROM train_events WHERE run_id=? ORDER BY step DESC LIMIT ?`, runID, n)
	if err != nil {
		return TrainStats{}, err
	}
	defer rows.Close()
	var ts TrainStats
	var skips int
	var tps []float64
	first := true
	for rows.Next() {
		var step int64
		var gnorm, tokPerSec, loss sql.NullFloat64
		var skipped sql.NullInt64
		var codec, rosa, engram sql.NullFloat64
		if err := rows.Scan(&step, &gnorm, &skipped, &tokPerSec, &loss, &codec, &rosa, &engram); err != nil {
			return ts, err
		}
		if first {
			ts.LastStep = step
			ts.LastTokPerSec = nzf(tokPerSec)
			ts.LastLoss = nzf(loss)
			if codec.Valid {
				v := codec.Float64
				ts.CodecRel = &v
			}
			if rosa.Valid {
				v := rosa.Float64
				ts.RosaInjRMS = &v
			}
			if engram.Valid {
				v := engram.Float64
				ts.EngramInjRMS = &v
			}
			first = false
		}
		if loss.Valid {
			ts.OldestLoss = loss.Float64 // overwritten each row; rows are DESC, so ends on oldest
		}
		if gnorm.Valid && gnorm.Float64 > ts.MaxGnorm {
			ts.MaxGnorm = gnorm.Float64
		}
		if skipped.Valid && skipped.Int64 != 0 {
			skips++
		}
		if tokPerSec.Valid && tokPerSec.Float64 > 0 {
			tps = append(tps, tokPerSec.Float64)
		}
		ts.N++
	}
	if ts.N > 0 {
		ts.SkipFrac = float64(skips) / float64(ts.N)
	}
	if len(tps) > 0 {
		// median throughput (robust baseline for a drop check)
		sortFloats(tps)
		ts.MedTokPerSec = tps[len(tps)/2]
	}
	return ts, rows.Err()
}

// EvalStats summarizes recent eval rows for the anti-grokking (held-out
// regression) check: a held-out metric rising above its own best.
type EvalStats struct {
	N            int
	LastPPL      float64
	MinPPL       float64
	LastBlockVal *float64
	MinBlockVal  *float64
}

// RecentEvalStats aggregates the last n eval rows for a run (ppl + held-out
// block_val from extra_json).
func (d *DB) RecentEvalStats(runID int64, n int) (EvalStats, error) {
	rows, err := d.Query(
		`SELECT ppl, json_extract(extra_json,'$.block_val')
		 FROM eval_events WHERE run_id=? ORDER BY step DESC LIMIT ?`, runID, n)
	if err != nil {
		return EvalStats{}, err
	}
	defer rows.Close()
	var es EvalStats
	first := true
	for rows.Next() {
		var ppl, block sql.NullFloat64
		if err := rows.Scan(&ppl, &block); err != nil {
			return es, err
		}
		if first {
			es.LastPPL = nzf(ppl)
			if block.Valid {
				v := block.Float64
				es.LastBlockVal = &v
			}
			first = false
		}
		if ppl.Valid && (es.MinPPL == 0 || ppl.Float64 < es.MinPPL) {
			es.MinPPL = ppl.Float64
		}
		if block.Valid && (es.MinBlockVal == nil || block.Float64 < *es.MinBlockVal) {
			v := block.Float64
			es.MinBlockVal = &v
		}
		es.N++
	}
	return es, rows.Err()
}

func nzf(n sql.NullFloat64) float64 {
	if n.Valid {
		return n.Float64
	}
	return 0
}

func sortFloats(a []float64) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j-1] > a[j]; j-- {
			a[j-1], a[j] = a[j], a[j-1]
		}
	}
}
