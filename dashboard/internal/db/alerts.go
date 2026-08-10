package db

import (
	"database/sql"
	"fmt"
	"strings"
)

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

// ResolveAlerts removes recovered health conditions from the live banner while
// retaining their rows for the run timeline and audit history.
func (d *DB) ResolveAlerts(runName, kind string) error {
	_, err := d.Exec(`UPDATE alerts SET acknowledged=1
		WHERE acknowledged=0 AND run_name=? AND kind=?`, runName, kind)
	return err
}

// ResolveInactiveStalls retires live-condition banners for trainers that no
// longer have a process. Their alert rows remain as acknowledged history.
func (d *DB) ResolveInactiveStalls(activeRunNames []string) error {
	if len(activeRunNames) == 0 {
		_, err := d.Exec(`UPDATE alerts SET acknowledged=1
			WHERE acknowledged=0 AND kind='stall'`)
		return err
	}
	marks := strings.TrimSuffix(strings.Repeat("?,", len(activeRunNames)), ",")
	args := make([]any, len(activeRunNames))
	for i, name := range activeRunNames {
		args[i] = name
	}
	_, err := d.Exec(`UPDATE alerts SET acknowledged=1
		WHERE acknowledged=0 AND kind='stall' AND run_name NOT IN (`+marks+`)`, args...)
	return err
}

// engramRecallMinValidRate is the smallest engram_recall_valid_rate that makes
// a row's injection RMS evidence about the recall path.  The field is a
// FRACTION of positions that recalled anything (count / valid.numel() in
// src/rwkv_lab/vision_train.py), so a bare >0 test admits a batch where a
// single position out of thousands hit the memory — whose injection RMS is
// legitimately ~0 and would bias the window toward a false memory_path_dead.
const engramRecallMinValidRate = 0.01

// engramRowIsEvidence decides whether one train row's engram_inj_rms belongs in
// the dead-path window.
//
// Two producers emit engram_inj_rms and they do NOT share a key set:
//   - src/rwkv_lab/vision_train.py `_engram_metrics` emits engram_inj_rms plus
//     engram_recall_valid_rate (and friends) — but only while the bank has a
//     recall record; the recall keys vanish otherwise.
//   - src/rwkv_lab/grokking_metrics.py `injection_stats()` — used by
//     looped_rwkv_rosa_engram_v3.py and the convert_train recipe, i.e. the
//     Engram-LMB / GDN-conversion runs this alert exists for — emits
//     engram_inj_rms with NO recall-rate key at all.
//
// So an absent recall rate must NOT be read as "no lookup was possible"; it
// means the producer does not report one, and the row is the only evidence
// available. A present rate is still honoured as a filter.
func engramRowIsEvidence(engram, engramValid sql.NullFloat64) bool {
	if !engram.Valid {
		return false
	}
	if !engramValid.Valid {
		return true
	}
	return engramValid.Float64 >= engramRecallMinValidRate
}

// TrainStats summarizes the most recent train rows for divergence detection.
type TrainStats struct {
	N              int
	MaxGnorm       float64
	SkipFrac       float64
	LastTokPerSec  float64
	MedTokPerSec   float64
	LastStep       int64
	LastTS         float64
	LastLoss       float64   // newest row in the window
	OldestLoss     float64   // oldest row in the window (coarse train-trend check)
	CodecRel       *float64  // newest non-null codec_rel in the window
	CodecRelWindow []float64 // every codec_rel in the window (newest first)
	RosaInjRMS     *float64  // latest ROSA injection RMS (nil if the run has no ROSA)
	EngramInjRMS   *float64  // latest usable Engram injection RMS
	RosaInjWindow  []float64
	// EngramInjWindow holds every row whose Engram injection RMS is usable
	// evidence: rows reporting a recall rate below engramRecallMinValidRate are
	// dropped, rows reporting no recall rate at all are kept (see
	// engramRowIsEvidence).
	EngramInjWindow []float64
}

type RunTrainStats struct {
	RunID int64
	Stats TrainStats
}

// RecentTrainStatsByName fetches every live run's bounded train window in one query.
func (d *DB) RecentTrainStatsByName(names []string, n int) (map[string]RunTrainStats, error) {
	out := make(map[string]RunTrainStats, len(names))
	if len(names) == 0 {
		return out, nil
	}
	marks := strings.TrimSuffix(strings.Repeat("?,", len(names)), ",")
	args := make([]any, 0, len(names)+1)
	for _, name := range names {
		args = append(args, name)
	}
	args = append(args, n)
	// Keep the SQL straightforward rather than depending on generated column aliases.
	query := fmt.Sprintf(`WITH ranked AS (
		SELECT r.id, r.name, e.step, e.ts, e.gnorm, e.skipped, e.tok_per_sec, e.loss,
		       json_extract(e.extra_json,'$.codec_rel') AS codec,
		       json_extract(e.extra_json,'$.rosa_inj_rms') AS rosa,
		       json_extract(e.extra_json,'$.engram_inj_rms') AS engram,
		       json_extract(e.extra_json,'$.engram_recall_valid_rate') AS engram_valid,
		       ROW_NUMBER() OVER (PARTITION BY r.id ORDER BY e.step DESC) AS rn
		FROM runs r JOIN train_events e ON e.run_id=r.id WHERE r.name IN (%s))
		SELECT id,name,step,ts,gnorm,skipped,tok_per_sec,loss,codec,rosa,engram,engram_valid
		FROM ranked WHERE rn<=? ORDER BY name,step DESC`, marks)
	rows, err := d.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type accumulator struct {
		id    int64
		ts    TrainStats
		skips int
		tps   []float64
	}
	acc := map[string]*accumulator{}
	for rows.Next() {
		var id, step int64
		var name string
		var rowTS, gnorm, tokPerSec, loss, codec, rosa, engram, engramValid sql.NullFloat64
		var skipped sql.NullInt64
		if err := rows.Scan(&id, &name, &step, &rowTS, &gnorm, &skipped, &tokPerSec, &loss,
			&codec, &rosa, &engram, &engramValid); err != nil {
			return nil, err
		}
		a := acc[name]
		if a == nil {
			a = &accumulator{id: id}
			acc[name] = a
			a.ts.LastStep, a.ts.LastTS = step, nzf(rowTS)
			a.ts.LastTokPerSec, a.ts.LastLoss = nzf(tokPerSec), nzf(loss)
		}
		if codec.Valid {
			a.ts.CodecRelWindow = append(a.ts.CodecRelWindow, codec.Float64)
			if a.ts.CodecRel == nil {
				v := codec.Float64
				a.ts.CodecRel = &v
			}
		}
		if rosa.Valid {
			a.ts.RosaInjWindow = append(a.ts.RosaInjWindow, rosa.Float64)
			if a.ts.RosaInjRMS == nil {
				v := rosa.Float64
				a.ts.RosaInjRMS = &v
			}
		}
		if engramRowIsEvidence(engram, engramValid) {
			a.ts.EngramInjWindow = append(
				a.ts.EngramInjWindow, engram.Float64)
			if a.ts.EngramInjRMS == nil {
				v := engram.Float64
				a.ts.EngramInjRMS = &v
			}
		}
		if loss.Valid {
			a.ts.OldestLoss = loss.Float64
		}
		if gnorm.Valid && gnorm.Float64 > a.ts.MaxGnorm {
			a.ts.MaxGnorm = gnorm.Float64
		}
		if skipped.Valid && skipped.Int64 != 0 {
			a.skips++
		}
		if tokPerSec.Valid && tokPerSec.Float64 > 0 {
			a.tps = append(a.tps, tokPerSec.Float64)
		}
		a.ts.N++
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for name, a := range acc {
		if a.ts.N > 0 {
			a.ts.SkipFrac = float64(a.skips) / float64(a.ts.N)
		}
		if len(a.tps) > 0 {
			sortFloats(a.tps)
			a.ts.MedTokPerSec = a.tps[len(a.tps)/2]
		}
		out[name] = RunTrainStats{RunID: a.id, Stats: a.ts}
	}
	return out, nil
}

// RecentTrainStats aggregates the last n train rows for a run.
func (d *DB) RecentTrainStats(runID int64, n int) (TrainStats, error) {
	return d.recentTrainStatsSince(runID, -1, -1, n)
}

// RecentTrainStatsAfter scopes a health window to optimizer steps committed
// after an eval-contract reset, excluding pre-mutation rows retained as history.
func (d *DB) RecentTrainStatsAfter(runID, afterStep int64, n int) (TrainStats, error) {
	return d.recentTrainStatsSince(runID, afterStep, -1, n)
}

// RecentTrainStatsSince scopes a health window to records produced after both
// a logical optimizer boundary and a durable publication time.  The timestamp
// predicate matters while the watcher is catching up: an abandoned branch can
// still have higher-step rows in SQLite until its rewritten log is observed.
func (d *DB) RecentTrainStatsSince(runID, afterStep int64, afterTS float64, n int) (TrainStats, error) {
	return d.recentTrainStatsSince(runID, afterStep, afterTS, n)
}

func (d *DB) recentTrainStatsSince(runID, afterStep int64, afterTS float64, n int) (TrainStats, error) {
	rows, err := d.Query(
		`SELECT step, ts, gnorm, skipped, tok_per_sec, loss,
		        json_extract(extra_json,'$.codec_rel'),
		        json_extract(extra_json,'$.rosa_inj_rms'),
		        json_extract(extra_json,'$.engram_inj_rms'),
		        json_extract(extra_json,'$.engram_recall_valid_rate')
		 FROM train_events WHERE run_id=? AND step>? AND ts>? ORDER BY step DESC LIMIT ?`,
		runID, afterStep, afterTS, n)
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
		var rowTS, gnorm, tokPerSec, loss sql.NullFloat64
		var skipped sql.NullInt64
		var codec, rosa, engram, engramValid sql.NullFloat64
		if err := rows.Scan(
			&step, &rowTS, &gnorm, &skipped, &tokPerSec, &loss,
			&codec, &rosa, &engram, &engramValid); err != nil {
			return ts, err
		}
		if first {
			ts.LastStep, ts.LastTS = step, nzf(rowTS)
			ts.LastTokPerSec = nzf(tokPerSec)
			ts.LastLoss = nzf(loss)
			first = false
		}
		if codec.Valid {
			ts.CodecRelWindow = append(ts.CodecRelWindow, codec.Float64)
			if ts.CodecRel == nil {
				v := codec.Float64
				ts.CodecRel = &v
			}
		}
		if rosa.Valid {
			ts.RosaInjWindow = append(ts.RosaInjWindow, rosa.Float64)
			if ts.RosaInjRMS == nil {
				v := rosa.Float64
				ts.RosaInjRMS = &v
			}
		}
		if engramRowIsEvidence(engram, engramValid) {
			ts.EngramInjWindow = append(
				ts.EngramInjWindow, engram.Float64)
			if ts.EngramInjRMS == nil {
				v := engram.Float64
				ts.EngramInjRMS = &v
			}
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
	LastStep     int64
	LastTS       float64
	LastPPL      float64
	MinPPL       float64
	LastBlockVal *float64
	MinBlockVal  *float64
	LastMaxRW    *float64 // latest loop_max_rw (LoopedRWKV gate magnitude; nil = no loop)
	LastLoopMult *float64 // trainer-reported EFFECTIVE loop_lr_mult (folds in the launch arg; nil = old run)
	LastPinThr   *float64 // trainer-reported pin threshold (scales with --loop-gate-cap; nil = legacy 0.245)
	LastLoopLive *float64 // 1 = live loop_lr_mult steering applies; 0 = baked in (schedulefree)
	LastLoopAnn  *float64 // 1 = trainer-side anneal owns boost cooling (--loop-anneal-rw); nil/0 = detector-managed
}

// RecentEvalStats aggregates the last n eval rows for a run (ppl + held-out
// block_val and the loop-gate state from extra_json).
func (d *DB) RecentEvalStats(runID int64, n int) (EvalStats, error) {
	return d.recentEvalStatsSince(runID, -1, -1, n)
}

// RecentEvalStatsAfter excludes retained metrics from an abandoned contract.
func (d *DB) RecentEvalStatsAfter(runID, afterStep int64, n int) (EvalStats, error) {
	return d.recentEvalStatsSince(runID, afterStep, -1, n)
}

// RecentEvalStatsSince excludes both earlier steps and retained future-branch
// rows whose ingest timestamp predates an eval-contract reset.
func (d *DB) RecentEvalStatsSince(runID, afterStep int64, afterTS float64, n int) (EvalStats, error) {
	return d.recentEvalStatsSince(runID, afterStep, afterTS, n)
}

func (d *DB) recentEvalStatsSince(runID, afterStep int64, afterTS float64, n int) (EvalStats, error) {
	rows, err := d.Query(
		`SELECT step, ts, ppl, json_extract(extra_json,'$.block_val'), json_extract(extra_json,'$.loop_max_rw'),
		        json_extract(extra_json,'$.loop_lr_mult'), json_extract(extra_json,'$.loop_pin_thr'),
		        json_extract(extra_json,'$.loop_live'), json_extract(extra_json,'$.loop_anneal')
		 FROM eval_events WHERE run_id=? AND step>? AND ts>? ORDER BY step DESC LIMIT ?`,
		runID, afterStep, afterTS, n)
	if err != nil {
		return EvalStats{}, err
	}
	defer rows.Close()
	var es EvalStats
	first := true
	for rows.Next() {
		var step int64
		var rowTS, ppl, block, maxRW, loopMult, pinThr, loopLive, loopAnn sql.NullFloat64
		if err := rows.Scan(&step, &rowTS, &ppl, &block, &maxRW, &loopMult, &pinThr, &loopLive, &loopAnn); err != nil {
			return es, err
		}
		if first {
			es.LastStep, es.LastTS = step, nzf(rowTS)
			es.LastPPL = nzf(ppl)
			if block.Valid {
				v := block.Float64
				es.LastBlockVal = &v
			}
			if maxRW.Valid {
				v := maxRW.Float64
				es.LastMaxRW = &v
			}
			if loopMult.Valid {
				v := loopMult.Float64
				es.LastLoopMult = &v
			}
			if pinThr.Valid {
				v := pinThr.Float64
				es.LastPinThr = &v
			}
			if loopLive.Valid {
				v := loopLive.Float64
				es.LastLoopLive = &v
			}
			if loopAnn.Valid {
				v := loopAnn.Float64
				es.LastLoopAnn = &v
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
