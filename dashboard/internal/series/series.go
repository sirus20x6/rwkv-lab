// Package series builds columnar metric arrays for the Pixi charts. Known
// columns are read directly; anything else is pulled from extra_json via
// SQLite's json_extract. Large overviews are stride-decimated; incremental
// (?since=) and ranged (?from/&to=) fetches return full resolution.
package series

import (
	"database/sql"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"trainboard/internal/db"
)

// Series is a shared step axis plus one nullable column per requested metric.
// Ts carries each row's wall-clock timestamp (0 when the row has none) so the
// client can offer a time-based x-axis.
type Series struct {
	Step []int64               `json:"step"`
	Ts   []float64             `json:"ts"`
	Cols map[string][]*float64 `json:"cols"`
}

// Result is the full payload for one run.
type Result struct {
	Train        Series             `json:"train"`
	Eval         Series             `json:"eval"`
	Baseline     map[string]float64 `json:"baseline,omitempty"`
	MaxStep      int64              `json:"max_step"`
	MaxTrainStep int64              `json:"max_train_step"`
	MaxEvalStep  int64              `json:"max_eval_step"`
	Generation   int64              `json:"generation"`
	Decimated    bool               `json:"decimated"`
	SuppressBest bool               `json:"suppress_best"`
}

// Known typed columns per table; anything else → json_extract(extra_json,...).
var trainCols = map[string]bool{"loss": true, "lr": true, "gnorm": true, "tok_per_sec": true, "skipped": true}
var evalCols = map[string]bool{"loss": true, "ppl": true, "top1": true, "top5": true}

var fieldRe = regexp.MustCompile(`^[A-Za-z0-9_]+$`)

func colExpr(field string, known map[string]bool) (string, bool) {
	if !fieldRe.MatchString(field) {
		return "", false // reject anything that isn't a bare identifier (injection guard)
	}
	if known[field] {
		return field, true
	}
	// Extra payloads also contain metadata strings, objects, and arrays.  The
	// custom chart is numeric, so return NULL for a row whose key changes type
	// instead of asking database/sql to coerce JSON text into a float (which
	// aborts the entire series response).
	return fmt.Sprintf(
		"CASE WHEN json_type(extra_json, '$.%[1]s') IN ('integer','real','true','false') "+
			"THEN CAST(json_extract(extra_json, '$.%[1]s') AS REAL) END", field), true
}

// Fetch returns the requested train/eval metrics for a run. since>0 returns only
// rows with step>since (incremental append). maxPoints>0 caps overview rows via
// stride decimation (ignored when since>0 — increments are already small).
func Fetch(d *db.DB, runID int64, trainFields, evalFields []string, since, to int64, maxPoints int) (Result, error) {
	return FetchCursors(d, runID, trainFields, evalFields, since, since, to, maxPoints)
}

// FetchCursors advances dense training rows and sparse evaluation rows
// independently. A shared cursor can permanently skip an eval written at step
// N after the client has already fetched train step N.
func FetchCursors(d *db.DB, runID int64, trainFields, evalFields []string,
	trainSince, evalSince, to int64, maxPoints int) (Result, error) {
	res := Result{Train: Series{Cols: map[string][]*float64{}}, Eval: Series{Cols: map[string][]*float64{}}}

	tr, decT, err := fetchTable(d, "train_events", trainCols, runID, trainFields, trainSince, to, maxPoints)
	if err != nil {
		return res, err
	}
	res.Train = tr
	ev, _, err := fetchTable(d, "eval_events", evalCols, runID, evalFields, evalSince, to, 0) // eval is small; never decimate
	if err != nil {
		return res, err
	}
	res.Eval = ev
	res.Decimated = decT

	// These are authoritative table tips, not merely the last rows returned by
	// an incremental window. The browser compares them with its append cursors
	// to detect a trainer log rewind and discard abandoned future points.
	// Keep empty tables distinct from a legitimate step-0 row: returning zero
	// for both makes deletion of the only step-0 event invisible to an open tab.
	if err := d.QueryRow(`SELECT
		COALESCE((SELECT max(step) FROM train_events WHERE run_id=?),-1),
		COALESCE((SELECT max(step) FROM eval_events WHERE run_id=?),-1),
		COALESCE((SELECT event_generation FROM runs WHERE id=?),0)`,
		runID, runID, runID).Scan(
		&res.MaxTrainStep, &res.MaxEvalStep, &res.Generation); err != nil {
		return res, err
	}
	res.MaxStep = res.MaxTrainStep
	if res.MaxEvalStep > res.MaxStep {
		res.MaxStep = res.MaxEvalStep
	}
	if res.MaxStep < 0 {
		res.MaxStep = 0
	}
	return res, nil
}

// Catalog discovers the metric keys available for a run — the known typed
// columns plus any extra_json keys seen in recent rows. Powers the dynamic
// metric picker so charts aren't limited to the hardcoded field list.
func Catalog(d *db.DB, runID int64) (map[string][]string, error) {
	return map[string][]string{
		"train": tableCatalog(d, "train_events", trainCols, runID),
		"eval":  tableCatalog(d, "eval_events", evalCols, runID),
	}, nil
}

func tableCatalog(d *db.DB, table string, known map[string]bool, runID int64) []string {
	set := map[string]bool{}
	for k := range known {
		set[k] = true
	}
	// enumerate extra_json keys over a recent sample (consistent key set per run)
	q := fmt.Sprintf(
		`SELECT DISTINCT je.key FROM
		   (SELECT extra_json FROM %s WHERE run_id=? AND extra_json IS NOT NULL ORDER BY step DESC LIMIT 300) t,
		   json_each(t.extra_json) je
		 WHERE je.type IN ('integer','real','true','false')`, table)
	if rows, err := d.Query(q, runID); err == nil {
		defer rows.Close()
		for rows.Next() {
			var k string
			if rows.Scan(&k) == nil && fieldRe.MatchString(k) {
				set[k] = true
			}
		}
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func fetchTable(d *db.DB, table string, known map[string]bool, runID int64, fields []string, since, to int64, maxPoints int) (Series, bool, error) {
	s := Series{Step: []int64{}, Ts: []float64{}, Cols: map[string][]*float64{}}
	if len(fields) == 0 {
		return s, false, nil
	}
	exprs := []string{"step", "ts"}
	var valid []string
	seen := map[string]bool{}
	for _, f := range fields {
		if seen[f] {
			continue
		}
		expr, ok := colExpr(f, known)
		if !ok {
			continue
		}
		seen[f] = true
		exprs = append(exprs, expr)
		valid = append(valid, f)
	}
	if len(valid) == 0 {
		return s, false, nil
	}
	// to>0 bounds the upper step (ranged/zoom fetch); to<=0 means open-ended.
	query := fmt.Sprintf(`SELECT %s FROM %s WHERE run_id=? AND step>? AND (?<=0 OR step<=?) ORDER BY step`,
		strings.Join(exprs, ", "), table)
	rows, err := d.Query(query, runID, since, to, to)
	if err != nil {
		return s, false, err
	}
	defer rows.Close()

	for _, f := range valid {
		s.Cols[f] = []*float64{}
	}
	dest := make([]any, len(valid)+2)
	var step int64
	var ts sql.NullFloat64
	dest[0] = &step
	dest[1] = &ts
	vals := make([]sql.NullFloat64, len(valid))
	for i := range vals {
		dest[i+2] = &vals[i]
	}
	for rows.Next() {
		if err := rows.Scan(dest...); err != nil {
			return s, false, err
		}
		s.Step = append(s.Step, step)
		s.Ts = append(s.Ts, ts.Float64) // 0 when NULL
		for i, f := range valid {
			if vals[i].Valid {
				v := vals[i].Float64
				s.Cols[f] = append(s.Cols[f], &v)
			} else {
				s.Cols[f] = append(s.Cols[f], nil)
			}
		}
	}
	if err := rows.Err(); err != nil {
		return s, false, err
	}

	if maxPoints > 0 && len(s.Step) > maxPoints {
		return decimate(s, maxPoints), true, nil
	}
	return s, false, nil
}

// decimate reduces rows to roughly maxPoints while PRESERVING SPIKES: within
// each bucket it keeps, per requested column, the rows holding that column's
// min and max (plus the first/last rows overall). A one-step loss or gnorm
// spike therefore survives the overview instead of vanishing between strides —
// zoom/?from-to fetches still return full resolution.
func decimate(s Series, maxPoints int) Series {
	n := len(s.Step)
	// count columns that actually carry data so bucket sizing matches the ~2
	// kept rows each contributes per bucket
	active := 0
	for _, col := range s.Cols {
		for _, v := range col {
			if v != nil {
				active++
				break
			}
		}
	}
	if active == 0 {
		active = 1
	}
	buckets := maxPoints / (2 * active)
	if buckets < 64 {
		buckets = 64
	}
	if buckets >= n {
		return s
	}
	size := (n + buckets - 1) / buckets
	keep := map[int]bool{0: true, n - 1: true}
	for b0 := 0; b0 < n; b0 += size {
		b1 := b0 + size
		if b1 > n {
			b1 = n
		}
		for _, col := range s.Cols {
			minI, maxI := -1, -1
			for i := b0; i < b1; i++ {
				v := col[i]
				if v == nil {
					continue
				}
				if minI < 0 || *v < *col[minI] {
					minI = i
				}
				if maxI < 0 || *v > *col[maxI] {
					maxI = i
				}
			}
			if minI >= 0 {
				keep[minI] = true
				keep[maxI] = true
			}
		}
	}
	idx := make([]int, 0, len(keep))
	for i := range keep {
		idx = append(idx, i)
	}
	sort.Ints(idx)
	out := Series{Step: make([]int64, 0, len(idx)), Ts: make([]float64, 0, len(idx)), Cols: map[string][]*float64{}}
	for k := range s.Cols {
		out.Cols[k] = make([]*float64, 0, len(idx))
	}
	for _, i := range idx {
		out.Step = append(out.Step, s.Step[i])
		if i < len(s.Ts) {
			out.Ts = append(out.Ts, s.Ts[i])
		} else {
			out.Ts = append(out.Ts, 0)
		}
		for k, col := range s.Cols {
			out.Cols[k] = append(out.Cols[k], col[i])
		}
	}
	return out
}
