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
type Series struct {
	Step []int64               `json:"step"`
	Cols map[string][]*float64 `json:"cols"`
}

// Result is the full payload for one run.
type Result struct {
	Train    Series             `json:"train"`
	Eval     Series             `json:"eval"`
	Baseline map[string]float64 `json:"baseline,omitempty"`
	MaxStep  int64              `json:"max_step"`
	Decimated bool              `json:"decimated"`
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
	return fmt.Sprintf("json_extract(extra_json, '$.%s')", field), true
}

// Fetch returns the requested train/eval metrics for a run. since>0 returns only
// rows with step>since (incremental append). maxPoints>0 caps overview rows via
// stride decimation (ignored when since>0 — increments are already small).
func Fetch(d *db.DB, runID int64, trainFields, evalFields []string, since, to int64, maxPoints int) (Result, error) {
	res := Result{Train: Series{Cols: map[string][]*float64{}}, Eval: Series{Cols: map[string][]*float64{}}}

	tr, decT, err := fetchTable(d, "train_events", trainCols, runID, trainFields, since, to, maxPoints)
	if err != nil {
		return res, err
	}
	res.Train = tr
	ev, _, err := fetchTable(d, "eval_events", evalCols, runID, evalFields, since, to, 0) // eval is small; never decimate
	if err != nil {
		return res, err
	}
	res.Eval = ev
	res.Decimated = decT

	if len(tr.Step) > 0 {
		res.MaxStep = tr.Step[len(tr.Step)-1]
	}
	if len(ev.Step) > 0 && ev.Step[len(ev.Step)-1] > res.MaxStep {
		res.MaxStep = ev.Step[len(ev.Step)-1]
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
		   json_each(t.extra_json) je`, table)
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
	s := Series{Step: []int64{}, Cols: map[string][]*float64{}}
	if len(fields) == 0 {
		return s, false, nil
	}
	exprs := []string{"step"}
	var valid []string
	for _, f := range fields {
		expr, ok := colExpr(f, known)
		if !ok {
			continue
		}
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
	dest := make([]any, len(valid)+1)
	var step int64
	dest[0] = &step
	vals := make([]sql.NullFloat64, len(valid))
	for i := range vals {
		dest[i+1] = &vals[i]
	}
	for rows.Next() {
		if err := rows.Scan(dest...); err != nil {
			return s, false, err
		}
		s.Step = append(s.Step, step)
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

// decimate stride-samples rows to ~maxPoints, always keeping the last row so the
// live tip is exact. (Spike-preserving min/max bucketing is a future refinement;
// zoom/?from-to fetches return full resolution.)
func decimate(s Series, maxPoints int) Series {
	n := len(s.Step)
	stride := (n + maxPoints - 1) / maxPoints
	if stride < 2 {
		return s
	}
	out := Series{Cols: map[string][]*float64{}}
	for k := range s.Cols {
		out.Cols[k] = []*float64{}
	}
	for i := 0; i < n; i += stride {
		out.Step = append(out.Step, s.Step[i])
		for k, col := range s.Cols {
			out.Cols[k] = append(out.Cols[k], col[i])
		}
	}
	// ensure the final point is present
	if last := n - 1; (n-1)%stride != 0 {
		out.Step = append(out.Step, s.Step[last])
		for k, col := range s.Cols {
			out.Cols[k] = append(out.Cols[k], col[last])
		}
	}
	return out
}
