package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"trainboard/internal/series"
)

const seriesMaxPoints = 4000

// handleSeries returns columnar metric arrays for one run, for the Pixi charts.
//
//	GET /api/series/{run}?train=loss,lr,gnorm&eval=loss,ppl,top1&since=4000
//
// since>0 returns only newer rows (incremental append). Known columns are read
// directly; other field names resolve against extra_json.
func (s *Server) handleSeries(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("run")
	runID, ok, err := s.db.RunID(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !ok {
		http.Error(w, "no such run", http.StatusNotFound)
		return
	}

	q := r.URL.Query()
	trainFields := splitCSV(q.Get("train"))
	evalFields := splitCSV(q.Get("eval"))
	if len(trainFields) == 0 && len(evalFields) == 0 {
		trainFields = []string{"loss"}
		evalFields = []string{"loss", "ppl", "top1"}
	}
	since := int64(-1)
	if v := q.Get("since"); v != "" {
		since, _ = strconv.ParseInt(v, 10, 64)
	}
	// from/to = a step window (zoom): refetch that span at full resolution.
	var to int64
	ranged := false
	if v := q.Get("from"); v != "" {
		if f, e := strconv.ParseInt(v, 10, 64); e == nil {
			since = f - 1
			ranged = true
		}
	}
	if v := q.Get("to"); v != "" {
		to, _ = strconv.ParseInt(v, 10, 64)
		ranged = true
	}

	maxPoints := seriesMaxPoints
	if since > 0 && !ranged {
		maxPoints = 0 // pure incremental append; never decimate
	}

	res, err := series.Fetch(s.db, runID, trainFields, evalFields, since, to, maxPoints)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if since <= 0 && !ranged {
		res.Baseline = s.loadBaseline()
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(res)
}

// handleMetrics returns the metric catalog for a run (known columns + extra_json
// keys), for the dynamic metric picker.
//
//	GET /api/metrics/{run}
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("run")
	runID, ok, err := s.db.RunID(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !ok {
		http.Error(w, "no such run", http.StatusNotFound)
		return
	}
	cat, err := series.Catalog(s.db, runID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(cat)
}

// handleTimeline returns checkpoint/alert/control/action markers for one run,
// for the Pixi timeline overlays and the event list.
//
//	GET /api/timeline/{run}
func (s *Server) handleTimeline(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("run")
	tl, err := s.db.GetTimeline(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(tl)
}

// loadBaseline reads the global original-model reference (runs/_baseline.json).
func (s *Server) loadBaseline() map[string]float64 {
	data, err := os.ReadFile(filepath.Join(s.cfg.RunsDir, "_baseline.json"))
	if err != nil {
		return nil
	}
	var raw map[string]any
	if json.Unmarshal(data, &raw) != nil {
		return nil
	}
	out := map[string]float64{}
	for _, k := range []string{"ppl", "loss", "top1_acc"} {
		if v, ok := raw[k].(float64); ok {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
