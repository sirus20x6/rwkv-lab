package server

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"time"

	"trainboard/internal/sysmon"
)

type runLivePayload struct {
	Run        string `json:"run"`
	Version    int64  `json:"version"`
	HeaderHTML string `json:"header_html"`
	KPIHTML    string `json:"kpi_html"`
}

// handleRunLive is a cheap, cache-free source of truth for the two headline
// regions users rely on while a run is active.  It intentionally queries the
// database instead of reading tickSnap: this endpoint is the recovery path if
// an SSE connection or shared refresh tick is delayed.
func (s *Server) handleRunLive(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	now := float64(time.Now().UnixNano()) / 1e9
	sum, ok, err := s.db.RunSummaryByName(name, now)
	if err != nil {
		http.Error(w, "live summary unavailable", http.StatusServiceUnavailable)
		return
	}
	if !ok {
		http.Error(w, "no such run", http.StatusNotFound)
		return
	}
	best := readBest(filepath.Join(s.cfg.RunsDir, name))
	applyEvalContractSummary(&sum, best)
	kpi, ok, err := s.db.RunKPIsByName(name)
	if err != nil {
		http.Error(w, "live metrics unavailable", http.StatusServiceUnavailable)
		return
	}
	if !ok {
		http.Error(w, "no such run", http.StatusNotFound)
		return
	}
	applyEvalContractKPIs(&kpi, best)

	var proc *sysmon.Proc
	if s.sampler != nil {
		if p, found := procIndex(s.sampler.Latest().Procs)[name]; found {
			proc = &p
		}
	}
	payload := runLivePayload{
		Run:        name,
		Version:    runVersion(sum),
		HeaderHTML: renderRunHeader(sum, proc, best, now),
		KPIHTML:    renderKPIs(kpi),
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(payload)
}
