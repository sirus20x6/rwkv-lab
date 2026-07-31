package server

import (
	"encoding/json"
	"net/http"
	"strconv"
)

func (s *Server) handleTrainVMRuns(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if s.trainvm == nil {
		_ = json.NewEncoder(w).Encode(map[string]any{"enabled": false, "runs": []any{}})
		return
	}
	runs, err := s.trainvm.Runs(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"enabled": true, "runs": runs})
}

func (s *Server) handleTrainVMRun(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM journal is not configured", http.StatusServiceUnavailable)
		return
	}
	run, found, err := s.trainvm.Run(r.Context(), r.PathValue("run"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(run)
}

func (s *Server) handleTrainVMTimeline(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM journal is not configured", http.StatusServiceUnavailable)
		return
	}
	after, _ := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	events, err := s.trainvm.Timeline(r.Context(), r.PathValue("run"), after, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(events)
}
