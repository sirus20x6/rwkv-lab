package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"time"
)

const trainVMDraftLimit = 2 << 20

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

func (s *Server) handleTrainVMSchema(w http.ResponseWriter, _ *http.Request) {
	s.handleTrainVMDocument(w, "schema")
}

func (s *Server) handleTrainVMExample(w http.ResponseWriter, _ *http.Request) {
	s.handleTrainVMDocument(w, "example")
}

func (s *Server) handleTrainVMDocument(w http.ResponseWriter, kind string) {
	if s.authoring == nil {
		http.Error(w, "TrainVM authoring is not configured", http.StatusServiceUnavailable)
		return
	}
	var document json.RawMessage
	var err error
	if kind == "schema" {
		document, err = s.authoring.Schema()
	} else {
		document, err = s.authoring.Example()
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(document)
}

func (s *Server) handleTrainVMCompile(w http.ResponseWriter, r *http.Request) {
	if s.authoring == nil {
		http.Error(w, "TrainVM authoring is not configured", http.StatusServiceUnavailable)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, trainVMDraftLimit)
	document, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "experiment draft exceeds the 2 MiB limit", http.StatusRequestEntityTooLarge)
		return
	}
	if !json.Valid(document) {
		http.Error(w, "experiment draft is not valid JSON", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	result, err := s.authoring.Compile(ctx, document)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(result)
}
