package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	trainvmstore "trainboard/internal/trainvm"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const trainVMDraftLimit = 2 << 20
const trainVMCommandLimit = 64 << 10

func (s *Server) handleTrainVMRuns(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	commandsEnabled := s.trainVMCommandsReachable(r.Context())
	if s.trainvm == nil {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"enabled": false, "commands_enabled": commandsEnabled, "runs": []any{},
		})
		return
	}
	runs, err := s.trainvm.Runs(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"enabled": true, "commands_enabled": commandsEnabled, "runs": runs,
	})
}

func (s *Server) trainVMCommandsReachable(parent context.Context) bool {
	if s.commander == nil {
		return false
	}
	probe, ok := s.commander.(interface{ Reachable(context.Context) bool })
	if !ok {
		return true
	}
	ctx, cancel := context.WithTimeout(parent, 150*time.Millisecond)
	defer cancel()
	return probe.Reachable(ctx)
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

type trainVMControlRequest struct {
	ExpectedRunRevision     uint64         `json:"expected_run_revision"`
	ExpectedControlRevision uint64         `json:"expected_control_revision"`
	IdempotencyKey          string         `json:"idempotency_key"`
	Reason                  string         `json:"reason"`
	Assignments             map[string]any `json:"assignments"`
}

func (s *Server) handleTrainVMControls(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil || s.trainvm == nil {
		http.Error(w, "TrainVM read model and command authority are both required", http.StatusServiceUnavailable)
		return
	}
	if !trainVMControlHostAllowed(r.Host) {
		http.Error(w, "TrainVM mutations are restricted to a loopback host", http.StatusForbidden)
		return
	}
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		http.Error(w, "control requests require application/json", http.StatusUnsupportedMediaType)
		return
	}
	if origin := r.Header.Get("Origin"); origin != "" {
		parsed, parseErr := url.Parse(origin)
		expectedScheme := "http"
		if r.TLS != nil {
			expectedScheme = "https"
		}
		if parseErr != nil || parsed.Host != r.Host || parsed.Scheme != expectedScheme {
			http.Error(w, "cross-origin control request rejected", http.StatusForbidden)
			return
		}
	}
	if site := r.Header.Get("Sec-Fetch-Site"); site != "" && site != "same-origin" && site != "none" {
		http.Error(w, "cross-site control request rejected", http.StatusForbidden)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, trainVMCommandLimit)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	decoder.UseNumber()
	var input trainVMControlRequest
	if err := decoder.Decode(&input); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			http.Error(w, "control request exceeds the 64 KiB limit", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "invalid control request: "+err.Error(), http.StatusBadRequest)
		return
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		http.Error(w, "control request must contain one JSON object", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	run, found, err := s.trainvm.Run(ctx, r.PathValue("run"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run", http.StatusNotFound)
		return
	}
	journalID, err := s.trainvm.JournalID(ctx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	result, err := s.commander.RequestControls(ctx, trainvmstore.ControlRequest{
		RunID: r.PathValue("run"), ExpectedJournalID: journalID, ExpectedPlanHash: run.PlanHash,
		ExpectedRunRevision:     input.ExpectedRunRevision,
		ExpectedControlRevision: input.ExpectedControlRevision,
		IdempotencyKey:          input.IdempotencyKey, Author: "dashboard", Reason: input.Reason,
		Assignments: input.Assignments,
	})
	if err != nil {
		var validationError *trainvmstore.ValidationError
		if errors.As(err, &validationError) {
			http.Error(w, validationError.Error(), http.StatusBadRequest)
			return
		}
		code := status.Code(err)
		httpStatus := http.StatusBadGateway
		switch code {
		case codes.Unavailable:
			httpStatus = http.StatusServiceUnavailable
		case codes.DeadlineExceeded, codes.Canceled:
			httpStatus = http.StatusGatewayTimeout
		case codes.InvalidArgument:
			httpStatus = http.StatusBadRequest
		case codes.NotFound:
			httpStatus = http.StatusNotFound
		case codes.ResourceExhausted:
			httpStatus = http.StatusRequestEntityTooLarge
		}
		http.Error(w, err.Error(), httpStatus)
		return
	}
	httpStatus := http.StatusOK
	switch result.Disposition {
	case "ACCEPTED":
		httpStatus = http.StatusAccepted
	case "ALREADY_APPLIED":
		httpStatus = http.StatusOK
	case "CONFLICT":
		httpStatus = http.StatusConflict
	case "REJECTED":
		httpStatus = http.StatusUnprocessableEntity
	default:
		http.Error(w, "native authority returned an unknown disposition", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(httpStatus)
	_ = json.NewEncoder(w).Encode(result)
}

func trainVMControlHostAllowed(hostPort string) bool {
	host := hostPort
	if parsed, _, err := net.SplitHostPort(hostPort); err == nil {
		host = parsed
	}
	host = strings.Trim(strings.TrimSpace(host), "[]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	address := net.ParseIP(host)
	return address != nil && address.IsLoopback()
}

func (s *Server) handleTrainVMControlView(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM journal is not configured", http.StatusServiceUnavailable)
		return
	}
	view, found, err := s.trainvm.Controls(r.Context(), r.PathValue("run"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run or persisted plan", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(view)
}
