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
	journalID, err := s.trainvm.JournalID(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"enabled": true, "commands_enabled": commandsEnabled,
		"journal_id": journalID, "runs": runs,
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

type trainVMSubmissionRequest struct {
	SourceDocument    string `json:"source_document"`
	SourceFormat      string `json:"source_format"`
	CreateRun         bool   `json:"create_run"`
	IdempotencyKey    string `json:"idempotency_key"`
	ExpectedJournalID string `json:"expected_journal_id"`
	ExpectedPlanHash  string `json:"expected_plan_hash"`
	Reason            string `json:"reason"`
}

func (s *Server) handleTrainVMSubmit(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil || s.trainvm == nil {
		http.Error(w, "TrainVM read model and command authority are both required", http.StatusServiceUnavailable)
		return
	}
	if !validateTrainVMMutation(w, r, "submission") {
		return
	}
	var input trainVMSubmissionRequest
	if !decodeTrainVMMutation(w, r, "submission", trainVMDraftLimit, &input) {
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	journalID, err := s.trainvm.JournalID(ctx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	input.ExpectedJournalID = strings.TrimSpace(input.ExpectedJournalID)
	input.ExpectedPlanHash = strings.TrimSpace(input.ExpectedPlanHash)
	if input.ExpectedJournalID == "" || (input.CreateRun && input.ExpectedPlanHash == "") {
		http.Error(w, "submission requires expected journal and plan identities", http.StatusBadRequest)
		return
	}
	if input.ExpectedJournalID != journalID {
		http.Error(w, "submission journal identity is stale", http.StatusConflict)
		return
	}
	result, err := s.commander.SubmitExperiment(ctx, trainvmstore.SubmissionRequest{
		SourceDocument: input.SourceDocument, SourceFormat: input.SourceFormat,
		CreateRun: input.CreateRun, IdempotencyKey: input.IdempotencyKey,
		ExpectedJournalID: input.ExpectedJournalID, ExpectedPlanHash: input.ExpectedPlanHash,
		Author: "dashboard", Reason: input.Reason,
	})
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	statusCode := http.StatusOK
	if input.CreateRun {
		if result.Run == nil {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			w.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(w).Encode(result)
			return
		}
		if result.PlanHash != input.ExpectedPlanHash || result.Run.PlanHash != input.ExpectedPlanHash ||
			strings.TrimSpace(result.Run.RunID) == "" {
			http.Error(w, "native authority returned a mismatched submission identity", http.StatusBadGateway)
			return
		}
		statusCode = http.StatusAccepted
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(result)
}

func (s *Server) handleTrainVMControls(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil || s.trainvm == nil {
		http.Error(w, "TrainVM read model and command authority are both required", http.StatusServiceUnavailable)
		return
	}
	if !validateTrainVMMutation(w, r, "control") {
		return
	}
	var input trainVMControlRequest
	if !decodeTrainVMMutation(w, r, "control", trainVMCommandLimit, &input) {
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
		writeTrainVMAuthorityError(w, err)
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

func validateTrainVMMutation(w http.ResponseWriter, r *http.Request, kind string) bool {
	if !trainVMControlHostAllowed(r.Host) {
		http.Error(w, "TrainVM mutations are restricted to a loopback host", http.StatusForbidden)
		return false
	}
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		http.Error(w, kind+" requests require application/json", http.StatusUnsupportedMediaType)
		return false
	}
	if origin := r.Header.Get("Origin"); origin != "" {
		parsed, parseErr := url.Parse(origin)
		expectedScheme := "http"
		if r.TLS != nil {
			expectedScheme = "https"
		}
		if parseErr != nil || parsed.Host != r.Host || parsed.Scheme != expectedScheme {
			http.Error(w, "cross-origin "+kind+" request rejected", http.StatusForbidden)
			return false
		}
	}
	if site := r.Header.Get("Sec-Fetch-Site"); site != "" && site != "same-origin" && site != "none" {
		http.Error(w, "cross-site "+kind+" request rejected", http.StatusForbidden)
		return false
	}
	return true
}

func decodeTrainVMMutation(w http.ResponseWriter, r *http.Request, kind string, limit int64, output any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	decoder.UseNumber()
	if err := decoder.Decode(output); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			http.Error(w, kind+" request exceeds its size limit", http.StatusRequestEntityTooLarge)
			return false
		}
		http.Error(w, "invalid "+kind+" request: "+err.Error(), http.StatusBadRequest)
		return false
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		http.Error(w, kind+" request must contain one JSON object", http.StatusBadRequest)
		return false
	}
	return true
}

func writeTrainVMAuthorityError(w http.ResponseWriter, err error) {
	var validationError *trainvmstore.ValidationError
	if errors.As(err, &validationError) {
		http.Error(w, validationError.Error(), http.StatusBadRequest)
		return
	}
	httpStatus := http.StatusBadGateway
	switch status.Code(err) {
	case codes.Unavailable:
		httpStatus = http.StatusServiceUnavailable
	case codes.DeadlineExceeded, codes.Canceled:
		httpStatus = http.StatusGatewayTimeout
	case codes.InvalidArgument:
		httpStatus = http.StatusBadRequest
	case codes.NotFound:
		httpStatus = http.StatusNotFound
	case codes.AlreadyExists, codes.FailedPrecondition, codes.Aborted:
		httpStatus = http.StatusConflict
	case codes.ResourceExhausted:
		httpStatus = http.StatusRequestEntityTooLarge
	}
	http.Error(w, err.Error(), httpStatus)
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
