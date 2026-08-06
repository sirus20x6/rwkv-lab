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
	"reflect"
	"strconv"
	"strings"
	"time"

	trainvmstore "trainboard/internal/trainvm"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const trainVMDraftLimit = 2 << 20
const trainVMCommandLimit = 64 << 10
const trainVMAuthorRunLimit = 5 << 20

func (s *Server) handleTrainVMHostAuthority(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	authority, err := s.commander.GetHostAuthorityStatus(r.Context())
	if err != nil {
		// A missing hostd binding is an operationally unavailable inspection
		// source, not a conflicting user mutation.
		if status.Code(err) == codes.FailedPrecondition {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		writeTrainVMAuthorityError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(authority)
}

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
		writeTrainVMAuthorityError(w, err)
		return
	}
	journalID, err := s.trainvm.JournalID(r.Context())
	if err != nil {
		writeTrainVMAuthorityError(w, err)
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
		writeTrainVMAuthorityError(w, err)
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

func (s *Server) handleTrainVMCompiledPlan(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM journal is not configured", http.StatusServiceUnavailable)
		return
	}
	view, found, err := s.trainvm.CompiledPlan(r.Context(), r.PathValue("run"))
	if err != nil {
		writeTrainVMAuthorityError(w, err)
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

func (s *Server) handleTrainVMTimeline(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM journal is not configured", http.StatusServiceUnavailable)
		return
	}
	after, _ := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	events, err := s.trainvm.Timeline(r.Context(), r.PathValue("run"), after, limit)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(events)
}

func (s *Server) handleTrainVMMetrics(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	after, _ := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	metrics, err := trainvmstore.Metrics(
		r.Context(), s.trainvm, r.PathValue("run"), after, limit)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(metrics)
}

func (s *Server) handleTrainVMArtifacts(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	after, _ := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	artifacts, err := trainvmstore.Artifacts(
		r.Context(), s.trainvm, r.PathValue("run"), after, limit)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	public := make([]trainvmstore.ObservableArtifact, 0, len(artifacts))
	for _, artifact := range artifacts {
		public = append(public, trainvmstore.RedactArtifact(artifact))
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(public)
}

func (s *Server) handleTrainVMObservability(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	after, parseErr := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
	if r.URL.Query().Get("after") == "" {
		after, parseErr = 0, nil
	}
	limit, limitErr := strconv.Atoi(r.URL.Query().Get("limit"))
	if r.URL.Query().Get("limit") == "" {
		limit, limitErr = 250, nil
	}
	if parseErr != nil || limitErr != nil || limit < 1 || limit > 1_000 {
		http.Error(w, "observability query requires a uint64 cursor and limit from 1 through 1000", http.StatusBadRequest)
		return
	}
	snapshot, found, err := trainvmstore.ProjectTelemetrySnapshot(
		r.Context(), s.trainvm, r.PathValue("run"), after, limit)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run or persisted plan", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(snapshot)
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
	if s.commander != nil && s.trainvm != nil {
		var merged map[string]any
		if json.Unmarshal(result, &merged) != nil {
			http.Error(w, "native compiler preview could not be merged", http.StatusBadGateway)
			return
		}
		valid, _ := merged["valid"].(bool)
		if !valid {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			_, _ = w.Write(result)
			return
		}
		journalID, journalErr := s.trainvm.JournalID(ctx)
		if journalErr != nil {
			http.Error(w, journalErr.Error(), http.StatusBadGateway)
			return
		}
		preview, previewErr := s.commander.SubmitExperiment(ctx, trainvmstore.SubmissionRequest{
			SourceDocument: string(document), SourceFormat: "json",
			ExpectedJournalID: journalID,
		})
		if previewErr != nil {
			http.Error(w, previewErr.Error(), http.StatusBadGateway)
			return
		}
		localPlanHash, hashOK := merged["plan_hash"].(string)
		localCanonicalPlan, canonicalOK := merged["canonical_plan"]
		var authorityCanonicalPlan any
		if !hashOK || localPlanHash == "" || !canonicalOK ||
			preview.PlanHash != localPlanHash || preview.CanonicalPlan == "" ||
			json.Unmarshal([]byte(preview.CanonicalPlan), &authorityCanonicalPlan) != nil ||
			!reflect.DeepEqual(localCanonicalPlan, authorityCanonicalPlan) {
			http.Error(w, "dashboard and authority compiler previews disagree", http.StatusBadGateway)
			return
		}
		if err := validateTrainingComponentPreview(
			authorityCanonicalPlan, preview.TrainingComponentLockDigest,
			preview.CanonicalTrainingComponentLock); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		merged["adapter_lock_digest"] = preview.AdapterLockDigest
		merged["canonical_adapter_lock"] = preview.CanonicalAdapterLock
		merged["training_component_lock_digest"] = preview.TrainingComponentLockDigest
		merged["canonical_training_component_lock"] = preview.CanonicalTrainingComponentLock
		if len(preview.Diagnostics) != 0 {
			existing, _ := merged["diagnostics"].([]any)
			for _, diagnostic := range preview.Diagnostics {
				existing = append(existing, map[string]any{
					"severity": diagnostic.Severity, "code": diagnostic.Code,
					"path": diagnostic.Path, "message": diagnostic.Message,
					"help": diagnostic.Help,
				})
			}
			merged["diagnostics"] = existing
			merged["valid"] = false
		}
		result, err = json.Marshal(merged)
		if err != nil {
			http.Error(w, "native compiler preview could not be encoded", http.StatusBadGateway)
			return
		}
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

type trainVMRunActionRequest struct {
	ExpectedRunRevision    uint64 `json:"expected_run_revision"`
	IdempotencyKey         string `json:"idempotency_key"`
	Reason                 string `json:"reason"`
	Action                 string `json:"action"`
	CheckpointFirst        bool   `json:"checkpoint_first"`
	ReleaseResources       bool   `json:"release_resources"`
	GracefulTimeoutSeconds uint64 `json:"graceful_timeout_seconds"`
}

type trainVMSubmissionRequest struct {
	SourceDocument                      string `json:"source_document"`
	SourceFormat                        string `json:"source_format"`
	CreateRun                           bool   `json:"create_run"`
	IdempotencyKey                      string `json:"idempotency_key"`
	ExpectedJournalID                   string `json:"expected_journal_id"`
	ExpectedPlanHash                    string `json:"expected_plan_hash"`
	ExpectedAdapterLockDigest           string `json:"expected_adapter_lock_digest"`
	ExpectedTrainingComponentLockDigest string `json:"expected_training_component_lock_digest"`
	Reason                              string `json:"reason"`
	ForkedFromRunID                     string `json:"forked_from_run_id"`
	ExpectedParentRunRevision           uint64 `json:"expected_parent_run_revision"`
	ExpectedParentPlanHash              string `json:"expected_parent_plan_hash"`
}

type trainVMAuthorRunRequest struct {
	RequestDocument  string `json:"request_document"`
	SourceFormat     string `json:"source_format"`
	DryRun           bool   `json:"dry_run"`
	ExpectedPlanHash string `json:"expected_plan_hash"`
}

func (s *Server) handleTrainVMAuthorRun(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	if !validateTrainVMMutation(w, r, "author run") {
		return
	}
	var input trainVMAuthorRunRequest
	if !decodeTrainVMMutation(w, r, "author run", trainVMAuthorRunLimit, &input) {
		return
	}
	input.SourceFormat = strings.ToLower(strings.TrimSpace(input.SourceFormat))
	input.ExpectedPlanHash = strings.TrimSpace(input.ExpectedPlanHash)
	if strings.TrimSpace(input.RequestDocument) == "" || len(input.RequestDocument) > trainVMDraftLimit ||
		input.SourceFormat != "json" && input.SourceFormat != "yaml" {
		http.Error(w, "author run requires a bounded JSON or YAML request document", http.StatusBadRequest)
		return
	}
	if input.DryRun && input.ExpectedPlanHash != "" ||
		!input.DryRun && !canonicalHTTPPlanHash(input.ExpectedPlanHash) {
		http.Error(w, "author run launch requires exactly one prior canonical plan hash", http.StatusBadRequest)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming responses are unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)
	encoder := json.NewEncoder(w)
	terminal := false
	writeUpdate := func(update trainvmstore.AuthorRunUpdate) error {
		if err := r.Context().Err(); err != nil {
			return err
		}
		if terminal {
			return errors.New("native authority emitted an AuthorRun update after terminal state")
		}
		if update.DryRun != input.DryRun {
			return errors.New("native authority changed the AuthorRun dry-run disposition")
		}
		if !input.DryRun && update.PlanHash != "" && update.PlanHash != input.ExpectedPlanHash {
			return errors.New("native authority changed the previewed AuthorRun plan identity")
		}
		terminalUpdate := update.Terminal
		terminalStage := update.Stage == "complete" || update.Stage == "failed"
		if terminalUpdate != terminalStage {
			return errors.New("native authority emitted an inconsistent terminal AuthorRun stage")
		}
		if terminalUpdate {
			if update.Stage == "complete" && (!canonicalHTTPPlanHash(update.PlanHash) ||
				input.DryRun && update.CanonicalPlanJSON == "" ||
				!input.DryRun && (update.PlanHash != input.ExpectedPlanHash || update.Run == nil ||
					update.Run.PlanHash != input.ExpectedPlanHash || update.Run.RunID == "")) {
				return errors.New("native authority completed AuthorRun without the fenced plan identity")
			}
			if update.Stage == "complete" {
				if err := validateAuthorRunCompletionEvidence(update); err != nil {
					return err
				}
			}
		}
		if err := encoder.Encode(update); err != nil {
			return err
		}
		flusher.Flush()
		terminal = terminalUpdate
		return nil
	}
	err := s.commander.AuthorRun(r.Context(), trainvmstore.AuthorRunRequest{
		RequestDocument: input.RequestDocument, SourceFormat: input.SourceFormat,
		DryRun: input.DryRun, ExpectedPlanHash: input.ExpectedPlanHash,
	}, writeUpdate)
	if err == nil && !terminal {
		err = errors.New("native authority AuthorRun stream ended without a terminal update")
	}
	if err != nil && !terminal && r.Context().Err() == nil {
		_ = writeUpdate(trainvmstore.AuthorRunUpdate{
			Stage: "failed", Detail: "dashboard authority stream failed", Terminal: true,
			DryRun: input.DryRun,
			Diagnostics: []trainvmstore.ControlDiagnostic{{
				Severity: "ERROR", Code: "dashboard.author_run_stream",
				Path: "/", Message: err.Error(),
				Help: "Inspect authority health and retry the exact compact request.",
			}},
		})
	}
}

func validateAuthorRunCompletionEvidence(update trainvmstore.AuthorRunUpdate) error {
	var receipt struct {
		Passed   bool   `json:"passed"`
		PlanHash string `json:"plan_hash"`
	}
	if update.PreflightReceiptJSON == "" ||
		json.Unmarshal([]byte(update.PreflightReceiptJSON), &receipt) != nil ||
		!receipt.Passed || !canonicalHTTPPlanHash(receipt.PlanHash) || receipt.PlanHash != update.PlanHash {
		return errors.New("native authority completed AuthorRun without matching passed preflight evidence")
	}
	if update.RecipeExpansionJSON != "" {
		var expansion struct {
			FinalPlanHash string `json:"final_plan_hash"`
		}
		if json.Unmarshal([]byte(update.RecipeExpansionJSON), &expansion) != nil ||
			!canonicalHTTPPlanHash(expansion.FinalPlanHash) || expansion.FinalPlanHash != update.PlanHash {
			return errors.New("native authority completed AuthorRun with mismatched recipe expansion evidence")
		}
	}
	return nil
}

func canonicalHTTPPlanHash(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if !(character >= '0' && character <= '9' || character >= 'a' && character <= 'f') {
			return false
		}
	}
	return true
}

type trainVMPlanDiffRequest struct {
	ExpectedRunRevision      uint64 `json:"expected_run_revision"`
	ProposedSourceDocument   string `json:"proposed_source_document"`
	SourceFormat             string `json:"source_format"`
	ExpectedJournalID        string `json:"expected_journal_id"`
	ExpectedCurrentPlanHash  string `json:"expected_current_plan_hash"`
	ExpectedProposedPlanHash string `json:"expected_proposed_plan_hash"`
}

func (s *Server) handleTrainVMPlanDiff(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil || s.trainvm == nil {
		http.Error(w, "TrainVM read model and command authority are both required", http.StatusServiceUnavailable)
		return
	}
	if !validateTrainVMMutation(w, r, "plan diff") {
		return
	}
	var input trainVMPlanDiffRequest
	if !decodeTrainVMMutation(w, r, "plan diff", trainVMDraftLimit, &input) {
		return
	}
	runID := strings.TrimSpace(r.PathValue("run"))
	input.ExpectedJournalID = strings.TrimSpace(input.ExpectedJournalID)
	input.ExpectedCurrentPlanHash = strings.TrimSpace(input.ExpectedCurrentPlanHash)
	input.ExpectedProposedPlanHash = strings.TrimSpace(input.ExpectedProposedPlanHash)
	if runID == "" || len(runID) > 256 || input.ExpectedRunRevision == 0 || input.ProposedSourceDocument == "" ||
		input.ExpectedJournalID == "" || input.ExpectedCurrentPlanHash == "" ||
		input.ExpectedProposedPlanHash == "" {
		http.Error(w, "plan diff requires run, revision, source, journal, and plan identities", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	journalID, err := s.trainvm.JournalID(ctx)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if input.ExpectedJournalID != journalID {
		http.Error(w, "plan diff journal identity is stale", http.StatusConflict)
		return
	}
	run, found, err := s.trainvm.Run(ctx, runID)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run", http.StatusNotFound)
		return
	}
	if run.RunRevision != input.ExpectedRunRevision || run.PlanHash != input.ExpectedCurrentPlanHash {
		http.Error(w, "plan diff run identity is stale", http.StatusConflict)
		return
	}
	result, err := s.commander.DiffPlan(ctx, trainvmstore.PlanDiffRequest{
		RunID: runID, ExpectedRunRevision: input.ExpectedRunRevision,
		ProposedSourceDocument: input.ProposedSourceDocument, SourceFormat: input.SourceFormat,
		ExpectedJournalID:        input.ExpectedJournalID,
		ExpectedCurrentPlanHash:  input.ExpectedCurrentPlanHash,
		ExpectedProposedPlanHash: input.ExpectedProposedPlanHash,
	})
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if result.ProposedPlanHash == "" || len(result.SemanticDiff) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_ = json.NewEncoder(w).Encode(result)
		return
	}
	if result.ProposedPlanHash != input.ExpectedProposedPlanHash {
		http.Error(w, "native authority returned a mismatched proposed plan identity", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(result)
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
	input.ExpectedAdapterLockDigest = strings.TrimSpace(input.ExpectedAdapterLockDigest)
	input.ExpectedTrainingComponentLockDigest = strings.TrimSpace(input.ExpectedTrainingComponentLockDigest)
	input.ForkedFromRunID = strings.TrimSpace(input.ForkedFromRunID)
	input.ExpectedParentPlanHash = strings.TrimSpace(input.ExpectedParentPlanHash)
	if input.ExpectedJournalID == "" || (input.CreateRun && (input.ExpectedPlanHash == "" || input.ExpectedAdapterLockDigest == "")) {
		http.Error(w, "submission requires expected journal, plan, and adapter-lock identities", http.StatusBadRequest)
		return
	}
	if input.ExpectedJournalID != journalID {
		http.Error(w, "submission journal identity is stale", http.StatusConflict)
		return
	}
	hasForkIdentity := input.ForkedFromRunID != "" || input.ExpectedParentRunRevision != 0 || input.ExpectedParentPlanHash != ""
	if hasForkIdentity {
		if !input.CreateRun || input.ForkedFromRunID == "" || len(input.ForkedFromRunID) > 256 ||
			input.ExpectedParentRunRevision == 0 || input.ExpectedParentPlanHash == "" {
			http.Error(w, "run fork requires a parent run, revision, and plan hash", http.StatusBadRequest)
			return
		}
	}
	result, err := s.commander.SubmitExperiment(ctx, trainvmstore.SubmissionRequest{
		SourceDocument: input.SourceDocument, SourceFormat: input.SourceFormat,
		CreateRun: input.CreateRun, IdempotencyKey: input.IdempotencyKey,
		ExpectedJournalID: input.ExpectedJournalID, ExpectedPlanHash: input.ExpectedPlanHash,
		ExpectedAdapterLockDigest:           input.ExpectedAdapterLockDigest,
		ExpectedTrainingComponentLockDigest: input.ExpectedTrainingComponentLockDigest,
		Author:                              "dashboard", Reason: input.Reason,
		ForkedFromRunID:           input.ForkedFromRunID,
		ExpectedParentRunRevision: input.ExpectedParentRunRevision,
		ExpectedParentPlanHash:    input.ExpectedParentPlanHash,
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
			result.AdapterLockDigest != input.ExpectedAdapterLockDigest ||
			result.TrainingComponentLockDigest != input.ExpectedTrainingComponentLockDigest ||
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

func (s *Server) handleTrainVMRunAction(w http.ResponseWriter, r *http.Request) {
	if s.commander == nil || s.trainvm == nil {
		http.Error(w, "TrainVM read model and command authority are both required", http.StatusServiceUnavailable)
		return
	}
	if !validateTrainVMMutation(w, r, "lifecycle action") {
		return
	}
	var input trainVMRunActionRequest
	if !decodeTrainVMMutation(w, r, "lifecycle action", trainVMCommandLimit, &input) {
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	runID := r.PathValue("run")
	run, found, err := s.trainvm.Run(ctx, runID)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run", http.StatusNotFound)
		return
	}
	journalID, err := s.trainvm.JournalID(ctx)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	result, err := s.commander.RequestRunAction(ctx, trainvmstore.RunActionRequest{
		RunID: runID, ExpectedJournalID: journalID, ExpectedPlanHash: run.PlanHash,
		ExpectedRunRevision: input.ExpectedRunRevision, IdempotencyKey: input.IdempotencyKey,
		Author: "dashboard", Reason: input.Reason, Action: input.Action,
		CheckpointFirst: input.CheckpointFirst, ReleaseResources: input.ReleaseResources,
		GracefulTimeoutSeconds: input.GracefulTimeoutSeconds,
	})
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if result.Action != strings.ToLower(strings.TrimSpace(input.Action)) &&
		(result.Disposition == "ACCEPTED" || result.Disposition == "ALREADY_APPLIED") {
		http.Error(w, "native authority returned a mismatched lifecycle action", http.StatusBadGateway)
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
		writeTrainVMAuthorityError(w, err)
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
