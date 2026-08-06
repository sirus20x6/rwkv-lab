package trainvm

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	trainvmv1 "trainboard/gen/trainvm/v1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Commander is the dashboard's mutation boundary. Implementations send typed
// commands to the native TrainVM authority and never receive a journal path.
type Commander interface {
	SubmitExperiment(context.Context, SubmissionRequest) (SubmissionResult, error)
	AuthorRun(context.Context, AuthorRunRequest, func(AuthorRunUpdate) error) error
	DiffPlan(context.Context, PlanDiffRequest) (PlanDiffResult, error)
	RequestControls(context.Context, ControlRequest) (ControlResult, error)
	RequestRunAction(context.Context, RunActionRequest) (RunActionResult, error)
	GetDescriptor(context.Context, DescriptorRequest) (DescriptorResult, error)
	GetHostAuthorityStatus(context.Context) (HostAuthorityStatus, error)
}

type AuthorRunRequest struct {
	RequestDocument  string
	SourceFormat     string
	DryRun           bool
	ExpectedPlanHash string
}

type AuthorRunUpdate struct {
	Stage                string              `json:"stage"`
	Detail               string              `json:"detail,omitempty"`
	PlanHash             string              `json:"plan_hash,omitempty"`
	CanonicalPlanJSON    string              `json:"canonical_plan_json,omitempty"`
	PreflightReceiptJSON string              `json:"preflight_receipt_json,omitempty"`
	RecipeExpansionJSON  string              `json:"recipe_expansion_json,omitempty"`
	Diagnostics          []ControlDiagnostic `json:"diagnostics,omitempty"`
	Run                  *RunIdentity        `json:"run,omitempty"`
	DashboardURL         string              `json:"dashboard_url,omitempty"`
	Terminal             bool                `json:"terminal"`
	DryRun               bool                `json:"dry_run"`
	ContentLockReused    bool                `json:"content_lock_reused"`
}

type DescriptorRequest struct {
	Provider string
	Version  string
}

type DescriptorResult struct {
	SchemaJSON string `json:"schema_json"`
	SchemaHash string `json:"schema_hash"`
}

type HostdCoordinatorStatus struct {
	APIVersion                       string `json:"api_version"`
	Lifecycle                        string `json:"lifecycle"`
	HostID                           string `json:"host_id"`
	BootID                           string `json:"boot_id"`
	BrokerEpoch                      string `json:"broker_epoch"`
	InventoryDigest                  string `json:"inventory_digest"`
	LiveSessions                     uint64 `json:"live_sessions"`
	AdmissionSessions                uint64 `json:"admission_sessions"`
	StaleAdmissionSessions           uint64 `json:"stale_admission_sessions"`
	ReleaseOnlySessions              uint64 `json:"release_only_sessions"`
	AdmissionCountsAreCachedEvidence bool   `json:"admission_counts_are_cached_evidence"`
	StartupAuditReceiptDigest        string `json:"startup_audit_receipt_digest,omitempty"`
	StartupAuditPassed               bool   `json:"startup_audit_passed"`
	PoisonReason                     string `json:"poison_reason,omitempty"`
}

type HostResourceFenceStatus struct {
	Kind            string `json:"kind"`
	Vendor          string `json:"vendor,omitempty"`
	StableID        string `json:"stable_id"`
	ParentID        string `json:"parent_id,omitempty"`
	Generation      uint64 `json:"generation"`
	InventoryDigest string `json:"inventory_digest"`
	TopologyDigest  string `json:"topology_digest"`
}

type HostdProcessAuthorityStatus struct {
	AllocationID                    string  `json:"allocation_id"`
	JournalID                       string  `json:"journal_id"`
	RunID                           string  `json:"run_id"`
	LogicalLeaseID                  string  `json:"logical_lease_id"`
	LogicalFencingToken             uint64  `json:"logical_fencing_token"`
	LaunchID                        string  `json:"launch_id"`
	Phase                           string  `json:"phase"`
	CgroupPath                      string  `json:"cgroup_path"`
	HostPID                         *int64  `json:"host_pid,omitempty"`
	ProcessStarttimeTicks           *uint64 `json:"process_starttime_ticks,omitempty"`
	DevicePolicyIntended            bool    `json:"device_policy_intended"`
	DevicePolicyInstalled           bool    `json:"device_policy_installed"`
	DevicePolicyDigest              string  `json:"device_policy_digest,omitempty"`
	DevicePolicyInstallationDigest  string  `json:"device_policy_installation_digest,omitempty"`
	ProcessPolicyIntended           bool    `json:"process_policy_intended"`
	ProcessPolicyInstalled          bool    `json:"process_policy_installed"`
	ProcessPolicyDigest             string  `json:"process_policy_digest,omitempty"`
	ProcessPolicyInstallationDigest string  `json:"process_policy_installation_digest,omitempty"`
	CgroupEmpty                     *bool   `json:"cgroup_empty,omitempty"`
	AcceleratorContextsEmpty        *bool   `json:"accelerator_contexts_empty,omitempty"`
	ContextAuditDigest              string  `json:"context_audit_digest,omitempty"`
	TerminalReceiptDigest           string  `json:"terminal_receipt_digest,omitempty"`
}

type HostAuthorityStatus struct {
	APIVersion                        string                        `json:"api_version"`
	Coordinator                       HostdCoordinatorStatus        `json:"coordinator"`
	StartupPhase                      string                        `json:"startup_phase"`
	StartupRecoverySteps              uint64                        `json:"startup_recovery_steps"`
	RemainingUnclosedProcessRecords   uint64                        `json:"remaining_unclosed_process_records"`
	RemainingTerminalReleaseRecords   uint64                        `json:"remaining_terminal_release_records"`
	LedgerVerified                    bool                          `json:"ledger_verified"`
	LedgerVerificationReason          string                        `json:"ledger_verification_reason,omitempty"`
	LedgerSequence                    uint64                        `json:"ledger_sequence"`
	LedgerChainHash                   string                        `json:"ledger_chain_hash"`
	LedgerRecordCount                 uint64                        `json:"ledger_record_count"`
	OccupancyLedgerSequence           uint64                        `json:"occupancy_ledger_sequence"`
	OccupancyDigest                   string                        `json:"occupancy_digest"`
	ActiveFenceCount                  uint64                        `json:"active_fence_count"`
	ActiveFences                      []HostResourceFenceStatus     `json:"active_fences"`
	ActiveFencesTruncated             bool                          `json:"active_fences_truncated"`
	ActiveProcessCount                uint64                        `json:"active_process_count"`
	ActiveProcesses                   []HostdProcessAuthorityStatus `json:"active_processes"`
	ActiveProcessesTruncated          bool                          `json:"active_processes_truncated"`
	ProcessLaunchEnabled              bool                          `json:"process_launch_enabled"`
	MutationEnabled                   bool                          `json:"mutation_enabled"`
	MutationDisabledReason            string                        `json:"mutation_disabled_reason,omitempty"`
	LeaseRenewalTrackedCount          uint64                        `json:"lease_renewal_tracked_count"`
	LeaseRenewalPoisoned              bool                          `json:"lease_renewal_poisoned"`
	LeaseRenewalFailure               string                        `json:"lease_renewal_failure,omitempty"`
	ResourceInventoryObserved         bool                          `json:"resource_inventory_observed"`
	ResourceInventoryObservationAgeNS uint64                        `json:"resource_inventory_observation_age_ns"`
	CurrentInventoryDigest            string                        `json:"current_inventory_digest,omitempty"`
	CurrentInventoryReceiptDigest     string                        `json:"current_inventory_receipt_digest,omitempty"`
	DegradedResourceCount             uint64                        `json:"degraded_resource_count"`
	ResourceDegradationReason         string                        `json:"resource_degradation_reason,omitempty"`
}

type SubmissionRequest struct {
	SourceDocument                      string
	SourceFormat                        string
	CreateRun                           bool
	IdempotencyKey                      string
	ExpectedJournalID                   string
	ExpectedPlanHash                    string
	ExpectedAdapterLockDigest           string
	ExpectedTrainingComponentLockDigest string
	Author                              string
	Reason                              string
	ForkedFromRunID                     string
	ExpectedParentRunRevision           uint64
	ExpectedParentPlanHash              string
}

type SubmissionResult struct {
	CanonicalDocument              string              `json:"canonical_document,omitempty"`
	CanonicalPlan                  string              `json:"canonical_plan,omitempty"`
	PlanHash                       string              `json:"plan_hash,omitempty"`
	AdapterLockDigest              string              `json:"adapter_lock_digest,omitempty"`
	CanonicalAdapterLock           string              `json:"canonical_adapter_lock,omitempty"`
	TrainingComponentLockDigest    string              `json:"training_component_lock_digest,omitempty"`
	CanonicalTrainingComponentLock string              `json:"canonical_training_component_lock,omitempty"`
	Run                            *RunIdentity        `json:"run,omitempty"`
	Diagnostics                    []ControlDiagnostic `json:"diagnostics,omitempty"`
}

type RunIdentity struct {
	RunID    string `json:"run_id"`
	Revision uint64 `json:"revision"`
	PlanHash string `json:"plan_hash"`
}

type PlanDiffRequest struct {
	RunID                    string
	ExpectedRunRevision      uint64
	ProposedSourceDocument   string
	SourceFormat             string
	ExpectedJournalID        string
	ExpectedCurrentPlanHash  string
	ExpectedProposedPlanHash string
}

type PlanDiffResult struct {
	AdoptableInPlace bool                `json:"adoptable_in_place"`
	SemanticDiff     json.RawMessage     `json:"semantic_diff,omitempty"`
	ProposedPlanHash string              `json:"proposed_plan_hash,omitempty"`
	Diagnostics      []ControlDiagnostic `json:"diagnostics,omitempty"`
}

type ControlRequest struct {
	RunID                   string
	ExpectedJournalID       string
	ExpectedPlanHash        string
	ExpectedRunRevision     uint64
	ExpectedControlRevision uint64
	IdempotencyKey          string
	Author                  string
	Reason                  string
	Assignments             map[string]any
}

type ControlResult struct {
	Disposition     string              `json:"disposition"`
	CommandID       string              `json:"command_id,omitempty"`
	ControlRevision uint64              `json:"control_revision,omitempty"`
	ApplyPoint      string              `json:"apply_point,omitempty"`
	RequiresPause   bool                `json:"requires_pause"`
	Status          string              `json:"status,omitempty"`
	Assignments     map[string]any      `json:"assignments,omitempty"`
	Diagnostics     []ControlDiagnostic `json:"diagnostics,omitempty"`
}

// RunActionRequest carries the complete immutable operator intent for a
// lifecycle mutation. Action is one of checkpoint, pause, resume, or cancel.
type RunActionRequest struct {
	RunID                  string
	ExpectedJournalID      string
	ExpectedPlanHash       string
	ExpectedRunRevision    uint64
	IdempotencyKey         string
	Author                 string
	Reason                 string
	Action                 string
	CheckpointFirst        bool
	ReleaseResources       bool
	GracefulTimeoutSeconds uint64
}

type RunActionResult struct {
	Disposition            string              `json:"disposition"`
	CommandSequence        uint64              `json:"command_sequence,omitempty"`
	Action                 string              `json:"action,omitempty"`
	CommandID              string              `json:"command_id,omitempty"`
	ControllerSequence     uint64              `json:"controller_sequence,omitempty"`
	Status                 string              `json:"status,omitempty"`
	CheckpointFirst        bool                `json:"checkpoint_first,omitempty"`
	ReleaseResources       bool                `json:"release_resources,omitempty"`
	OptimizerStep          uint64              `json:"optimizer_step,omitempty"`
	ArtifactID             string              `json:"artifact_id,omitempty"`
	Reason                 string              `json:"reason,omitempty"`
	GracefulTimeoutSeconds uint64              `json:"graceful_timeout_seconds,omitempty"`
	Diagnostics            []ControlDiagnostic `json:"diagnostics,omitempty"`
}

type ControlDiagnostic struct {
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Path     string `json:"path"`
	Message  string `json:"message"`
	Help     string `json:"help,omitempty"`
}

// ValidationError identifies a request rejected locally before any RPC was
// attempted. HTTP callers can distinguish it from an ambiguous authority or
// transport failure and safely report a 400 response.
type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string { return e.Message }

func invalidControlRequest(format string, arguments ...any) error {
	return &ValidationError{Message: fmt.Sprintf(format, arguments...)}
}

// GRPCCommander owns a single reusable client connection to the independently
// supervised native authority. grpc.NewClient is lazy: the read-only dashboard
// can still start when the authority socket is temporarily unavailable.
type GRPCCommander struct {
	connection *grpc.ClientConn
	client     trainvmv1.TrainVMClient
}

var _ ReadModel = (*GRPCCommander)(nil)

func DialCommander(socketPath string) (*GRPCCommander, error) {
	if socketPath == "" {
		return nil, fmt.Errorf("TrainVM authority socket is not configured")
	}
	absolute, err := filepath.Abs(socketPath)
	if err != nil {
		return nil, fmt.Errorf("resolve TrainVM authority socket: %w", err)
	}
	connection, err := grpc.NewClient("unix://"+absolute,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("configure TrainVM authority client: %w", err)
	}
	return &GRPCCommander{
		connection: connection,
		client:     trainvmv1.NewTrainVMClient(connection),
	}, nil
}

func (c *GRPCCommander) Close() error {
	if c == nil || c.connection == nil {
		return nil
	}
	return c.connection.Close()
}

// Reachable actively drives the lazy channel toward READY until the caller's
// short deadline expires. It performs no mutation and lets the dashboard
// distinguish a configured authority from one that is currently serving.
func (c *GRPCCommander) Reachable(ctx context.Context) bool {
	if c == nil || c.connection == nil {
		return false
	}
	c.connection.Connect()
	for {
		state := c.connection.GetState()
		if state == connectivity.Ready {
			return true
		}
		if state == connectivity.Shutdown || !c.connection.WaitForStateChange(ctx, state) {
			return false
		}
	}
}

func (c *GRPCCommander) JournalID(ctx context.Context) (string, error) {
	if c == nil || c.client == nil {
		return "", fmt.Errorf("TrainVM authority client is not configured")
	}
	response, err := c.client.ListRuns(ctx, &trainvmv1.ListRunsRequest{Limit: 1})
	if err != nil {
		return "", err
	}
	identity := response.GetJournalId()
	if len(identity) != 32 {
		return "", fmt.Errorf("TrainVM authority returned a malformed journal identity")
	}
	return identity, nil
}

func (c *GRPCCommander) Runs(ctx context.Context) ([]Run, error) {
	if c == nil || c.client == nil {
		return nil, fmt.Errorf("TrainVM authority client is not configured")
	}
	const maximumRuns = 10_000
	result := make([]Run, 0)
	token := ""
	journalID := ""
	for {
		response, err := c.client.ListRuns(ctx, &trainvmv1.ListRunsRequest{
			Limit: 250, PageToken: token,
		})
		if err != nil {
			return nil, err
		}
		if len(response.GetJournalId()) != 32 ||
			(journalID != "" && response.GetJournalId() != journalID) {
			return nil, fmt.Errorf("TrainVM authority changed or malformed its journal identity during pagination")
		}
		journalID = response.GetJournalId()
		for _, summary := range response.GetRuns() {
			run, err := runFromProto(summary)
			if err != nil {
				return nil, err
			}
			result = append(result, run)
			if len(result) > maximumRuns {
				return nil, fmt.Errorf("TrainVM run listing exceeds the dashboard bound")
			}
		}
		next := response.GetNextPageToken()
		if next == "" {
			return result, nil
		}
		if next == token {
			return nil, fmt.Errorf("TrainVM authority repeated a pagination token")
		}
		token = next
	}
}

func (c *GRPCCommander) Run(ctx context.Context, runID string) (Run, bool, error) {
	if c == nil || c.client == nil {
		return Run{}, false, fmt.Errorf("TrainVM authority client is not configured")
	}
	runID = strings.TrimSpace(runID)
	if runID == "" || len(runID) > 256 {
		return Run{}, false, &ValidationError{Message: "bounded run ID is required"}
	}
	response, err := c.client.GetRun(ctx, &trainvmv1.GetRunRequest{RunId: runID})
	if status.Code(err) == codes.NotFound {
		return Run{}, false, nil
	}
	if err != nil {
		return Run{}, false, err
	}
	run, err := runFromProto(response)
	return run, err == nil, err
}

func (c *GRPCCommander) CompiledPlan(ctx context.Context, runID string) (CompiledPlanView, bool, error) {
	if c == nil || c.client == nil {
		return CompiledPlanView{}, false, fmt.Errorf("TrainVM authority client is not configured")
	}
	runID = strings.TrimSpace(runID)
	if runID == "" || len(runID) > 256 {
		return CompiledPlanView{}, false, &ValidationError{Message: "bounded run ID is required"}
	}
	response, err := c.client.GetCompiledPlan(ctx, &trainvmv1.GetCompiledPlanRequest{RunId: runID})
	if status.Code(err) == codes.NotFound {
		return CompiledPlanView{}, false, nil
	}
	if err != nil {
		return CompiledPlanView{}, false, err
	}
	view, err := compiledPlanFromProto(runID, response)
	return view, err == nil, err
}

func compiledPlanFromProto(runID string, response *trainvmv1.GetCompiledPlanResponse) (CompiledPlanView, error) {
	if response == nil {
		return CompiledPlanView{}, fmt.Errorf("TrainVM authority returned an empty compiled plan")
	}
	identity := response.GetRun()
	canonical := response.GetCanonicalPlanJson()
	if identity == nil || response.GetJournalId() == "" || len(response.GetJournalId()) != 32 ||
		identity.GetRunId() != runID || identity.GetRevision() == 0 || identity.GetPlanHash() == "" ||
		canonical == "" || len(canonical) > compiledPlanMaximumBytes || !json.Valid([]byte(canonical)) {
		return CompiledPlanView{}, fmt.Errorf("TrainVM authority returned a malformed compiled plan")
	}
	actualHash := fmt.Sprintf("%x", sha256.Sum256([]byte(canonical)))
	if actualHash != identity.GetPlanHash() {
		return CompiledPlanView{}, fmt.Errorf(
			"TrainVM authority returned a mismatched compiled-plan hash: expected %s, got %s",
			identity.GetPlanHash(), actualHash)
	}
	return CompiledPlanView{
		JournalID: response.GetJournalId(), RunID: identity.GetRunId(),
		RunRevision: identity.GetRevision(), PlanHash: identity.GetPlanHash(),
		CanonicalPlan: json.RawMessage(canonical),
	}, nil
}

func (c *GRPCCommander) Timeline(ctx context.Context, runID string, after uint64, limit int) ([]Event, error) {
	return c.Events(ctx, EventQuery{RunID: runID, After: after, Limit: limit})
}

func (c *GRPCCommander) Events(ctx context.Context, input EventQuery) ([]Event, error) {
	if c == nil || c.client == nil {
		return nil, fmt.Errorf("TrainVM authority client is not configured")
	}
	query, err := normalizeEventQuery(input)
	if err != nil {
		return nil, err
	}
	stream, err := c.client.WatchEvents(ctx, &trainvmv1.WatchEventsRequest{
		RunIds: []string{query.RunID}, AfterJournalSequence: query.After,
		EventTypes: query.EventTypes, ReplayLimit: uint32(query.Limit),
		ThroughJournalSequence: query.Through, NewestFirst: query.NewestFirst,
		NewestPerMetricSeries: query.NewestPerMetricSeries,
	})
	if err != nil {
		return nil, err
	}
	result := make([]Event, 0, query.Limit)
	for {
		envelope, recvErr := stream.Recv()
		if recvErr == io.EOF {
			return result, nil
		}
		if recvErr != nil {
			return nil, recvErr
		}
		event, convertErr := eventFromProto(envelope)
		if convertErr != nil {
			return nil, convertErr
		}
		result = append(result, event)
	}
}

func (c *GRPCCommander) Controls(ctx context.Context, runID string) (ControlView, bool, error) {
	if c == nil || c.client == nil {
		return ControlView{}, false, fmt.Errorf("TrainVM authority client is not configured")
	}
	runID = strings.TrimSpace(runID)
	if runID == "" || len(runID) > 256 {
		return ControlView{}, false, &ValidationError{Message: "bounded run ID is required"}
	}
	response, err := c.client.GetControlView(ctx, &trainvmv1.GetControlViewRequest{RunId: runID})
	if status.Code(err) == codes.NotFound {
		return ControlView{}, false, nil
	}
	if err != nil {
		return ControlView{}, false, err
	}
	view, err := controlViewFromProto(response)
	return view, err == nil, err
}

func (c *GRPCCommander) RequestControls(ctx context.Context, request ControlRequest) (ControlResult, error) {
	if c == nil || c.client == nil {
		return ControlResult{}, fmt.Errorf("TrainVM authority client is not configured")
	}
	rpcRequest, err := controlRPCRequest(request)
	if err != nil {
		return ControlResult{}, err
	}
	response, err := c.client.CommandRun(ctx, rpcRequest)
	if err != nil {
		return ControlResult{}, err
	}
	return controlResult(response), nil
}

func (c *GRPCCommander) RequestRunAction(ctx context.Context, request RunActionRequest) (RunActionResult, error) {
	if c == nil || c.client == nil {
		return RunActionResult{}, fmt.Errorf("TrainVM authority client is not configured")
	}
	rpcRequest, err := runActionRPCRequest(request)
	if err != nil {
		return RunActionResult{}, err
	}
	response, err := c.client.CommandRun(ctx, rpcRequest)
	if err != nil {
		return RunActionResult{}, err
	}
	return runActionResult(request.Action, response)
}

func (c *GRPCCommander) SubmitExperiment(ctx context.Context, request SubmissionRequest) (SubmissionResult, error) {
	if c == nil || c.client == nil {
		return SubmissionResult{}, fmt.Errorf("TrainVM authority client is not configured")
	}
	rpcRequest, err := submissionRPCRequest(request)
	if err != nil {
		return SubmissionResult{}, err
	}
	response, err := c.client.SubmitExperiment(ctx, rpcRequest)
	if err != nil {
		return SubmissionResult{}, err
	}
	result := SubmissionResult{
		CanonicalDocument: response.GetCanonicalDocument(), CanonicalPlan: response.GetCanonicalPlan(),
		PlanHash:                       response.GetPlanHash(),
		AdapterLockDigest:              response.GetAdapterLockDigest(),
		CanonicalAdapterLock:           response.GetCanonicalAdapterLock(),
		TrainingComponentLockDigest:    response.GetTrainingComponentLockDigest(),
		CanonicalTrainingComponentLock: response.GetCanonicalTrainingComponentLock(),
	}
	if run := response.GetRun(); run != nil {
		result.Run = &RunIdentity{RunID: run.GetRunId(), Revision: run.GetRevision(), PlanHash: run.GetPlanHash()}
	}
	for _, diagnostic := range response.GetDiagnostics() {
		result.Diagnostics = append(result.Diagnostics, ControlDiagnostic{
			Severity: strings.TrimPrefix(diagnostic.GetSeverity().String(), "SEVERITY_"),
			Code:     diagnostic.GetCode(), Path: diagnostic.GetDocumentPath(),
			Message: diagnostic.GetMessage(), Help: diagnostic.GetHelp(),
		})
	}
	return result, nil
}

func (c *GRPCCommander) AuthorRun(ctx context.Context, request AuthorRunRequest,
	consume func(AuthorRunUpdate) error) error {
	if c == nil || c.client == nil {
		return fmt.Errorf("TrainVM authority client is not configured")
	}
	request.SourceFormat = strings.ToLower(strings.TrimSpace(request.SourceFormat))
	request.ExpectedPlanHash = strings.TrimSpace(request.ExpectedPlanHash)
	if strings.TrimSpace(request.RequestDocument) == "" || len(request.RequestDocument) > 2<<20 ||
		request.SourceFormat != "json" && request.SourceFormat != "yaml" || consume == nil {
		return &ValidationError{Message: "author run requires a bounded JSON/YAML document and update consumer"}
	}
	if request.DryRun && request.ExpectedPlanHash != "" ||
		!request.DryRun && !canonicalPlanHash(request.ExpectedPlanHash) {
		return &ValidationError{Message: "author run launch requires exactly one prior canonical plan hash"}
	}
	stream, err := c.client.AuthorRun(ctx, &trainvmv1.AuthorRunRequest{
		RequestDocument: request.RequestDocument, SourceFormat: request.SourceFormat,
		DryRun: request.DryRun, ExpectedPlanHash: request.ExpectedPlanHash,
	})
	if err != nil {
		return err
	}
	for {
		wire, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		update, err := authorRunUpdateFromProto(wire)
		if err != nil {
			return err
		}
		if err := consume(update); err != nil {
			return err
		}
	}
}

func authorRunUpdateFromProto(wire *trainvmv1.AuthorRunUpdate) (AuthorRunUpdate, error) {
	if wire == nil {
		return AuthorRunUpdate{}, fmt.Errorf("native authority emitted an invalid AuthorRun stage")
	}
	stage := ""
	switch wire.GetStage() {
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_VALIDATING:
		stage = "validating"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_RESOLVING:
		stage = "resolving"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_LOCKING_INPUTS:
		stage = "locking_inputs"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_PREFLIGHT:
		stage = "preflight"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_PROVISIONING:
		stage = "provisioning"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_SUBMITTING:
		stage = "submitting"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_COMPLETE:
		stage = "complete"
	case trainvmv1.AuthorRunStage_AUTHOR_RUN_STAGE_FAILED:
		stage = "failed"
	default:
		return AuthorRunUpdate{}, fmt.Errorf("native authority emitted an invalid AuthorRun stage")
	}
	update := AuthorRunUpdate{
		Stage:  stage,
		Detail: wire.GetDetail(), PlanHash: wire.GetPlanHash(),
		CanonicalPlanJSON:    wire.GetCanonicalPlanJson(),
		PreflightReceiptJSON: wire.GetPreflightReceiptJson(),
		RecipeExpansionJSON:  wire.GetRecipeExpansionJson(),
		DashboardURL:         wire.GetDashboardUrl(), Terminal: wire.GetTerminal(),
		DryRun: wire.GetDryRun(), ContentLockReused: wire.GetContentLockReused(),
	}
	if update.Terminal != (stage == "complete" || stage == "failed") {
		return AuthorRunUpdate{}, fmt.Errorf("native authority emitted an inconsistent terminal AuthorRun stage")
	}
	if update.PlanHash != "" && !canonicalPlanHash(update.PlanHash) {
		return AuthorRunUpdate{}, fmt.Errorf("native authority emitted a malformed AuthorRun plan hash")
	}
	for name, document := range map[string]string{
		"canonical plan":    update.CanonicalPlanJSON,
		"preflight receipt": update.PreflightReceiptJSON,
		"recipe expansion":  update.RecipeExpansionJSON,
	} {
		if document != "" && !json.Valid([]byte(document)) {
			return AuthorRunUpdate{}, fmt.Errorf("native authority emitted malformed AuthorRun %s JSON", name)
		}
	}
	if run := wire.GetRun(); run != nil {
		update.Run = &RunIdentity{RunID: run.GetRunId(), Revision: run.GetRevision(), PlanHash: run.GetPlanHash()}
		if update.Run.RunID == "" || !canonicalPlanHash(update.Run.PlanHash) ||
			update.PlanHash != "" && update.Run.PlanHash != update.PlanHash {
			return AuthorRunUpdate{}, fmt.Errorf("native authority emitted an invalid AuthorRun identity")
		}
	}
	for _, diagnostic := range wire.GetDiagnostics() {
		if diagnostic == nil || diagnostic.GetSeverity() == trainvmv1.Diagnostic_SEVERITY_UNSPECIFIED {
			return AuthorRunUpdate{}, fmt.Errorf("native authority emitted an invalid AuthorRun diagnostic")
		}
		update.Diagnostics = append(update.Diagnostics, ControlDiagnostic{
			Severity: strings.TrimPrefix(diagnostic.GetSeverity().String(), "SEVERITY_"),
			Code:     diagnostic.GetCode(), Path: diagnostic.GetDocumentPath(),
			Message: diagnostic.GetMessage(), Help: diagnostic.GetHelp(),
		})
	}
	return update, nil
}

func canonicalPlanHash(value string) bool {
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

func (c *GRPCCommander) DiffPlan(ctx context.Context, request PlanDiffRequest) (PlanDiffResult, error) {
	if c == nil || c.client == nil {
		return PlanDiffResult{}, fmt.Errorf("TrainVM authority client is not configured")
	}
	rpcRequest, err := planDiffRPCRequest(request)
	if err != nil {
		return PlanDiffResult{}, err
	}
	response, err := c.client.DiffPlan(ctx, rpcRequest)
	if err != nil {
		return PlanDiffResult{}, err
	}
	result := PlanDiffResult{
		AdoptableInPlace: response.GetAdoptableInPlace(),
		ProposedPlanHash: strings.TrimSpace(response.GetProposedPlanHash()),
	}
	if encoded := response.GetSemanticDiff(); encoded != "" {
		if len(encoded) > 2<<20 || !json.Valid([]byte(encoded)) {
			return PlanDiffResult{}, fmt.Errorf("TrainVM authority returned an invalid semantic plan diff")
		}
		var patch []json.RawMessage
		if err := json.Unmarshal([]byte(encoded), &patch); err != nil {
			return PlanDiffResult{}, fmt.Errorf("TrainVM authority returned a non-array semantic plan diff: %w", err)
		}
		result.SemanticDiff = json.RawMessage(encoded)
	}
	for _, diagnostic := range response.GetDiagnostics() {
		result.Diagnostics = append(result.Diagnostics, ControlDiagnostic{
			Severity: strings.TrimPrefix(diagnostic.GetSeverity().String(), "SEVERITY_"),
			Code:     diagnostic.GetCode(), Path: diagnostic.GetDocumentPath(),
			Message: diagnostic.GetMessage(), Help: diagnostic.GetHelp(),
		})
	}
	return result, nil
}

func (c *GRPCCommander) GetDescriptor(ctx context.Context, request DescriptorRequest) (DescriptorResult, error) {
	if c == nil || c.client == nil {
		return DescriptorResult{}, fmt.Errorf("TrainVM authority client is not configured")
	}
	request.Provider = strings.TrimSpace(request.Provider)
	request.Version = strings.TrimSpace(request.Version)
	if request.Provider == "" || request.Version == "" || len(request.Provider) > 256 || len(request.Version) > 256 {
		return DescriptorResult{}, &ValidationError{Message: "bounded descriptor provider and version are required"}
	}
	response, err := c.client.GetDescriptor(ctx, &trainvmv1.DescriptorRequest{
		Adapter: request.Provider,
		Version: request.Version,
	})
	if err != nil {
		return DescriptorResult{}, err
	}
	return DescriptorResult{
		SchemaJSON: response.GetSchemaJson(),
		SchemaHash: response.GetSchemaHash(),
	}, nil
}

func (c *GRPCCommander) GetHostAuthorityStatus(ctx context.Context) (HostAuthorityStatus, error) {
	if c == nil || c.client == nil {
		return HostAuthorityStatus{}, fmt.Errorf("TrainVM authority client is not configured")
	}
	response, err := c.client.GetHostAuthorityStatus(ctx, &trainvmv1.GetHostAuthorityStatusRequest{})
	if err != nil {
		return HostAuthorityStatus{}, err
	}
	return hostAuthorityStatusFromProto(response)
}

func enumSuffix(value fmt.Stringer, prefix string) (string, error) {
	name := value.String()
	if !strings.HasPrefix(name, prefix) || strings.HasSuffix(name, "UNSPECIFIED") {
		return "", fmt.Errorf("TrainVM authority returned an unknown %s value", strings.ToLower(strings.TrimSuffix(prefix, "_")))
	}
	return strings.ToLower(strings.TrimPrefix(name, prefix)), nil
}

func hostAuthorityStatusFromProto(response *trainvmv1.GetHostAuthorityStatusResponse) (HostAuthorityStatus, error) {
	if response == nil || response.GetCoordinator() == nil ||
		response.GetApiVersion() != "trainvm.hostd-authority-status/v1" {
		return HostAuthorityStatus{}, fmt.Errorf("TrainVM authority returned a malformed host authority status")
	}
	coordinator := response.GetCoordinator()
	lifecycle, err := enumSuffix(coordinator.GetLifecycle(), "HOSTD_LIFECYCLE_")
	if err != nil {
		return HostAuthorityStatus{}, err
	}
	startupPhase, err := enumSuffix(response.GetStartupPhase(), "HOSTD_STARTUP_PHASE_")
	if err != nil {
		return HostAuthorityStatus{}, err
	}
	if coordinator.GetApiVersion() == "" || coordinator.GetHostId() == "" ||
		coordinator.GetBootId() == "" || coordinator.GetBrokerEpoch() == "" ||
		coordinator.GetInventoryDigest() == "" || response.GetLedgerChainHash() == "" ||
		response.GetOccupancyDigest() == "" ||
		uint64(len(response.GetActiveFences())) > response.GetActiveFenceCount() ||
		uint64(len(response.GetActiveProcesses())) > response.GetActiveProcessCount() ||
		response.GetActiveFencesTruncated() != (uint64(len(response.GetActiveFences())) < response.GetActiveFenceCount()) ||
		response.GetActiveProcessesTruncated() != (uint64(len(response.GetActiveProcesses())) < response.GetActiveProcessCount()) ||
		response.GetActiveProcessCount() != response.GetRemainingUnclosedProcessRecords()+response.GetRemainingTerminalReleaseRecords() ||
		response.GetLedgerVerified() == (response.GetLedgerVerificationReason() != "") ||
		response.GetMutationEnabled() == (response.GetMutationDisabledReason() != "") ||
		response.GetLeaseRenewalPoisoned() && response.GetLeaseRenewalFailure() == "" ||
		response.GetDegradedResourceCount() > response.GetActiveFenceCount() ||
		response.GetResourceInventoryObserved() !=
			(response.GetCurrentInventoryDigest() != "" && response.GetCurrentInventoryReceiptDigest() != "") ||
		(response.GetResourceDegradationReason() == "") !=
			(response.GetResourceInventoryObserved() && response.GetDegradedResourceCount() == 0) {
		return HostAuthorityStatus{}, fmt.Errorf("TrainVM authority returned contradictory host authority evidence")
	}
	result := HostAuthorityStatus{
		APIVersion: response.GetApiVersion(),
		Coordinator: HostdCoordinatorStatus{
			APIVersion: coordinator.GetApiVersion(), Lifecycle: lifecycle,
			HostID: coordinator.GetHostId(), BootID: coordinator.GetBootId(),
			BrokerEpoch: coordinator.GetBrokerEpoch(), InventoryDigest: coordinator.GetInventoryDigest(),
			LiveSessions: coordinator.GetLiveSessions(), AdmissionSessions: coordinator.GetAdmissionSessions(),
			StaleAdmissionSessions: coordinator.GetStaleAdmissionSessions(), ReleaseOnlySessions: coordinator.GetReleaseOnlySessions(),
			AdmissionCountsAreCachedEvidence: coordinator.GetAdmissionCountsAreCachedEvidence(),
			StartupAuditReceiptDigest:        coordinator.GetStartupAuditReceiptDigest(),
			StartupAuditPassed:               coordinator.GetStartupAuditPassed(), PoisonReason: coordinator.GetPoisonReason(),
		},
		StartupPhase: startupPhase, StartupRecoverySteps: response.GetStartupRecoverySteps(),
		RemainingUnclosedProcessRecords: response.GetRemainingUnclosedProcessRecords(),
		RemainingTerminalReleaseRecords: response.GetRemainingTerminalReleaseRecords(),
		LedgerVerified:                  response.GetLedgerVerified(), LedgerVerificationReason: response.GetLedgerVerificationReason(),
		LedgerSequence: response.GetLedgerSequence(), LedgerChainHash: response.GetLedgerChainHash(),
		LedgerRecordCount: response.GetLedgerRecordCount(), OccupancyLedgerSequence: response.GetOccupancyLedgerSequence(),
		OccupancyDigest: response.GetOccupancyDigest(), ActiveFenceCount: response.GetActiveFenceCount(),
		ActiveFences:          make([]HostResourceFenceStatus, 0, len(response.GetActiveFences())),
		ActiveFencesTruncated: response.GetActiveFencesTruncated(), ActiveProcessCount: response.GetActiveProcessCount(),
		ActiveProcesses:          make([]HostdProcessAuthorityStatus, 0, len(response.GetActiveProcesses())),
		ActiveProcessesTruncated: response.GetActiveProcessesTruncated(), ProcessLaunchEnabled: response.GetProcessLaunchEnabled(),
		MutationEnabled: response.GetMutationEnabled(), MutationDisabledReason: response.GetMutationDisabledReason(),
		LeaseRenewalTrackedCount: response.GetLeaseRenewalTrackedCount(), LeaseRenewalPoisoned: response.GetLeaseRenewalPoisoned(),
		LeaseRenewalFailure: response.GetLeaseRenewalFailure(), ResourceInventoryObserved: response.GetResourceInventoryObserved(),
		ResourceInventoryObservationAgeNS: response.GetResourceInventoryObservationAgeNs(),
		CurrentInventoryDigest:            response.GetCurrentInventoryDigest(), CurrentInventoryReceiptDigest: response.GetCurrentInventoryReceiptDigest(),
		DegradedResourceCount: response.GetDegradedResourceCount(), ResourceDegradationReason: response.GetResourceDegradationReason(),
	}
	for _, fence := range response.GetActiveFences() {
		if fence == nil || fence.GetStableId() == "" || fence.GetGeneration() == 0 {
			return HostAuthorityStatus{}, fmt.Errorf("TrainVM authority returned a malformed resource fence")
		}
		kind, err := enumSuffix(fence.GetKind(), "HOST_RESOURCE_KIND_")
		if err != nil {
			return HostAuthorityStatus{}, err
		}
		vendor := ""
		if fence.GetVendor() != trainvmv1.HostAcceleratorVendor_HOST_ACCELERATOR_VENDOR_UNSPECIFIED {
			vendor, err = enumSuffix(fence.GetVendor(), "HOST_ACCELERATOR_VENDOR_")
			if err != nil {
				return HostAuthorityStatus{}, err
			}
		}
		result.ActiveFences = append(result.ActiveFences, HostResourceFenceStatus{
			Kind: kind, Vendor: vendor, StableID: fence.GetStableId(), ParentID: fence.GetParentId(),
			Generation: fence.GetGeneration(), InventoryDigest: fence.GetInventoryDigest(), TopologyDigest: fence.GetTopologyDigest(),
		})
	}
	for _, process := range response.GetActiveProcesses() {
		if process == nil || process.GetAllocationId() == "" || process.GetRunId() == "" ||
			process.GetLaunchId() == "" || process.GetCgroupPath() == "" {
			return HostAuthorityStatus{}, fmt.Errorf("TrainVM authority returned a malformed process authority row")
		}
		phase, err := enumSuffix(process.GetPhase(), "HOSTD_PROCESS_PHASE_")
		if err != nil {
			return HostAuthorityStatus{}, err
		}
		row := HostdProcessAuthorityStatus{
			AllocationID: process.GetAllocationId(), JournalID: process.GetJournalId(), RunID: process.GetRunId(),
			LogicalLeaseID: process.GetLogicalLeaseId(), LogicalFencingToken: process.GetLogicalFencingToken(),
			LaunchID: process.GetLaunchId(), Phase: phase, CgroupPath: process.GetCgroupPath(),
			DevicePolicyIntended: process.GetDevicePolicyIntended(), DevicePolicyInstalled: process.GetDevicePolicyInstalled(),
			DevicePolicyDigest: process.GetDevicePolicyDigest(), DevicePolicyInstallationDigest: process.GetDevicePolicyInstallationDigest(),
			ProcessPolicyIntended: process.GetProcessPolicyIntended(), ProcessPolicyInstalled: process.GetProcessPolicyInstalled(),
			ProcessPolicyDigest: process.GetProcessPolicyDigest(), ProcessPolicyInstallationDigest: process.GetProcessPolicyInstallationDigest(),
			ContextAuditDigest: process.GetContextAuditDigest(), TerminalReceiptDigest: process.GetTerminalReceiptDigest(),
		}
		if process.HostPid != nil {
			value := process.GetHostPid()
			row.HostPID = &value
		}
		if process.ProcessStarttimeTicks != nil {
			value := process.GetProcessStarttimeTicks()
			row.ProcessStarttimeTicks = &value
		}
		if process.CgroupEmpty != nil {
			value := process.GetCgroupEmpty()
			row.CgroupEmpty = &value
		}
		if process.AcceleratorContextsEmpty != nil {
			value := process.GetAcceleratorContextsEmpty()
			row.AcceleratorContextsEmpty = &value
		}
		hasProcessIdentity := row.HostPID != nil && row.ProcessStarttimeTicks != nil && *row.HostPID > 0 && *row.ProcessStarttimeTicks > 0
		policyEvidenceValid := row.DevicePolicyIntended == (row.DevicePolicyDigest != "") &&
			row.DevicePolicyInstalled == (row.DevicePolicyInstallationDigest != "") &&
			row.ProcessPolicyIntended == (row.ProcessPolicyDigest != "") &&
			row.ProcessPolicyInstalled == (row.ProcessPolicyInstallationDigest != "") &&
			(!row.DevicePolicyInstalled || row.DevicePolicyIntended) &&
			(!row.ProcessPolicyInstalled || row.ProcessPolicyIntended)
		terminalEvidence := row.CgroupEmpty != nil && row.AcceleratorContextsEmpty != nil &&
			row.ContextAuditDigest != "" && row.TerminalReceiptDigest != ""
		phaseEvidenceValid := (phase == "launch_intent" && row.HostPID == nil && row.ProcessStarttimeTicks == nil && !terminalEvidence) ||
			(phase == "spawned" && hasProcessIdentity && !terminalEvidence) ||
			(phase == "terminal_pending_release" && hasProcessIdentity && terminalEvidence)
		if !policyEvidenceValid || !phaseEvidenceValid {
			return HostAuthorityStatus{}, fmt.Errorf("TrainVM authority returned contradictory process enforcement evidence")
		}
		result.ActiveProcesses = append(result.ActiveProcesses, row)
	}
	return result, nil
}

func submissionRPCRequest(request SubmissionRequest) (*trainvmv1.SubmitExperimentRequest, error) {
	request.SourceFormat = strings.TrimSpace(strings.ToLower(request.SourceFormat))
	request.IdempotencyKey = strings.TrimSpace(request.IdempotencyKey)
	request.ExpectedJournalID = strings.TrimSpace(request.ExpectedJournalID)
	request.ExpectedPlanHash = strings.TrimSpace(request.ExpectedPlanHash)
	request.ExpectedAdapterLockDigest = strings.TrimSpace(request.ExpectedAdapterLockDigest)
	request.ExpectedTrainingComponentLockDigest = strings.TrimSpace(request.ExpectedTrainingComponentLockDigest)
	request.Author = strings.TrimSpace(request.Author)
	request.Reason = strings.TrimSpace(request.Reason)
	request.ForkedFromRunID = strings.TrimSpace(request.ForkedFromRunID)
	request.ExpectedParentPlanHash = strings.TrimSpace(request.ExpectedParentPlanHash)
	if request.SourceDocument == "" || request.ExpectedJournalID == "" ||
		(request.SourceFormat != "json" && request.SourceFormat != "yaml") {
		return nil, &ValidationError{Message: "source document, json/yaml format, and journal ID are required"}
	}
	if request.CreateRun && (request.IdempotencyKey == "" || request.Author == "" ||
		request.Reason == "" || request.ExpectedPlanHash == "" || request.ExpectedAdapterLockDigest == "") {
		return nil, &ValidationError{Message: "run creation requires an idempotency key, author, reason, and expected plan and adapter-lock hashes"}
	}
	hasForkIdentity := request.ForkedFromRunID != "" || request.ExpectedParentRunRevision != 0 || request.ExpectedParentPlanHash != ""
	if hasForkIdentity && (!request.CreateRun || request.ForkedFromRunID == "" || len(request.ForkedFromRunID) > 256 ||
		request.ExpectedParentRunRevision == 0 || request.ExpectedParentPlanHash == "") {
		return nil, &ValidationError{Message: "run fork requires a bounded parent run ID, revision, and plan hash"}
	}
	return &trainvmv1.SubmitExperimentRequest{
		SourceDocument: request.SourceDocument, SourceFormat: request.SourceFormat,
		CreateRun: request.CreateRun, IdempotencyKey: request.IdempotencyKey,
		ExpectedJournalId: request.ExpectedJournalID, Author: request.Author, Reason: request.Reason,
		ExpectedPlanHash:                    request.ExpectedPlanHash,
		ExpectedAdapterLockDigest:           request.ExpectedAdapterLockDigest,
		ExpectedTrainingComponentLockDigest: request.ExpectedTrainingComponentLockDigest,
		ForkedFromRunId:                     request.ForkedFromRunID,
		ExpectedParentRunRevision:           request.ExpectedParentRunRevision,
		ExpectedParentPlanHash:              request.ExpectedParentPlanHash,
	}, nil
}

func planDiffRPCRequest(request PlanDiffRequest) (*trainvmv1.PlanDiffRequest, error) {
	request.RunID = strings.TrimSpace(request.RunID)
	request.SourceFormat = strings.TrimSpace(strings.ToLower(request.SourceFormat))
	request.ExpectedJournalID = strings.TrimSpace(request.ExpectedJournalID)
	request.ExpectedCurrentPlanHash = strings.TrimSpace(request.ExpectedCurrentPlanHash)
	request.ExpectedProposedPlanHash = strings.TrimSpace(request.ExpectedProposedPlanHash)
	if request.RunID == "" || len(request.RunID) > 256 || request.ExpectedRunRevision == 0 ||
		request.ProposedSourceDocument == "" || len(request.ProposedSourceDocument) > 2<<20 ||
		(request.SourceFormat != "json" && request.SourceFormat != "yaml") ||
		request.ExpectedJournalID == "" || request.ExpectedCurrentPlanHash == "" ||
		request.ExpectedProposedPlanHash == "" {
		return nil, &ValidationError{Message: "plan diff requires bounded run, revision, source, journal, current-plan, and proposed-plan identities"}
	}
	return &trainvmv1.PlanDiffRequest{
		RunId: request.RunID, ExpectedRevision: request.ExpectedRunRevision,
		ProposedSourceDocument: request.ProposedSourceDocument, SourceFormat: request.SourceFormat,
		ExpectedJournalId:        request.ExpectedJournalID,
		ExpectedCurrentPlanHash:  request.ExpectedCurrentPlanHash,
		ExpectedProposedPlanHash: request.ExpectedProposedPlanHash,
	}, nil
}

func timestampNS(value *timestamppb.Timestamp) (int64, error) {
	if value == nil {
		return 0, nil
	}
	if err := value.CheckValid(); err != nil {
		return 0, fmt.Errorf("TrainVM authority returned an invalid timestamp: %w", err)
	}
	return value.AsTime().UnixNano(), nil
}

func runFromProto(summary *trainvmv1.RunSummary) (Run, error) {
	if summary == nil || summary.GetIdentity() == nil ||
		strings.TrimSpace(summary.GetIdentity().GetRunId()) == "" ||
		summary.GetIdentity().GetPlanHash() == "" ||
		summary.GetDesiredState() == trainvmv1.DesiredState_DESIRED_STATE_UNSPECIFIED ||
		summary.GetObservedState() == trainvmv1.ObservedState_OBSERVED_STATE_UNSPECIFIED {
		return Run{}, fmt.Errorf("TrainVM authority returned a malformed run summary")
	}
	heartbeat, err := timestampNS(summary.GetLastHeartbeatAt())
	if err != nil {
		return Run{}, err
	}
	return Run{
		RunID:            summary.GetIdentity().GetRunId(),
		ExperimentName:   summary.GetExperimentName(),
		PlanHash:         summary.GetIdentity().GetPlanHash(),
		DesiredState:     strings.ToLower(strings.TrimPrefix(summary.GetDesiredState().String(), "DESIRED_STATE_")),
		ObservedState:    strings.ToLower(strings.TrimPrefix(summary.GetObservedState().String(), "OBSERVED_STATE_")),
		CurrentNodeID:    summary.GetCurrentNodeId(),
		CurrentAttemptID: summary.GetCurrentAttemptId(),
		RunRevision:      summary.GetIdentity().GetRevision(),
		OptimizerStep:    summary.GetOptimizerStep(),
		LastHeartbeatNS:  heartbeat,
		LastEventSeq:     summary.GetLastEventSequence(),
		FailureSummary:   summary.GetFailureSummary(),

		ForkedFromRunID:       summary.GetForkedFromRunId(),
		ForkedFromRunRevision: summary.GetForkedFromRunRevision(),
		ForkedFromPlanHash:    summary.GetForkedFromPlanHash(),
	}, nil
}

func eventFromProto(envelope *trainvmv1.EventEnvelope) (Event, error) {
	if envelope == nil || envelope.GetJournalSequence() == 0 ||
		envelope.GetEventId() == "" || envelope.GetRunId() == "" ||
		envelope.GetEventType() == "" || envelope.GetEventVersion() == 0 {
		return Event{}, fmt.Errorf("TrainVM authority returned a malformed event identity")
	}
	wallTime, err := timestampNS(envelope.GetWallTime())
	if err != nil {
		return Event{}, err
	}
	payload := envelope.GetCanonicalJsonPayload()
	if !json.Valid(payload) {
		return Event{}, fmt.Errorf("TrainVM event %q contains invalid payload JSON", envelope.GetEventId())
	}
	result := Event{
		Sequence: envelope.GetJournalSequence(), EventID: envelope.GetEventId(),
		RunID: envelope.GetRunId(), RunRevision: envelope.GetRunRevision(),
		PlanRevision: envelope.GetPlanRevision(), NodeID: envelope.GetNodeId(),
		AttemptID: envelope.GetAttemptId(), WorkerSequence: envelope.GetWorkerSequence(),
		EventType: envelope.GetEventType(), EventVersion: envelope.GetEventVersion(),
		WallTimeNS: wallTime, MonotonicTimeNS: envelope.GetMonotonicTimeNs(),
		Payload: append(json.RawMessage(nil), payload...),
	}
	if envelope.OptimizerStep != nil {
		step := envelope.GetOptimizerStep()
		result.OptimizerStep = &step
	}
	return result, nil
}

func controlTypeName(value trainvmv1.ControlType) (string, error) {
	switch value {
	case trainvmv1.ControlType_CONTROL_TYPE_NUMBER:
		return "number", nil
	case trainvmv1.ControlType_CONTROL_TYPE_INTEGER:
		return "integer", nil
	case trainvmv1.ControlType_CONTROL_TYPE_BOOLEAN:
		return "boolean", nil
	case trainvmv1.ControlType_CONTROL_TYPE_STRING:
		return "string", nil
	case trainvmv1.ControlType_CONTROL_TYPE_ENUMERATION:
		return "enum", nil
	default:
		return "", fmt.Errorf("TrainVM authority returned an unknown control type")
	}
}

func assignmentMap(values []*trainvmv1.ControlAssignment) (map[string]any, error) {
	result := make(map[string]any, len(values))
	for _, assignment := range values {
		if assignment == nil || assignment.GetKey() == "" || assignment.GetValue() == nil {
			return nil, fmt.Errorf("TrainVM authority returned a malformed control assignment")
		}
		if _, exists := result[assignment.GetKey()]; exists {
			return nil, fmt.Errorf("TrainVM authority returned a duplicate control assignment")
		}
		value, err := checkedScalarJSONValue(assignment.GetValue())
		if err != nil {
			return nil, err
		}
		result[assignment.GetKey()] = value
	}
	return result, nil
}

func rawJSON(value any) (json.RawMessage, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode TrainVM control view: %w", err)
	}
	return json.RawMessage(encoded), nil
}

func controlViewFromProto(response *trainvmv1.GetControlViewResponse) (ControlView, error) {
	if response == nil {
		return ControlView{}, fmt.Errorf("TrainVM authority returned no control view")
	}
	view := ControlView{
		Catalog:                 make(map[string]ControlDescriptor, len(response.GetCatalog())),
		EffectiveValues:         make(map[string]any, len(response.GetEffectiveValues())),
		LatestRequestedRevision: response.GetLatestRequestedRevision(),
		LatestEffectiveRevision: response.GetLatestEffectiveRevision(),
		Commands:                []ControlCommandView{},
	}
	for name, wire := range response.GetCatalog() {
		if name == "" || wire == nil || wire.GetDefaultValue() == nil {
			return ControlView{}, fmt.Errorf("TrainVM authority returned a malformed control descriptor")
		}
		typeName, err := controlTypeName(wire.GetType())
		if err != nil {
			return ControlView{}, err
		}
		defaultValue, err := checkedScalarJSONValue(wire.GetDefaultValue())
		if err != nil || wire.GetApplyPoint() == trainvmv1.ApplyPoint_APPLY_POINT_UNSPECIFIED {
			return ControlView{}, fmt.Errorf("TrainVM authority returned a malformed control descriptor")
		}
		descriptor := ControlDescriptor{
			Type: typeName, Default: defaultValue,
			Apply:             strings.ToLower(strings.TrimPrefix(wire.GetApplyPoint().String(), "APPLY_POINT_")),
			MutableAfterStart: wire.GetMutableAfterStart(), RequiresPause: wire.GetRequiresPause(),
			Description: wire.GetDescription(), Unit: wire.GetUnit(),
		}
		if wire.Minimum != nil {
			value := wire.GetMinimum()
			descriptor.Minimum = &value
		}
		if wire.Maximum != nil {
			value := wire.GetMaximum()
			descriptor.Maximum = &value
		}
		for _, value := range wire.GetValues() {
			converted, convertErr := checkedScalarJSONValue(value)
			if convertErr != nil {
				return ControlView{}, convertErr
			}
			descriptor.Values = append(descriptor.Values, converted)
		}
		view.Catalog[name] = descriptor
	}
	effective, err := assignmentMap(response.GetEffectiveValues())
	if err != nil {
		return ControlView{}, err
	}
	view.EffectiveValues = effective
	for _, wire := range response.GetCommands() {
		if wire == nil || wire.GetCommandId() == "" || wire.GetControlRevision() == 0 ||
			wire.GetApplyPoint() == trainvmv1.ApplyPoint_APPLY_POINT_UNSPECIFIED ||
			wire.GetStatus() == trainvmv1.ControlCommandResult_STATUS_UNSPECIFIED {
			return ControlView{}, fmt.Errorf("TrainVM authority returned a malformed control command")
		}
		assignments, err := assignmentMap(wire.GetAssignments())
		if err != nil {
			return ControlView{}, err
		}
		effectiveValues, err := assignmentMap(wire.GetEffectiveValues())
		if err != nil {
			return ControlView{}, err
		}
		diagnostics := make([]ControlDiagnostic, 0, len(wire.GetDiagnostics()))
		for _, diagnostic := range wire.GetDiagnostics() {
			if diagnostic == nil ||
				diagnostic.GetSeverity() == trainvmv1.Diagnostic_SEVERITY_UNSPECIFIED {
				return ControlView{}, fmt.Errorf("TrainVM authority returned a malformed control diagnostic")
			}
			diagnostics = append(diagnostics, ControlDiagnostic{
				Severity: strings.ToLower(strings.TrimPrefix(diagnostic.GetSeverity().String(), "SEVERITY_")),
				Code:     diagnostic.GetCode(), Path: diagnostic.GetDocumentPath(),
				Message: diagnostic.GetMessage(), Help: diagnostic.GetHelp(),
			})
		}
		assignmentsJSON, err := rawJSON(assignments)
		if err != nil {
			return ControlView{}, err
		}
		effectiveJSON, err := rawJSON(effectiveValues)
		if err != nil {
			return ControlView{}, err
		}
		diagnosticsJSON, err := rawJSON(diagnostics)
		if err != nil {
			return ControlView{}, err
		}
		command := ControlCommandView{
			CommandID: wire.GetCommandId(), ControlRevision: wire.GetControlRevision(),
			ApplyPoint:  strings.ToLower(strings.TrimPrefix(wire.GetApplyPoint().String(), "APPLY_POINT_")),
			Assignments: assignmentsJSON, Author: wire.GetAuthor(), Reason: wire.GetReason(),
			Status:          strings.ToLower(strings.TrimPrefix(wire.GetStatus().String(), "STATUS_")),
			EffectiveValues: effectiveJSON, Diagnostics: diagnosticsJSON,
		}
		if wire.EffectiveStep != nil {
			step := wire.GetEffectiveStep()
			command.EffectiveStep = &step
		}
		view.Commands = append(view.Commands, command)
	}
	return view, nil
}

func controlRPCRequest(request ControlRequest) (*trainvmv1.RunCommandRequest, error) {
	request.RunID = strings.TrimSpace(request.RunID)
	request.ExpectedJournalID = strings.TrimSpace(request.ExpectedJournalID)
	request.ExpectedPlanHash = strings.TrimSpace(request.ExpectedPlanHash)
	request.IdempotencyKey = strings.TrimSpace(request.IdempotencyKey)
	request.Author = strings.TrimSpace(request.Author)
	request.Reason = strings.TrimSpace(request.Reason)
	if request.RunID == "" || request.ExpectedJournalID == "" || request.ExpectedPlanHash == "" ||
		request.IdempotencyKey == "" || request.Author == "" || request.Reason == "" {
		return nil, invalidControlRequest(
			"run ID, journal ID, plan hash, idempotency key, author, and reason are required")
	}
	if len(request.Assignments) == 0 {
		return nil, invalidControlRequest("control assignments must not be empty")
	}
	keys := make([]string, 0, len(request.Assignments))
	for key := range request.Assignments {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	assignments := make([]*trainvmv1.ControlAssignment, 0, len(keys))
	for _, key := range keys {
		if strings.TrimSpace(key) == "" {
			return nil, invalidControlRequest("control assignment key must not be empty")
		}
		value, err := scalarValue(request.Assignments[key])
		if err != nil {
			return nil, invalidControlRequest("control %q: %v", key, err)
		}
		assignments = append(assignments, &trainvmv1.ControlAssignment{Key: key, Value: value})
	}
	return &trainvmv1.RunCommandRequest{
		RunId:               request.RunID,
		ExpectedRunRevision: request.ExpectedRunRevision,
		IdempotencyKey:      request.IdempotencyKey,
		Author:              request.Author,
		Reason:              request.Reason,
		ExpectedJournalId:   request.ExpectedJournalID,
		ExpectedPlanHash:    request.ExpectedPlanHash,
		Command: &trainvmv1.RunCommandRequest_Controls{Controls: &trainvmv1.ControlPatchCommand{
			ExpectedControlRevision: request.ExpectedControlRevision,
			Assignments:             assignments,
		}},
	}, nil
}

func runActionRPCRequest(request RunActionRequest) (*trainvmv1.RunCommandRequest, error) {
	request.RunID = strings.TrimSpace(request.RunID)
	request.ExpectedJournalID = strings.TrimSpace(request.ExpectedJournalID)
	request.ExpectedPlanHash = strings.TrimSpace(request.ExpectedPlanHash)
	request.IdempotencyKey = strings.TrimSpace(request.IdempotencyKey)
	request.Author = strings.TrimSpace(request.Author)
	request.Reason = strings.TrimSpace(request.Reason)
	request.Action = strings.ToLower(strings.TrimSpace(request.Action))
	if request.RunID == "" || request.ExpectedJournalID == "" || request.ExpectedPlanHash == "" ||
		request.IdempotencyKey == "" || request.Author == "" || request.Reason == "" {
		return nil, invalidControlRequest(
			"run ID, journal ID, plan hash, idempotency key, author, and reason are required")
	}
	if len(request.RunID) > 256 || len(request.ExpectedJournalID) > 256 ||
		len(request.ExpectedPlanHash) > 256 || len(request.IdempotencyKey) > 256 ||
		len(request.Author) > 256 || len(request.Reason) > 4096 {
		return nil, invalidControlRequest("lifecycle command contains an overlong bounded field")
	}
	if request.Action != "pause" && (request.CheckpointFirst || request.ReleaseResources) {
		return nil, invalidControlRequest("checkpoint_first and release_resources are valid only for pause")
	}
	if request.ReleaseResources && !request.CheckpointFirst {
		return nil, invalidControlRequest("a resource-releasing pause must checkpoint first")
	}
	if request.Action != "cancel" && request.GracefulTimeoutSeconds != 0 {
		return nil, invalidControlRequest("graceful_timeout_seconds is valid only for cancel")
	}
	if request.GracefulTimeoutSeconds > 86400 {
		return nil, invalidControlRequest("graceful_timeout_seconds must not exceed 86400")
	}

	wire := &trainvmv1.RunCommandRequest{
		RunId: request.RunID, ExpectedRunRevision: request.ExpectedRunRevision,
		IdempotencyKey: request.IdempotencyKey, Author: request.Author, Reason: request.Reason,
		ExpectedJournalId: request.ExpectedJournalID, ExpectedPlanHash: request.ExpectedPlanHash,
	}
	switch request.Action {
	case "checkpoint":
		wire.Command = &trainvmv1.RunCommandRequest_Checkpoint{Checkpoint: &trainvmv1.CheckpointCommand{Reason: request.Reason}}
	case "pause":
		wire.Command = &trainvmv1.RunCommandRequest_Pause{Pause: &trainvmv1.PauseCommand{
			CheckpointFirst: request.CheckpointFirst, ReleaseResources: request.ReleaseResources,
		}}
	case "resume":
		wire.Command = &trainvmv1.RunCommandRequest_Resume{Resume: &trainvmv1.ResumeCommand{}}
	case "cancel":
		timeout := durationpb.New(time.Duration(request.GracefulTimeoutSeconds) * time.Second)
		if err := timeout.CheckValid(); err != nil {
			return nil, invalidControlRequest("invalid graceful timeout: %v", err)
		}
		wire.Command = &trainvmv1.RunCommandRequest_Cancel{Cancel: &trainvmv1.CancelCommand{
			Reason: request.Reason, GracefulTimeout: timeout,
		}}
	default:
		return nil, invalidControlRequest("action must be checkpoint, pause, resume, or cancel")
	}
	return wire, nil
}

func scalarValue(value any) (*trainvmv1.ScalarValue, error) {
	switch typed := value.(type) {
	case json.Number:
		if strings.ContainsAny(string(typed), ".eE") {
			parsed, err := strconv.ParseFloat(string(typed), 64)
			if err != nil || math.IsInf(parsed, 0) || math.IsNaN(parsed) {
				return nil, fmt.Errorf("invalid finite number")
			}
			return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: parsed}}, nil
		}
		parsed, err := strconv.ParseInt(string(typed), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid signed integer")
		}
		return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_IntegerValue{IntegerValue: parsed}}, nil
	case float64:
		if math.IsInf(typed, 0) || math.IsNaN(typed) {
			return nil, fmt.Errorf("number must be finite")
		}
		return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: typed}}, nil
	case int:
		return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_IntegerValue{IntegerValue: int64(typed)}}, nil
	case int64:
		return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_IntegerValue{IntegerValue: typed}}, nil
	case bool:
		return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_BooleanValue{BooleanValue: typed}}, nil
	case string:
		return &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_StringValue{StringValue: typed}}, nil
	default:
		return nil, fmt.Errorf("unsupported scalar type %T", value)
	}
}

func controlResult(response *trainvmv1.RunCommandResponse) ControlResult {
	result := ControlResult{
		Disposition: strings.TrimPrefix(response.GetDisposition().String(), "DISPOSITION_"),
		Assignments: map[string]any{},
	}
	if control := response.GetControl(); control != nil {
		result.CommandID = control.GetCommandId()
		result.ControlRevision = control.GetControlRevision()
		result.ApplyPoint = strings.TrimPrefix(control.GetApplyPoint().String(), "APPLY_POINT_")
		result.RequiresPause = control.GetRequiresPause()
		result.Status = strings.TrimPrefix(control.GetStatus().String(), "STATUS_")
		for _, assignment := range control.GetAssignments() {
			result.Assignments[assignment.GetKey()] = scalarJSONValue(assignment.GetValue())
		}
	}
	for _, diagnostic := range response.GetDiagnostics() {
		result.Diagnostics = append(result.Diagnostics, ControlDiagnostic{
			Severity: strings.TrimPrefix(diagnostic.GetSeverity().String(), "SEVERITY_"),
			Code:     diagnostic.GetCode(), Path: diagnostic.GetDocumentPath(),
			Message: diagnostic.GetMessage(), Help: diagnostic.GetHelp(),
		})
	}
	return result
}

func runActionResult(action string, response *trainvmv1.RunCommandResponse) (RunActionResult, error) {
	if response == nil {
		return RunActionResult{}, fmt.Errorf("TrainVM authority returned an empty lifecycle response")
	}
	action = strings.ToLower(strings.TrimSpace(action))
	result := RunActionResult{
		Disposition:     strings.TrimPrefix(response.GetDisposition().String(), "DISPOSITION_"),
		CommandSequence: response.GetCommandSequence(), Action: action,
	}
	for _, diagnostic := range response.GetDiagnostics() {
		result.Diagnostics = append(result.Diagnostics, ControlDiagnostic{
			Severity: strings.TrimPrefix(diagnostic.GetSeverity().String(), "SEVERITY_"),
			Code:     diagnostic.GetCode(), Path: diagnostic.GetDocumentPath(),
			Message: diagnostic.GetMessage(), Help: diagnostic.GetHelp(),
		})
	}
	if action == "checkpoint" {
		if checkpoint := response.GetCheckpoint(); checkpoint != nil {
			result.CommandID = checkpoint.GetCommandId()
			result.ControllerSequence = checkpoint.GetControllerSequence()
			result.Status = strings.TrimPrefix(checkpoint.GetStatus().String(), "STATUS_")
			result.Reason = checkpoint.GetReason()
			result.OptimizerStep = checkpoint.GetOptimizerStep()
			result.ArtifactID = checkpoint.GetArtifactId()
		} else if result.Disposition == "ACCEPTED" || result.Disposition == "ALREADY_APPLIED" {
			return RunActionResult{}, fmt.Errorf("TrainVM authority omitted the checkpoint result")
		}
		if response.GetLifecycle() != nil {
			return RunActionResult{}, fmt.Errorf("TrainVM authority returned a lifecycle result for checkpoint")
		}
		return result, nil
	}
	if response.GetCheckpoint() != nil {
		return RunActionResult{}, fmt.Errorf("TrainVM authority returned a checkpoint result for %s", action)
	}
	if lifecycle := response.GetLifecycle(); lifecycle != nil {
		kind := strings.ToLower(strings.TrimPrefix(lifecycle.GetKind().String(), "KIND_"))
		if kind != action {
			return RunActionResult{}, fmt.Errorf("TrainVM authority returned %s result for %s", kind, action)
		}
		result.CommandID = lifecycle.GetCommandId()
		result.ControllerSequence = lifecycle.GetControllerSequence()
		result.Status = strings.TrimPrefix(lifecycle.GetStatus().String(), "STATUS_")
		result.CheckpointFirst = lifecycle.GetCheckpointFirst()
		result.ReleaseResources = lifecycle.GetReleaseResources()
		result.OptimizerStep = lifecycle.GetOptimizerStep()
		result.ArtifactID = lifecycle.GetArtifactId()
		result.Reason = lifecycle.GetReason()
		if timeout := lifecycle.GetGracefulTimeout(); timeout != nil {
			if err := timeout.CheckValid(); err != nil || timeout.Nanos != 0 || timeout.Seconds < 0 {
				return RunActionResult{}, fmt.Errorf("TrainVM authority returned an invalid graceful timeout")
			}
			result.GracefulTimeoutSeconds = uint64(timeout.Seconds)
		}
	} else if result.Disposition == "ACCEPTED" || result.Disposition == "ALREADY_APPLIED" {
		return RunActionResult{}, fmt.Errorf("TrainVM authority omitted the %s lifecycle result", action)
	}
	return result, nil
}

func scalarJSONValue(value *trainvmv1.ScalarValue) any {
	if value == nil {
		return nil
	}
	switch typed := value.Value.(type) {
	case *trainvmv1.ScalarValue_NumberValue:
		return typed.NumberValue
	case *trainvmv1.ScalarValue_IntegerValue:
		return typed.IntegerValue
	case *trainvmv1.ScalarValue_BooleanValue:
		return typed.BooleanValue
	case *trainvmv1.ScalarValue_StringValue:
		return typed.StringValue
	default:
		return nil
	}
}

func checkedScalarJSONValue(value *trainvmv1.ScalarValue) (any, error) {
	converted := scalarJSONValue(value)
	if converted == nil {
		return nil, fmt.Errorf("TrainVM authority returned an unset scalar value")
	}
	return converted, nil
}
