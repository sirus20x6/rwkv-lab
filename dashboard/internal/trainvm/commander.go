package trainvm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	trainvmv1 "trainboard/gen/trainvm/v1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Commander is the dashboard's mutation boundary. Implementations send typed
// commands to the native TrainVM authority and never receive a journal path.
type Commander interface {
	SubmitExperiment(context.Context, SubmissionRequest) (SubmissionResult, error)
	RequestControls(context.Context, ControlRequest) (ControlResult, error)
	GetDescriptor(context.Context, DescriptorRequest) (DescriptorResult, error)
}

type DescriptorRequest struct {
	Provider string
	Version  string
}

type DescriptorResult struct {
	SchemaJSON string `json:"schema_json"`
	SchemaHash string `json:"schema_hash"`
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

func submissionRPCRequest(request SubmissionRequest) (*trainvmv1.SubmitExperimentRequest, error) {
	request.SourceFormat = strings.TrimSpace(strings.ToLower(request.SourceFormat))
	request.IdempotencyKey = strings.TrimSpace(request.IdempotencyKey)
	request.ExpectedJournalID = strings.TrimSpace(request.ExpectedJournalID)
	request.ExpectedPlanHash = strings.TrimSpace(request.ExpectedPlanHash)
	request.ExpectedAdapterLockDigest = strings.TrimSpace(request.ExpectedAdapterLockDigest)
	request.ExpectedTrainingComponentLockDigest = strings.TrimSpace(request.ExpectedTrainingComponentLockDigest)
	request.Author = strings.TrimSpace(request.Author)
	request.Reason = strings.TrimSpace(request.Reason)
	if request.SourceDocument == "" || request.ExpectedJournalID == "" ||
		(request.SourceFormat != "json" && request.SourceFormat != "yaml") {
		return nil, &ValidationError{Message: "source document, json/yaml format, and journal ID are required"}
	}
	if request.CreateRun && (request.IdempotencyKey == "" || request.Author == "" ||
		request.Reason == "" || request.ExpectedPlanHash == "" || request.ExpectedAdapterLockDigest == "") {
		return nil, &ValidationError{Message: "run creation requires an idempotency key, author, reason, and expected plan and adapter-lock hashes"}
	}
	return &trainvmv1.SubmitExperimentRequest{
		SourceDocument: request.SourceDocument, SourceFormat: request.SourceFormat,
		CreateRun: request.CreateRun, IdempotencyKey: request.IdempotencyKey,
		ExpectedJournalId: request.ExpectedJournalID, Author: request.Author, Reason: request.Reason,
		ExpectedPlanHash:                    request.ExpectedPlanHash,
		ExpectedAdapterLockDigest:           request.ExpectedAdapterLockDigest,
		ExpectedTrainingComponentLockDigest: request.ExpectedTrainingComponentLockDigest,
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
