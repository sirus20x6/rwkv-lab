package trainvm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	trainvmv1 "trainboard/gen/trainvm/v1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
)

// Commander is the dashboard's mutation boundary. Implementations send typed
// commands to the native TrainVM authority and never receive a journal path.
type Commander interface {
	RequestControls(context.Context, ControlRequest) (ControlResult, error)
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
