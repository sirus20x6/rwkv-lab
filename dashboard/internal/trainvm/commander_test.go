package trainvm

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"path/filepath"
	"testing"
	"time"

	trainvmv1 "trainboard/gen/trainvm/v1"

	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestControlRPCRequestPreservesTypedScalarsAndSortsKeys(t *testing.T) {
	request, err := controlRPCRequest(ControlRequest{
		RunID: "run-1", ExpectedJournalID: "journal-1", ExpectedPlanHash: "plan-1",
		ExpectedRunRevision: 7, ExpectedControlRevision: 2,
		IdempotencyKey: "tab-intent-1", Author: "operator", Reason: "tune",
		Assignments: map[string]any{
			"zeta":  json.Number("3"),
			"beta":  true,
			"alpha": json.Number("0.25"),
		},
	})
	if err != nil {
		t.Fatalf("map control request: %v", err)
	}
	controls := request.GetControls()
	if request.GetRunId() != "run-1" || request.GetExpectedJournalId() != "journal-1" ||
		request.GetExpectedPlanHash() != "plan-1" || request.GetExpectedRunRevision() != 7 ||
		controls.GetExpectedControlRevision() != 2 || len(controls.GetAssignments()) != 3 {
		t.Fatalf("unexpected mapped request: %#v", request)
	}
	if controls.Assignments[0].GetKey() != "alpha" || controls.Assignments[0].GetValue().GetNumberValue() != 0.25 ||
		controls.Assignments[1].GetKey() != "beta" || !controls.Assignments[1].GetValue().GetBooleanValue() ||
		controls.Assignments[2].GetKey() != "zeta" || controls.Assignments[2].GetValue().GetIntegerValue() != 3 {
		t.Fatalf("typed assignments were not deterministic: %#v", controls.Assignments)
	}
}

func TestSubmissionRequiresConfiguredAuthority(t *testing.T) {
	commander := &GRPCCommander{client: nil}
	_, err := commander.SubmitExperiment(context.Background(), SubmissionRequest{})
	if err == nil {
		t.Fatal("incomplete submission unexpectedly accepted")
	}
}

func TestDescriptorRequiresConfiguredAuthority(t *testing.T) {
	commander := &GRPCCommander{client: nil}
	_, err := commander.GetDescriptor(context.Background(), DescriptorRequest{
		Provider: "trainvm.training-components", Version: "1.0.0",
	})
	if err == nil {
		t.Fatal("descriptor request without an authority unexpectedly accepted")
	}
}

func TestSubmissionRPCRequestFencesAuthorityAndPreview(t *testing.T) {
	request, err := submissionRPCRequest(SubmissionRequest{
		SourceDocument: `{"kind":"Experiment"}`, SourceFormat: " JSON ", CreateRun: true,
		IdempotencyKey: " submission-1 ", ExpectedJournalID: " journal-1 ",
		ExpectedPlanHash: " plan-1 ", ExpectedAdapterLockDigest: " lock-1 ",
		ExpectedTrainingComponentLockDigest: " training-lock-1 ",
		Author:                              " operator ", Reason: " launch ",
	})
	if err != nil {
		t.Fatalf("map submission request: %v", err)
	}
	if request.GetSourceFormat() != "json" || !request.GetCreateRun() ||
		request.GetIdempotencyKey() != "submission-1" ||
		request.GetExpectedJournalId() != "journal-1" ||
		request.GetExpectedPlanHash() != "plan-1" ||
		request.GetExpectedAdapterLockDigest() != "lock-1" ||
		request.GetExpectedTrainingComponentLockDigest() != "training-lock-1" ||
		request.GetAuthor() != "operator" ||
		request.GetReason() != "launch" {
		t.Fatalf("submission fence was not preserved: %#v", request)
	}
}

func TestSubmissionRPCRequestRequiresPreviewFenceForCreate(t *testing.T) {
	_, err := submissionRPCRequest(SubmissionRequest{
		SourceDocument: "{}", SourceFormat: "json", CreateRun: true,
		IdempotencyKey: "submission-1", ExpectedJournalID: "journal-1",
		Author: "operator", Reason: "launch",
	})
	var validationError *ValidationError
	if !errors.As(err, &validationError) {
		t.Fatalf("expected ValidationError, got %T: %v", err, err)
	}
}

func TestCommanderReachabilityFailsClosedWhenSocketIsAbsent(t *testing.T) {
	commander, err := DialCommander(filepath.Join(t.TempDir(), "missing.sock"))
	if err != nil {
		t.Fatal(err)
	}
	defer commander.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if commander.Reachable(ctx) {
		t.Fatal("missing authority socket unexpectedly reported ready")
	}
}

func TestControlRPCRequestRejectsWhitespaceIdentityAsValidationError(t *testing.T) {
	_, err := controlRPCRequest(ControlRequest{
		RunID: "run-1", ExpectedJournalID: "journal-1", ExpectedPlanHash: "plan-1",
		IdempotencyKey: "   ", Author: "operator", Reason: "tune",
		Assignments: map[string]any{"rate": json.Number("1")},
	})
	var validationError *ValidationError
	if !errors.As(err, &validationError) {
		t.Fatalf("expected ValidationError, got %T: %v", err, err)
	}
}

func TestControlRPCRequestRejectsUnsupportedOrUnsafeValues(t *testing.T) {
	base := ControlRequest{
		RunID: "run-1", ExpectedJournalID: "journal-1", ExpectedPlanHash: "plan-1",
		IdempotencyKey: "intent", Author: "operator", Reason: "tune",
	}
	for name, value := range map[string]any{
		"object": map[string]any{"nested": true},
		"array":  []any{1},
		"nan":    math.NaN(),
		"huge":   json.Number("18446744073709551615"),
	} {
		t.Run(name, func(t *testing.T) {
			request := base
			request.Assignments = map[string]any{"value": value}
			if _, err := controlRPCRequest(request); err == nil {
				t.Fatal("unsafe scalar unexpectedly mapped")
			}
		})
	}
}

func TestControlResultMapsTypedNativeResponse(t *testing.T) {
	result := controlResult(&trainvmv1.RunCommandResponse{
		Disposition: trainvmv1.RunCommandResponse_DISPOSITION_ALREADY_APPLIED,
		Diagnostics: []*trainvmv1.Diagnostic{{
			Severity: trainvmv1.Diagnostic_SEVERITY_WARNING, Code: "control.note",
			DocumentPath: "/assignments/rate", Message: "rounded",
		}},
		Control: &trainvmv1.ControlCommandResult{
			CommandId: "control-1", ControlRevision: 4,
			ApplyPoint: trainvmv1.ApplyPoint_APPLY_POINT_NEXT_OPTIMIZER_STEP,
			Status:     trainvmv1.ControlCommandResult_STATUS_APPLIED,
			Assignments: []*trainvmv1.ControlAssignment{{
				Key: "rate", Value: &trainvmv1.ScalarValue{
					Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: 0.125},
				},
			}},
		},
	})
	if result.Disposition != "ALREADY_APPLIED" || result.CommandID != "control-1" ||
		result.ControlRevision != 4 || result.ApplyPoint != "NEXT_OPTIMIZER_STEP" ||
		result.Status != "APPLIED" || result.Assignments["rate"] != 0.125 ||
		len(result.Diagnostics) != 1 || result.Diagnostics[0].Code != "control.note" {
		t.Fatalf("unexpected control result: %#v", result)
	}
}

func TestNativeReadProjectionConversionsPreserveCursorsAndTypedControls(t *testing.T) {
	heartbeat := timestamppb.New(time.Unix(12, 34))
	run, err := runFromProto(&trainvmv1.RunSummary{
		Identity:       &trainvmv1.RunIdentity{RunId: "run-1", Revision: 7, PlanHash: "plan-1"},
		ExperimentName: "mageflow", DesiredState: trainvmv1.DesiredState_DESIRED_STATE_RUNNING,
		ObservedState: trainvmv1.ObservedState_OBSERVED_STATE_RUNNING,
		CurrentNodeId: "train", CurrentAttemptId: "train@1", OptimizerStep: 42,
		LastHeartbeatAt: heartbeat, LastEventSequence: 19,
	})
	if err != nil || run.RunID != "run-1" || run.RunRevision != 7 ||
		run.DesiredState != "running" || run.ObservedState != "running" ||
		run.LastHeartbeatNS != heartbeat.AsTime().UnixNano() || run.LastEventSeq != 19 {
		t.Fatalf("unexpected native run conversion: %#v err=%v", run, err)
	}
	step := uint64(42)
	event, err := eventFromProto(&trainvmv1.EventEnvelope{
		JournalSequence: 19, EventId: "event-19", RunId: "run-1",
		RunRevision: 7, PlanRevision: 1, NodeId: "train", AttemptId: "train@1",
		WorkerSequence: 3, EventType: "worker.metric", EventVersion: 1,
		WallTime: heartbeat, MonotonicTimeNs: 50, OptimizerStep: &step,
		CanonicalJsonPayload: []byte(`{"loss":1.25}`),
	})
	if err != nil || event.Sequence != 19 || event.OptimizerStep == nil ||
		*event.OptimizerStep != 42 || string(event.Payload) != `{"loss":1.25}` {
		t.Fatalf("unexpected native event conversion: %#v err=%v", event, err)
	}
	minimum, maximum := 0.0, 1.0
	effectiveStep := uint64(44)
	view, err := controlViewFromProto(&trainvmv1.GetControlViewResponse{
		Catalog: map[string]*trainvmv1.ControlDescriptor{
			"learning_rate": {
				Type:         trainvmv1.ControlType_CONTROL_TYPE_NUMBER,
				DefaultValue: &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: 0.001}},
				Minimum:      &minimum, Maximum: &maximum,
				ApplyPoint:        trainvmv1.ApplyPoint_APPLY_POINT_NEXT_OPTIMIZER_STEP,
				MutableAfterStart: true, Unit: "ratio",
			},
		},
		EffectiveValues: []*trainvmv1.ControlAssignment{{
			Key: "learning_rate", Value: &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: 0.0005}},
		}},
		LatestRequestedRevision: 2, LatestEffectiveRevision: 1,
		Commands: []*trainvmv1.ControlCommandView{{
			CommandId: "control-2", ControlRevision: 2,
			ApplyPoint: trainvmv1.ApplyPoint_APPLY_POINT_NEXT_OPTIMIZER_STEP,
			Assignments: []*trainvmv1.ControlAssignment{{
				Key: "learning_rate", Value: &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: 0.00025}},
			}},
			Author: "operator", Reason: "tune", Status: trainvmv1.ControlCommandResult_STATUS_APPLIED,
			EffectiveStep: &effectiveStep,
			EffectiveValues: []*trainvmv1.ControlAssignment{{
				Key: "learning_rate", Value: &trainvmv1.ScalarValue{Value: &trainvmv1.ScalarValue_NumberValue{NumberValue: 0.00025}},
			}},
		}},
	})
	if err != nil || view.Catalog["learning_rate"].Type != "number" ||
		view.Catalog["learning_rate"].Minimum == nil ||
		view.EffectiveValues["learning_rate"] != 0.0005 ||
		view.LatestRequestedRevision != 2 || len(view.Commands) != 1 ||
		view.Commands[0].Status != "applied" || view.Commands[0].EffectiveStep == nil ||
		string(view.Commands[0].Assignments) != `{"learning_rate":0.00025}` {
		t.Fatalf("unexpected native control view conversion: %#v err=%v", view, err)
	}
}

func TestNativeReadProjectionConversionsRejectUnsetWireScalars(t *testing.T) {
	_, err := controlViewFromProto(&trainvmv1.GetControlViewResponse{
		Catalog: map[string]*trainvmv1.ControlDescriptor{
			"rate": {
				Type:         trainvmv1.ControlType_CONTROL_TYPE_NUMBER,
				DefaultValue: &trainvmv1.ScalarValue{},
				ApplyPoint:   trainvmv1.ApplyPoint_APPLY_POINT_NEXT_OPTIMIZER_STEP,
			},
		},
	})
	if err == nil {
		t.Fatal("unset authority scalar unexpectedly crossed the dashboard boundary")
	}
}
