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
