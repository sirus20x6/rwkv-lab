package trainvm

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"testing"
)

func TestReproducibilityCapsuleBindsOneDurableRunPrefix(t *testing.T) {
	step := uint64(7)
	metric := optimizerMetricEvent(1, step)
	artifact := Event{
		Sequence: 2, EventID: "artifact-2", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", WorkerSequence: 2, EventType: "artifact.published",
		WallTimeNS: 2_000,
		Payload: json.RawMessage(`{
			"artifact_id":"checkpoint-7","logical_name":"checkpoint","kind":"checkpoint",
			"schema":"rwkv-lab.checkpoint.v1","uri":"file:///sealed/checkpoint-7",
			"size_bytes":4096,"fingerprint_algorithm":"sha256",
			"fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"complete":true,"producer_node_id":"train","producer_attempt_id":"train@1",
			"parent_artifact_ids":[],"published_at_ns":2000
		}`),
	}
	requested := Event{
		Sequence: 3, EventID: "control-requested", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", EventType: "control.requested", WallTimeNS: 3_000,
		Payload: json.RawMessage(`{"command_id":"control-1","control_revision":1,"assignments":{"learning_rate":0.0001}}`),
	}
	applied := Event{
		Sequence: 4, EventID: "control-applied", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", WorkerSequence: 3, EventType: "control.applied",
		WallTimeNS: 4_000, OptimizerStep: &step,
		Payload: json.RawMessage(`{"command_id":"control-1","control_revision":1,"effective_values":{"learning_rate":0.0001}}`),
	}
	reader := telemetrySnapshotFixture(
		[]Event{metric, artifact, requested, applied}, 4,
		`[{"name":"train.loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean"}]`,
	)

	capsule, found, err := BuildReproducibilityCapsule(
		context.Background(), reader, "run-1",
	)
	if err != nil || !found {
		t.Fatalf("capture capsule: found=%t err=%v", found, err)
	}
	if capsule.Body.APIVersion != "trainvm.reproducibility-capsule/v1" ||
		capsule.Body.ThroughSequence != 4 || capsule.Body.PlanHash != reader.plan.PlanHash ||
		len(capsule.Body.LatestMetrics) != 1 ||
		capsule.Body.LatestMetrics[0].Name != "train.loss" ||
		len(capsule.Body.Artifacts) != 1 ||
		capsule.Body.Artifacts[0].ArtifactID != "checkpoint-7" ||
		capsule.Body.ArtifactHistoryTruncated || len(capsule.Body.ControlEvents) != 2 ||
		capsule.Body.ControlEvents[0].EventType != "control.requested" ||
		capsule.Body.ControlEvents[1].EventType != "control.applied" {
		t.Fatalf("unexpected capsule body: %#v", capsule.Body)
	}
	encoded, err := json.Marshal(capsule.Body)
	if err != nil {
		t.Fatal(err)
	}
	wantDigest := fmt.Sprintf("sha256:%x", sha256.Sum256(encoded))
	if capsule.CapsuleDigest != wantDigest {
		t.Fatalf("capsule digest=%q want=%q", capsule.CapsuleDigest, wantDigest)
	}
	repeated, found, err := BuildReproducibilityCapsule(
		context.Background(), reader, "run-1",
	)
	if err != nil || !found || repeated.CapsuleDigest != capsule.CapsuleDigest {
		t.Fatalf("same durable prefix was not byte-stable: found=%t capsule=%#v err=%v", found, repeated, err)
	}
}
