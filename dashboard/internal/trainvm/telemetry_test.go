package trainvm

import (
	"context"
	"encoding/json"
	"testing"
)

type telemetryFixtureReader struct {
	events []Event
	query  EventQuery
}

func (*telemetryFixtureReader) JournalID(context.Context) (string, error) { return "journal", nil }
func (*telemetryFixtureReader) Runs(context.Context) ([]Run, error)       { return nil, nil }
func (*telemetryFixtureReader) Run(context.Context, string) (Run, bool, error) {
	return Run{}, false, nil
}
func (*telemetryFixtureReader) CompiledPlan(context.Context, string) (CompiledPlanView, bool, error) {
	return CompiledPlanView{}, false, nil
}
func (r *telemetryFixtureReader) Events(_ context.Context, query EventQuery) ([]Event, error) {
	r.query = query
	return r.events, nil
}
func (r *telemetryFixtureReader) Timeline(ctx context.Context, runID string, after uint64, limit int) ([]Event, error) {
	return r.Events(ctx, EventQuery{RunID: runID, After: after, Limit: limit})
}
func (*telemetryFixtureReader) Controls(context.Context, string) (ControlView, bool, error) {
	return ControlView{}, false, nil
}

func TestTypedMetricAndArtifactViewsUseFilteredDurableCursors(t *testing.T) {
	metricStep := uint64(11)
	metricReader := &telemetryFixtureReader{events: []Event{{
		Sequence: 9, EventID: "metric-9", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", WorkerSequence: 4, EventType: "metric.sampled",
		WallTimeNS: 2_000, OptimizerStep: &metricStep, Payload: json.RawMessage(`{
			"name":"loss","value":1.25,"unit":"loss","step_domain":"optimizer_step",
			"step":11,"sample_weight":1,"labels":{"route":"animation"}
		}`),
	}}}
	metrics, err := Metrics(context.Background(), metricReader, "run-1", 7, 25)
	if err != nil || len(metrics) != 1 || metrics[0].Name != "loss" ||
		metrics[0].Step != 11 || metrics[0].Labels["route"] != "animation" ||
		metricReader.query.After != 7 || metricReader.query.Limit != 25 ||
		len(metricReader.query.EventTypes) != 1 ||
		metricReader.query.EventTypes[0] != "metric.sampled" {
		t.Fatalf("unexpected metric projection: %#v query=%#v err=%v", metrics, metricReader.query, err)
	}

	artifactReader := &telemetryFixtureReader{events: []Event{{
		Sequence: 10, EventID: "artifact-10", RunID: "run-1", NodeID: "eval",
		AttemptID: "eval@1", WorkerSequence: 5, EventType: "artifact.published",
		WallTimeNS: 3_000,
		Payload: json.RawMessage(`{
			"artifact_id":"gallery-11","logical_name":"eval/gallery","kind":"image_gallery",
			"schema":"trainvm.eval-gallery/v1","uri":"file:///sealed/gallery-11","size_bytes":4096,
			"fingerprint_algorithm":"sha256","fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"complete":true,"producer_node_id":"eval","producer_attempt_id":"eval@1",
			"parent_artifact_ids":["checkpoint-10"],"published_at_ns":3000
		}`),
	}}}
	artifacts, err := Artifacts(context.Background(), artifactReader, "run-1", 9, 10)
	if err != nil || len(artifacts) != 1 || artifacts[0].ArtifactID != "gallery-11" ||
		artifacts[0].Kind != "image_gallery" || len(artifacts[0].ParentArtifactIDs) != 1 ||
		artifactReader.query.EventTypes[0] != "artifact.published" {
		t.Fatalf("unexpected artifact projection: %#v query=%#v err=%v", artifacts, artifactReader.query, err)
	}
}

func TestTypedTelemetryRejectsMalformedAuthorityPayloads(t *testing.T) {
	reader := &telemetryFixtureReader{events: []Event{{
		Sequence: 1, EventID: "bad", RunID: "run-1", EventType: "metric.sampled",
		Payload: json.RawMessage(`{"name":"loss","value":1,"step_domain":"optimizer_step","step":1,"sample_weight":0,"labels":{},"extra":true}`),
	}}}
	if _, err := Metrics(context.Background(), reader, "run-1", 0, 10); err == nil {
		t.Fatal("malformed metric payload unexpectedly crossed the read-model boundary")
	}
	if _, err := normalizeEventQuery(EventQuery{
		RunID: "run-1", EventTypes: []string{"metric.sampled", "metric.sampled"},
	}); err == nil {
		t.Fatal("duplicate event filter unexpectedly accepted")
	}
}
