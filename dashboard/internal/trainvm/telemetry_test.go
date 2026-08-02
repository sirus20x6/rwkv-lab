package trainvm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
)

type telemetryFixtureReader struct {
	journalID      string
	journalIDs     []string
	journalIDReads int
	run            Run
	plan           CompiledPlanView
	events         []Event
	query          EventQuery
	queries        []EventQuery
}

func (r *telemetryFixtureReader) JournalID(context.Context) (string, error) {
	if len(r.journalIDs) > 0 {
		index := r.journalIDReads
		if index >= len(r.journalIDs) {
			index = len(r.journalIDs) - 1
		}
		r.journalIDReads++
		return r.journalIDs[index], nil
	}
	if r.journalID != "" {
		return r.journalID, nil
	}
	return "journal", nil
}

func TestTelemetrySnapshotRejectsJournalReplacementDuringEventReads(t *testing.T) {
	plan := CompiledPlanView{
		JournalID: "journal-before", RunID: "run-1", RunRevision: 1,
		PlanHash: "plan-1", CanonicalPlan: json.RawMessage(`{
			"spec":{"observability":{"heartbeat_seconds":5,"metrics":[{
				"name":"loss","type":"gauge","unit":"loss",
				"step_domain":"optimizer_step","aggregation":"last"
			}]}}
		}`),
	}
	step := uint64(1)
	reader := &telemetryFixtureReader{
		journalIDs: []string{"journal-before", "journal-before", "journal-after"},
		run: Run{RunID: "run-1", PlanHash: "plan-1", RunRevision: 1,
			CurrentNodeID: "train", CurrentAttemptID: "train@1", LastEventSeq: 2},
		plan: plan,
		events: []Event{{
			Sequence: 2, EventID: "metric-2", RunID: "run-1", NodeID: "train",
			AttemptID: "train@1", WorkerSequence: 1, EventType: "metric.sampled",
			EventVersion: 1, WallTimeNS: 2, OptimizerStep: &step,
			Payload: json.RawMessage(`{"name":"loss","value":1,"unit":"loss","step_domain":"optimizer_step","step":1,"sample_weight":1,"labels":{}}`),
		}},
	}
	if _, found, err := ProjectTelemetrySnapshot(
		context.Background(), reader, "run-1", 0, 16,
	); err == nil || !found {
		t.Fatalf("journal replacement during event reads was not rejected: found=%v err=%v", found, err)
	}
}
func (*telemetryFixtureReader) Runs(context.Context) ([]Run, error) { return nil, nil }
func (r *telemetryFixtureReader) Run(_ context.Context, runID string) (Run, bool, error) {
	return r.run, r.run.RunID == runID, nil
}
func (r *telemetryFixtureReader) CompiledPlan(_ context.Context, runID string) (CompiledPlanView, bool, error) {
	return r.plan, r.plan.RunID == runID, nil
}
func (r *telemetryFixtureReader) Events(_ context.Context, query EventQuery) ([]Event, error) {
	r.query = query
	r.queries = append(r.queries, query)
	types := make(map[string]struct{}, len(query.EventTypes))
	for _, eventType := range query.EventTypes {
		types[eventType] = struct{}{}
	}
	result := make([]Event, 0, len(r.events))
	for _, event := range r.events {
		_, typeMatches := types[event.EventType]
		if event.RunID == query.RunID && event.Sequence > query.After &&
			(query.Through == 0 || event.Sequence <= query.Through) &&
			(len(types) == 0 || typeMatches) {
			result = append(result, event)
		}
	}
	if query.NewestPerMetricSeries {
		sort.Slice(result, func(left, right int) bool {
			return result[left].Sequence > result[right].Sequence
		})
		latest := make([]Event, 0, len(result))
		seen := map[string]struct{}{}
		for _, event := range result {
			var payload struct {
				Name   string            `json:"name"`
				Labels map[string]string `json:"labels"`
			}
			if err := json.Unmarshal(event.Payload, &payload); err != nil {
				return nil, err
			}
			labels, err := json.Marshal(payload.Labels)
			if err != nil {
				return nil, err
			}
			key := event.NodeID + "\x00" + event.AttemptID + "\x00" + payload.Name + "\x00" + string(labels)
			if _, duplicate := seen[key]; duplicate {
				continue
			}
			seen[key] = struct{}{}
			latest = append(latest, event)
		}
		result = latest
	}
	sort.Slice(result, func(left, right int) bool {
		if query.NewestFirst {
			return result[left].Sequence > result[right].Sequence
		}
		return result[left].Sequence < result[right].Sequence
	})
	if query.Limit > 0 && len(result) > query.Limit {
		result = result[:query.Limit]
	}
	return result, nil
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

func TestRecentArtifactsKeepsBoundedTailSemanticsAndReportsTruncationThroughFence(t *testing.T) {
	events := make([]Event, 0, 3)
	for sequence := 1; sequence <= 3; sequence++ {
		events = append(events, Event{
			Sequence: uint64(sequence), EventID: fmt.Sprintf("artifact-%d", sequence),
			RunID: "run-1", NodeID: "eval", AttemptID: "eval@1",
			WorkerSequence: uint64(sequence), EventType: "artifact.published",
			EventVersion: 1, WallTimeNS: int64(sequence),
			Payload: json.RawMessage(fmt.Sprintf(`{
				"artifact_id":"gallery-%d","logical_name":"eval_gallery","kind":"image_gallery",
				"schema":"rwkv-lab.eval-gallery.v2","uri":"file:///sealed/gallery-%d",
				"size_bytes":1,"fingerprint_algorithm":"sha256",
				"fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				"complete":true,"producer_node_id":"eval","producer_attempt_id":"eval@1",
				"parent_artifact_ids":[],"published_at_ns":%d
			}`, sequence, sequence, sequence)),
		})
	}
	reader := &telemetryFixtureReader{
		run:    Run{RunID: "run-1", LastEventSeq: 3},
		events: events,
	}
	tail, found, err := RecentArtifacts(context.Background(), reader, "run-1", 2)
	if err != nil || !found || len(tail) != 2 ||
		tail[0].ArtifactID != "gallery-3" || tail[1].ArtifactID != "gallery-2" {
		t.Fatalf("bounded recent tail changed semantics: found=%t tail=%#v err=%v", found, tail, err)
	}
	fenced, truncated, err := RecentArtifactsThrough(context.Background(), reader, "run-1", 3, 2)
	if err != nil || !truncated || len(fenced) != 2 || fenced[0].Sequence != 3 || fenced[1].Sequence != 2 {
		t.Fatalf("fenced recent tail did not expose truncation: truncated=%t tail=%#v err=%v",
			truncated, fenced, err)
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

func compiledTelemetryFixture(metrics string) CompiledPlanView {
	return CompiledPlanView{
		JournalID: "0123456789abcdef0123456789abcdef",
		RunID:     "run-1", RunRevision: 3, PlanHash: "plan-hash",
		CanonicalPlan: json.RawMessage(`{
			"api_version":"training.local/v1","kind":"Experiment","metadata":{"name":"generic"},
			"spec":{"adapter":{"name":"any-family"},"observability":{
				"heartbeat_seconds":10,"metrics":` + metrics + `,
				"retain_raw_metrics_days":30,"eval_gallery_artifact":"eval/gallery",
				"log_artifact":"train/log"
			}}
		}`),
	}
}

func telemetrySnapshotFixture(events []Event, target uint64, metrics string) *telemetryFixtureReader {
	plan := compiledTelemetryFixture(metrics)
	return &telemetryFixtureReader{
		journalID: plan.JournalID,
		run: Run{RunID: "run-1", ExperimentName: "generic", PlanHash: plan.PlanHash,
			RunRevision: plan.RunRevision, LastEventSeq: target},
		plan: plan, events: events,
	}
}

func optimizerMetricEvent(sequence, step uint64) Event {
	return Event{
		Sequence: sequence, EventID: "metric", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", WorkerSequence: sequence, EventType: "metric.sampled",
		WallTimeNS: int64(sequence * 1_000), OptimizerStep: &step,
		Payload: json.RawMessage(`{
			"name":"train.loss","value":1.25,"unit":"loss","step_domain":"optimizer_step",
			"step":` + strconv.FormatUint(step, 10) + `,"sample_weight":1,"labels":{"route":"animation"}
		}`),
	}
}

func TestObservabilityFromCompiledPlanIsFamilyNeutralAndStrict(t *testing.T) {
	plan := compiledTelemetryFixture(`[
		{"name":"train.loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean","description":"objective"},
		{"name":"train.mode","type":"gauge","unit":"state","step_domain":"wall_time","aggregation":"last"}
	]`)
	declaration, err := ObservabilityFromCompiledPlan(plan)
	if err != nil || declaration.HeartbeatSeconds != 10 || len(declaration.Metrics) != 2 ||
		declaration.Metrics[0].Name != "train.loss" ||
		declaration.Metrics[1].StepDomain != "wall_time" ||
		declaration.EvalGalleryArtifact != "eval/gallery" {
		t.Fatalf("unexpected generic observability declaration: %#v err=%v", declaration, err)
	}

	malformed := plan
	malformed.CanonicalPlan = json.RawMessage(`{"spec":{"observability":{
		"heartbeat_seconds":10,"metrics":[
			{"name":"loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean","family":"rwkv"}
		],"retain_raw_metrics_days":30}}}`)
	if _, err := ObservabilityFromCompiledPlan(malformed); err == nil {
		t.Fatal("unknown family-specific metric field unexpectedly accepted")
	}

	duplicate := compiledTelemetryFixture(`[
		{"name":"loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean"},
		{"name":"loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean"}
	]`)
	if _, err := ObservabilityFromCompiledPlan(duplicate); err == nil {
		t.Fatal("duplicate metric declaration unexpectedly accepted")
	}
}

func TestTelemetrySnapshotColdLoadsOneCoherentTail(t *testing.T) {
	step := uint64(4)
	events := []Event{
		{Sequence: 2, EventID: "heartbeat", RunID: "run-1", NodeID: "train",
			AttemptID: "train@1", WorkerSequence: 2, EventType: "worker.heartbeat",
			WallTimeNS: 2_000, OptimizerStep: &step,
			Payload: json.RawMessage(`{"phase":"train","observed_at_ns":1900}`)},
		optimizerMetricEvent(4, step),
		{Sequence: 7, EventID: "artifact", RunID: "run-1", NodeID: "eval",
			AttemptID: "eval@1", WorkerSequence: 7, EventType: "artifact.published",
			WallTimeNS: 7_000, Payload: json.RawMessage(`{
				"artifact_id":"gallery-4","logical_name":"eval/gallery","kind":"image_gallery",
				"schema":"trainvm.eval-gallery/v1","uri":"file:///gallery-4","size_bytes":12,
				"fingerprint_algorithm":"sha256","fingerprint":"abc","complete":true,
				"producer_node_id":"eval","producer_attempt_id":"eval@1",
				"parent_artifact_ids":[],"published_at_ns":7000
			}`)},
	}
	reader := telemetrySnapshotFixture(events, 9, `[
		{"name":"train.loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"weighted_mean","description":"objective"}
	]`)
	snapshot, found, err := ProjectTelemetrySnapshot(context.Background(), reader, "run-1", 0, 10)
	if err != nil || !found || !snapshot.CaughtUp || snapshot.ReplayPending ||
		snapshot.TargetSequence != 9 || snapshot.NextSequence != 9 ||
		len(snapshot.Heartbeats) != 1 || len(snapshot.Metrics) != 1 ||
		len(snapshot.Artifacts) != 1 || snapshot.Metrics[0].Aggregation != "weighted_mean" ||
		snapshot.Metrics[0].Description != "objective" {
		t.Fatalf("unexpected telemetry tail: %#v found=%v err=%v", snapshot, found, err)
	}
	if len(reader.queries) != 3 || !reader.queries[0].NewestFirst ||
		reader.queries[1].NewestFirst || !reader.queries[1].NewestPerMetricSeries ||
		!reader.queries[2].NewestFirst ||
		reader.queries[0].Through != 9 || reader.queries[0].Limit != 1 ||
		reader.queries[1].Limit != maximumLiveMetricSeries+1 || reader.queries[2].Limit != 10 {
		t.Fatalf("cold load did not use independently bounded upper-fenced tails: %#v", reader.queries)
	}
}

func TestTelemetryColdLoadRepresentsEveryDeclaredMetricBeyondOldBrowserCap(t *testing.T) {
	const metricCount = 256
	metrics := make([]map[string]string, 0, metricCount)
	events := make([]Event, 0, metricCount)
	for index := 0; index < metricCount; index++ {
		name := "metric." + strconv.Itoa(index)
		metrics = append(metrics, map[string]string{
			"name": name, "type": "gauge", "unit": "value",
			"step_domain": "wall_time", "aggregation": "last",
		})
		events = append(events, Event{
			Sequence: uint64(index + 1), EventID: "metric-" + strconv.Itoa(index),
			RunID: "run-1", NodeID: "train", AttemptID: "train@1",
			WorkerSequence: uint64(index + 1), EventType: "metric.sampled",
			WallTimeNS: int64(index + 1), Payload: json.RawMessage(fmt.Sprintf(
				`{"name":%q,"value":%d,"unit":"value","step_domain":"wall_time","step":%d,"sample_weight":1,"labels":{}}`,
				name, index, index)),
		})
	}
	declaration, err := json.Marshal(metrics)
	if err != nil {
		t.Fatal(err)
	}
	reader := telemetrySnapshotFixture(events, metricCount, string(declaration))
	snapshot, found, err := ProjectTelemetrySnapshot(context.Background(), reader, "run-1", 0, 10)
	if err != nil || !found || len(snapshot.Metrics) != metricCount {
		t.Fatalf("cold load lost declared metrics: count=%d found=%v err=%v", len(snapshot.Metrics), found, err)
	}
}

func TestTelemetryColdLoadRejectsMoreThanBoundedMetricSeries(t *testing.T) {
	events := make([]Event, 0, maximumLiveMetricSeries+1)
	for index := 0; index <= maximumLiveMetricSeries; index++ {
		events = append(events, Event{
			Sequence: uint64(index + 1), EventID: "metric-" + strconv.Itoa(index),
			RunID: "run-1", NodeID: "train", AttemptID: "train@1",
			WorkerSequence: uint64(index + 1), EventType: "metric.sampled",
			WallTimeNS: int64(index + 1), Payload: json.RawMessage(fmt.Sprintf(
				`{"name":"metric.routes","value":%d,"unit":"value","step_domain":"wall_time","step":%d,"sample_weight":1,"labels":{"route":%q}}`,
				index, index, strconv.Itoa(index))),
		})
	}
	reader := telemetrySnapshotFixture(events, maximumLiveMetricSeries+1, `[
		{"name":"metric.routes","type":"gauge","unit":"value","step_domain":"wall_time","aggregation":"last"}
	]`)
	if _, found, err := ProjectTelemetrySnapshot(
		context.Background(), reader, "run-1", 0, 10,
	); err == nil || !found {
		t.Fatalf("oversized live metric cardinality was accepted: found=%v err=%v", found, err)
	}
}

func TestTelemetrySnapshotIncrementalReplayUsesOneCursor(t *testing.T) {
	step := uint64(4)
	events := []Event{
		optimizerMetricEvent(4, step),
		{Sequence: 6, EventID: "heartbeat-6", RunID: "run-1", NodeID: "train",
			AttemptID: "train@1", WorkerSequence: 6, EventType: "worker.heartbeat",
			WallTimeNS: 6_000, OptimizerStep: &step,
			Payload: json.RawMessage(`{"phase":"train","observed_at_ns":5900}`)},
		optimizerMetricEvent(8, step),
	}
	metrics := `[{"name":"train.loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean"}]`
	reader := telemetrySnapshotFixture(events, 10, metrics)
	first, found, err := ProjectTelemetrySnapshot(context.Background(), reader, "run-1", 3, 2)
	if err != nil || !found || !first.ReplayPending || first.CaughtUp ||
		first.NextSequence != 6 || len(first.Metrics) != 1 || len(first.Heartbeats) != 1 ||
		reader.queries[0].Limit != 3 || reader.queries[0].NewestFirst {
		t.Fatalf("unexpected pending replay: %#v queries=%#v err=%v", first, reader.queries, err)
	}
	second, found, err := ProjectTelemetrySnapshot(
		context.Background(), reader, "run-1", first.NextSequence, 2)
	if err != nil || !found || second.ReplayPending || !second.CaughtUp ||
		second.NextSequence != 10 || len(second.Metrics) != 1 {
		t.Fatalf("unexpected caught-up replay: %#v err=%v", second, err)
	}
}

func TestMetricStepSemanticsFollowDeclaredDomain(t *testing.T) {
	event := Event{
		Sequence: 1, EventID: "tokens", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", WorkerSequence: 1, EventType: "metric.sampled",
		WallTimeNS: 1_000, Payload: json.RawMessage(`{
			"name":"train.mode","value":"warming","unit":"state","step_domain":"token",
			"step":2048,"sample_weight":1,"labels":{}
		}`),
	}
	point, err := metricFromEvent(event, &MetricDescriptor{
		Name: "train.mode", Type: "gauge", Unit: "state",
		StepDomain: "token", Aggregation: "last",
	})
	if err != nil || point.OptimizerStep != nil || point.Value != "warming" {
		t.Fatalf("non-optimizer metric was not preserved: %#v err=%v", point, err)
	}
	optimizerStep := uint64(1)
	event.OptimizerStep = &optimizerStep
	if _, err := metricFromEvent(event, nil); err == nil {
		t.Fatal("non-optimizer metric unexpectedly accepted optimizer-step metadata")
	}
	event.OptimizerStep = nil
	event.Payload = json.RawMessage(`{
		"name":"train.loss","value":true,"unit":"loss","step_domain":"optimizer_step",
		"step":1,"sample_weight":1,"labels":{}
	}`)
	if _, err := metricFromEvent(event, nil); err == nil {
		t.Fatal("optimizer-domain metric unexpectedly accepted without optimizer-step metadata")
	}
}

func TestHeartbeatProjectionIsStrict(t *testing.T) {
	step := uint64(8)
	base := Event{
		Sequence: 1, EventID: "heartbeat", RunID: "run-1", NodeID: "train",
		AttemptID: "train@1", WorkerSequence: 1, EventType: "worker.heartbeat",
		WallTimeNS: 1_000, OptimizerStep: &step,
		Payload: json.RawMessage(`{"phase":"train","observed_at_ns":900}`),
	}
	point, err := heartbeatFromEvent(base)
	if err != nil || point.Phase != "train" || point.OptimizerStep != 8 ||
		point.ObservedAtNS != 900 || point.AcceptedAtNS != 1_000 {
		t.Fatalf("valid heartbeat was not projected: %#v err=%v", point, err)
	}
	malformed := []Event{base, base, base}
	malformed[0].Payload = json.RawMessage(`{"phase":"train","observed_at_ns":900,"family":"rwkv"}`)
	malformed[1].OptimizerStep = nil
	malformed[2].Payload = json.RawMessage(`{"phase":"","observed_at_ns":900}`)
	for index, event := range malformed {
		if _, err := heartbeatFromEvent(event); err == nil {
			t.Fatalf("malformed heartbeat %d unexpectedly accepted", index)
		}
	}
}

func TestTelemetrySnapshotRejectsUndeclaredMetricsAndStaleCursors(t *testing.T) {
	step := uint64(1)
	reader := telemetrySnapshotFixture([]Event{optimizerMetricEvent(2, step)}, 2, `[]`)
	if _, _, err := ProjectTelemetrySnapshot(context.Background(), reader, "run-1", 0, 10); err == nil {
		t.Fatal("undeclared metric unexpectedly crossed the telemetry projection")
	}
	if _, _, err := ProjectTelemetrySnapshot(context.Background(), reader, "run-1", 3, 10); err == nil {
		t.Fatal("cursor beyond the captured run prefix unexpectedly accepted")
	}
}

func TestTelemetrySnapshotUsesOnePathAcrossRepresentativeFamiliesAndStatelessNodes(t *testing.T) {
	repository := filepath.Clean(filepath.Join("..", "..", ".."))
	cases := []struct {
		name, document, node string
	}{
		{"mageflow", "docs/experiment-vm/examples/mageflow-cache-resume.json", "train_to_boundary"},
		{"rwkv", "docs/experiment-vm/examples/coverage/scratch-rwkv-pretrain.json", "rwkv_scratch"},
		{"transformer", "docs/experiment-vm/examples/coverage/transformer-mla-continuation.json", "mla_continuation"},
		{"vision", "docs/experiment-vm/examples/coverage/vision-distillation-training.json", "vision_distillation"},
		{"stateless", "docs/experiment-vm/examples/mageflow-cache-resume.json", "prepare_cache_span"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			document, err := os.ReadFile(filepath.Join(repository, test.document))
			if err != nil {
				t.Fatal(err)
			}
			plan := CompiledPlanView{
				JournalID: "0123456789abcdef0123456789abcdef", RunID: "run-1",
				RunRevision: 1, PlanHash: "plan-" + test.name, CanonicalPlan: document,
			}
			declaration, err := ObservabilityFromCompiledPlan(plan)
			if err != nil || len(declaration.Metrics) == 0 {
				t.Fatalf("representative declaration failed: %#v err=%v", declaration, err)
			}
			descriptor := declaration.Metrics[0]
			payload, err := json.Marshal(map[string]any{
				"name": descriptor.Name, "value": 1.0, "unit": descriptor.Unit,
				"step_domain": descriptor.StepDomain, "step": uint64(7),
				"sample_weight": 1.0, "labels": map[string]string{"family": test.name},
			})
			if err != nil {
				t.Fatal(err)
			}
			var optimizerStep *uint64
			if descriptor.StepDomain == "optimizer_step" {
				step := uint64(7)
				optimizerStep = &step
			}
			event := Event{
				Sequence: 2, EventID: "metric-" + test.name, RunID: "run-1",
				NodeID: test.node, AttemptID: test.node + "@1", WorkerSequence: 1,
				EventType: "metric.sampled", EventVersion: 1, WallTimeNS: 2_000,
				OptimizerStep: optimizerStep, Payload: payload,
			}
			reader := &telemetryFixtureReader{
				journalID: plan.JournalID,
				run: Run{RunID: "run-1", PlanHash: plan.PlanHash, RunRevision: 1,
					CurrentNodeID: test.node, CurrentAttemptID: test.node + "@1", LastEventSeq: 2},
				plan: plan, events: []Event{event},
			}
			snapshot, found, err := ProjectTelemetrySnapshot(context.Background(), reader, "run-1", 0, 16)
			if err != nil || !found || !snapshot.CaughtUp || len(snapshot.Metrics) != 1 ||
				snapshot.Metrics[0].NodeID != test.node ||
				snapshot.Metrics[0].Type != descriptor.Type ||
				snapshot.Metrics[0].Labels["family"] != test.name {
				t.Fatalf("family-neutral projection failed: %#v found=%v err=%v", snapshot, found, err)
			}
		})
	}
}
