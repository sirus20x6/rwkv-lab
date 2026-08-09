package server

import (
	"encoding/json"
	"strings"
	"testing"

	trainvmstore "trainboard/internal/trainvm"
)

func evaluationMetric(name string, step uint64, value any) trainvmstore.MetricPoint {
	return trainvmstore.MetricPoint{
		Name: name, Step: step, Value: value, StepDomain: "optimizer_step",
	}
}

// The timeline must show a step-zero baseline, cheap interior probes, the
// generation milestones, and the final audit as four distinguishable things.
// Rendering them as one undifferentiated "eval" series is the failure this
// endpoint exists to prevent.
func TestEvaluationTimelineSeparatesEveryMilestoneKind(t *testing.T) {
	galleries := []trainVMGallerySummary{
		{
			ArtifactID: "gallery-0", Step: 0,
			CheckpointManifestDigest: "sha256:" + strings.Repeat("a", 64),
			EvaluatorProfileDigest:   "sha256:" + strings.Repeat("b", 64),
		},
		{
			ArtifactID: "gallery-250", Step: 250,
			CheckpointManifestDigest: "sha256:" + strings.Repeat("c", 64),
			EvaluatorProfileDigest:   "sha256:" + strings.Repeat("b", 64),
		},
	}
	metrics := []trainvmstore.MetricPoint{
		evaluationMetric("eval.planned_scalar_full_milestones", 0, float64(2)),
		evaluationMetric("eval.planned_scalar_probe_milestones", 0, float64(9)),
		evaluationMetric("eval.planned_qualitative_milestones", 0, float64(4)),
		evaluationMetric("eval.cadence_revision", 0, float64(0)),
		evaluationMetric("eval.loss", 0, 1.7319023985),
		evaluationMetric("eval.probe_loss", 75, 1.61),
		evaluationMetric("eval.probe_examples", 75, float64(64)),
		evaluationMetric("eval.probe_loss", 250, 1.42),
		evaluationMetric("eval.cadence_revision", 250, float64(1)),
		evaluationMetric("eval.loss", 745, 1.20),
		evaluationMetric("eval.test_loss", 745, 1.31),
		// A different step domain is a different timeline and must not leak in.
		{Name: "eval.loss", Step: 999, Value: 9.9, StepDomain: "sample"},
	}

	timeline := buildEvaluationTimeline(galleries, metrics, false)

	steps := make([]uint64, 0, len(timeline.Milestones))
	kinds := map[uint64][]string{}
	for _, milestone := range timeline.Milestones {
		steps = append(steps, milestone.Step)
		kinds[milestone.Step] = milestone.Kinds
	}
	if want := []uint64{0, 75, 250, 745}; !equalSteps(steps, want) {
		t.Fatalf("milestone steps = %v, want %v", steps, want)
	}
	if got := kinds[0]; !equalStrings(got, []string{"qualitative", "scalar_full"}) {
		t.Fatalf("step-zero kinds = %v", got)
	}
	if got := kinds[75]; !equalStrings(got, []string{"scalar_probe"}) {
		t.Fatalf("interior probe kinds = %v", got)
	}
	if got := kinds[250]; !equalStrings(got, []string{"qualitative", "scalar_probe"}) {
		t.Fatalf("step 250 kinds = %v", got)
	}
	if got := kinds[745]; !equalStrings(got, []string{"final_audit", "scalar_full"}) {
		t.Fatalf("final kinds = %v", got)
	}

	baseline := timeline.Milestones[0]
	if baseline.ScalarLoss == nil || *baseline.ScalarLoss != 1.7319023985 {
		t.Fatalf("step-zero baseline loss missing: %+v", baseline)
	}
	// Every rendered milestone is anchored: a gallery names the checkpoint it
	// was generated from and the frozen evaluation manifest it was read under.
	if baseline.CheckpointManifestDigest == "" || baseline.EvalManifestDigest == "" {
		t.Fatalf("step-zero milestone is not anchored: %+v", baseline)
	}
	if timeline.Milestones[1].ProbeExamples == nil ||
		*timeline.Milestones[1].ProbeExamples != 64 {
		t.Fatalf("probe example budget missing: %+v", timeline.Milestones[1])
	}
	if timeline.Milestones[1].ScalarLoss != nil {
		t.Fatalf("a probe must never be reported as a full scalar read")
	}
	if timeline.CadenceRevision != 1 {
		t.Fatalf("cadence revision = %d, want the latest revision 1", timeline.CadenceRevision)
	}
	if timeline.Planned["scalar_probe"] != 9 || timeline.Planned["qualitative"] != 4 {
		t.Fatalf("planned milestone counts = %v", timeline.Planned)
	}
	if timeline.Observed["scalar_probe"] != 2 || timeline.Observed["qualitative"] != 2 {
		t.Fatalf("observed milestone counts = %v", timeline.Observed)
	}

	encoded, err := json.Marshal(timeline)
	if err != nil {
		t.Fatalf("timeline does not encode: %v", err)
	}
	for _, want := range []string{
		`"step":0`, `"scalar_full"`, `"scalar_probe"`, `"qualitative"`, `"final_audit"`,
		`"eval_manifest_digest"`, `"cadence_revision":1`,
	} {
		if !strings.Contains(string(encoded), want) {
			t.Fatalf("timeline JSON is missing %s: %s", want, encoded)
		}
	}
}

func TestEvaluationTimelineOmitsUnreadableAndEmptyMilestones(t *testing.T) {
	metrics := []trainvmstore.MetricPoint{
		// Non-numeric scalars are absent, not zero.
		evaluationMetric("eval.loss", 10, "not-a-number"),
		// A bare probe-budget sample carries no milestone of its own.
		evaluationMetric("eval.probe_examples", 20, float64(8)),
		evaluationMetric("eval.probe_loss", 30, json.Number("0.5")),
	}
	timeline := buildEvaluationTimeline(nil, metrics, true)
	if len(timeline.Milestones) != 1 || timeline.Milestones[0].Step != 30 {
		t.Fatalf("milestones = %+v", timeline.Milestones)
	}
	if timeline.Milestones[0].ProbeLoss == nil || *timeline.Milestones[0].ProbeLoss != 0.5 {
		t.Fatalf("json.Number probe loss was not decoded: %+v", timeline.Milestones[0])
	}
	if !timeline.HistoryTruncated {
		t.Fatalf("truncation must survive into the rendered timeline")
	}
}

func equalSteps(got, want []uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if got[index] != want[index] {
			return false
		}
	}
	return true
}
