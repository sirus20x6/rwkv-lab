package trainvm

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
)

const maximumCapsuleControlEvents = 1_000

var capsuleControlEventTypes = []string{
	"control.applied",
	"control.rejected",
	"control.requested",
	"control.restart_required",
}

// ReproducibilityCapsuleBody is an integrity-bound, browser-safe view of one
// durable run prefix. The canonical plan carries adapter, component, code,
// runtime and static-input identities; control events carry every live
// trajectory change; metrics and artifact manifests carry bounded outcomes.
type ReproducibilityCapsuleBody struct {
	APIVersion               string                       `json:"api_version"`
	JournalID                string                       `json:"journal_id"`
	ThroughSequence          uint64                       `json:"through_sequence"`
	Run                      Run                          `json:"run"`
	PlanHash                 string                       `json:"plan_hash"`
	CanonicalPlan            json.RawMessage              `json:"canonical_plan"`
	Observability            ObservabilityDeclaration     `json:"observability"`
	LatestMetrics            []MetricPoint                `json:"latest_metrics"`
	Artifacts                []ObservableArtifact         `json:"artifacts"`
	ArtifactHistoryTruncated bool                         `json:"artifact_history_truncated"`
	ExecutionPhases          []ExecutionPhaseReceiptPoint `json:"execution_phases"`
	ControlEvents            []Event                      `json:"control_events"`
}

type ReproducibilityCapsule struct {
	Body          ReproducibilityCapsuleBody `json:"body"`
	CapsuleDigest string                     `json:"capsule_digest"`
}

// BuildReproducibilityCapsule captures a coherent, immutable journal prefix.
// It never reads worker paths or the legacy dashboard database and never
// mutates the run. A run may advance after capture; ThroughSequence keeps the
// resulting document an honest snapshot rather than a claim about later state.
func BuildReproducibilityCapsule(
	ctx context.Context, reader ReadModel, runID string,
) (ReproducibilityCapsule, bool, error) {
	snapshot, found, err := ProjectTelemetrySnapshot(ctx, reader, runID, 0, 1_000)
	if err != nil || !found {
		return ReproducibilityCapsule{}, found, err
	}
	plan, found, err := reader.CompiledPlan(ctx, runID)
	if err != nil || !found {
		return ReproducibilityCapsule{}, found, err
	}
	if plan.JournalID != snapshot.JournalID || plan.RunID != runID ||
		plan.PlanHash != snapshot.Run.PlanHash {
		return ReproducibilityCapsule{}, true,
			fmt.Errorf("TrainVM run changed while its reproducibility capsule was being captured")
	}
	controls, err := capsuleControlsThrough(
		ctx, reader, runID, snapshot.TargetSequence,
	)
	if err != nil {
		return ReproducibilityCapsule{}, true, err
	}
	published, artifactHistoryTruncated, err := RecentArtifactsThrough(
		ctx, reader, runID, snapshot.TargetSequence, 1_000,
	)
	if err != nil {
		return ReproducibilityCapsule{}, true, err
	}
	artifacts := make([]ObservableArtifact, len(published))
	for index := range published {
		// RecentArtifactsThrough is newest-first. Capsules are canonicalized in
		// journal order so repeated capture of one prefix has identical bytes.
		artifacts[len(published)-1-index] = RedactArtifact(published[index])
	}
	body := ReproducibilityCapsuleBody{
		APIVersion:               "trainvm.reproducibility-capsule/v1",
		JournalID:                snapshot.JournalID,
		ThroughSequence:          snapshot.TargetSequence,
		Run:                      snapshot.Run,
		PlanHash:                 plan.PlanHash,
		CanonicalPlan:            append(json.RawMessage(nil), plan.CanonicalPlan...),
		Observability:            snapshot.Observability,
		LatestMetrics:            append([]MetricPoint(nil), snapshot.Metrics...),
		Artifacts:                artifacts,
		ArtifactHistoryTruncated: artifactHistoryTruncated,
		ExecutionPhases:          append([]ExecutionPhaseReceiptPoint(nil), snapshot.ExecutionPhases...),
		ControlEvents:            controls,
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return ReproducibilityCapsule{}, true,
			fmt.Errorf("encode TrainVM reproducibility capsule: %w", err)
	}
	confirmedJournalID, err := reader.JournalID(ctx)
	if err != nil {
		return ReproducibilityCapsule{}, true, err
	}
	if confirmedJournalID != snapshot.JournalID {
		return ReproducibilityCapsule{}, true,
			fmt.Errorf("TrainVM journal changed while its reproducibility capsule was being captured")
	}
	return ReproducibilityCapsule{
		Body:          body,
		CapsuleDigest: fmt.Sprintf("sha256:%x", sha256.Sum256(encoded)),
	}, true, nil
}

func capsuleControlsThrough(
	ctx context.Context, reader ReadModel, runID string, through uint64,
) ([]Event, error) {
	if through == 0 {
		return []Event{}, nil
	}
	events, err := reader.Events(ctx, EventQuery{
		RunID: runID, After: 0, Through: through,
		Limit:       maximumCapsuleControlEvents,
		EventTypes:  append([]string(nil), capsuleControlEventTypes...),
		NewestFirst: true,
	})
	if err != nil {
		return nil, err
	}
	if len(events) > maximumCapsuleControlEvents {
		return nil, fmt.Errorf("TrainVM authority exceeded the bounded capsule control history")
	}
	var previous uint64
	for index, event := range events {
		if event.RunID != runID || event.Sequence == 0 || event.Sequence > through ||
			(index > 0 && event.Sequence >= previous) {
			return nil, fmt.Errorf("TrainVM authority returned an incoherent capsule control history")
		}
		previous = event.Sequence
	}
	if len(events) == maximumCapsuleControlEvents && events[len(events)-1].Sequence > 1 {
		older, olderErr := reader.Events(ctx, EventQuery{
			RunID: runID, After: 0, Through: events[len(events)-1].Sequence - 1,
			Limit: 1, EventTypes: append([]string(nil), capsuleControlEventTypes...),
			NewestFirst: true,
		})
		if olderErr != nil {
			return nil, olderErr
		}
		if len(older) != 0 {
			return nil, fmt.Errorf(
				"TrainVM run exceeds the %d-event reproducibility control bound",
				maximumCapsuleControlEvents,
			)
		}
	}
	for left, right := 0, len(events)-1; left < right; left, right = left+1, right-1 {
		events[left], events[right] = events[right], events[left]
	}
	return events, nil
}
