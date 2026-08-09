package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"

	trainvmstore "trainboard/internal/trainvm"
)

// The eval-examples artifact is the universal pre-mutation evidence: it is what
// the controller consults before it will record an optimizer step past an
// attempt's baseline. Until now the dashboard ignored the kind entirely, so a
// run could be gated — or fail to be gated — with nothing on screen either way.
// It is folded into the evaluation timeline rather than given its own page,
// because its whole meaning is "this milestone was paid for before training
// moved", which is only legible beside the milestone.
const (
	trainVMEvalExamplesSchema     = "rwkv-lab.eval-examples.v1"
	trainVMEvalExamplesMaxBytes   = 64 << 10
	trainVMEvalExamplesMaxRecords = 256
)

type trainVMEvalExamplesManifest struct {
	APIVersion    string `json:"api_version"`
	RunID         string `json:"run_id"`
	NodeID        string `json:"node_id"`
	AttemptID     string `json:"attempt_id"`
	OptimizerStep uint64 `json:"optimizer_step"`
	StepDomain    string `json:"step_domain"`
	SeriesID      string `json:"series_id"`
	Heldout       struct {
		IdentityField    string `json:"identity_field"`
		IdentitiesDigest string `json:"identities_digest"`
		SelectorDigest   string `json:"selector_digest"`
	} `json:"heldout"`
	Evaluator struct {
		ComponentDigest string   `json:"component_digest"`
		MetricNames     []string `json:"metric_names"`
	} `json:"evaluator"`
	Checkpoint struct {
		ArtifactID     string `json:"artifact_id"`
		ManifestDigest string `json:"manifest_digest"`
	} `json:"checkpoint"`
	PolicyDigest            string           `json:"policy_digest"`
	Examples                []map[string]any `json:"examples"`
	CanonicalManifestDigest string           `json:"canonical_manifest_digest"`
}

type trainVMEvalExamplesSummary struct {
	ArtifactID               string   `json:"artifact_id"`
	LogicalName              string   `json:"logical_name"`
	Step                     uint64   `json:"step"`
	SeriesID                 string   `json:"series_id"`
	NodeID                   string   `json:"node_id"`
	AttemptID                string   `json:"attempt_id"`
	ExampleCount             int      `json:"example_count"`
	MetricNames              []string `json:"metric_names"`
	CheckpointArtifactID     string   `json:"checkpoint_artifact_id"`
	CheckpointManifestDigest string   `json:"checkpoint_manifest_digest"`
	PolicyDigest             string   `json:"policy_digest"`
	PublishedAtNS            int64    `json:"published_at_ns"`
}

// loadEvalExamples verifies the manifest against the published artifact's own
// identity before believing a byte of it: declared size, declared fingerprint,
// the canonical self-digest, and producer node/attempt. A manifest that only
// parses is not evidence.
func (s *Server) loadEvalExamples(
	artifact trainvmstore.PublishedArtifact,
) (trainVMEvalExamplesManifest, error) {
	var manifest trainVMEvalExamplesManifest
	manifestPath, err := fileURIPath(artifact.URI)
	if err != nil {
		return manifest, err
	}
	file, err := s.openGalleryFile(manifestPath)
	if err != nil {
		return manifest, fmt.Errorf("open eval-examples manifest: %w", err)
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, trainVMEvalExamplesMaxBytes+1))
	if err != nil {
		return manifest, fmt.Errorf("read eval-examples manifest: %w", err)
	}
	if len(data) > trainVMEvalExamplesMaxBytes {
		return manifest, fmt.Errorf("eval-examples manifest exceeds %d-byte limit",
			trainVMEvalExamplesMaxBytes)
	}
	if uint64(len(data)) != artifact.SizeBytes {
		return manifest, fmt.Errorf("published eval-examples manifest size mismatch")
	}
	expected, ok := normalizedSHA256(artifact.Fingerprint)
	actual := sha256.Sum256(data)
	if artifact.FingerprintAlgorithm != "manifest_sha256" || !ok ||
		expected != hex.EncodeToString(actual[:]) {
		return manifest, fmt.Errorf("published eval-examples manifest fingerprint mismatch")
	}
	if err := strictJSONDocument(data, &manifest); err != nil {
		return manifest, fmt.Errorf("decode eval-examples manifest: %w", err)
	}
	canonical, canonicalErr := canonicalDigestWithoutField(data, "canonical_manifest_digest")
	declaredCanonical, canonicalOK := normalizedSHA256(manifest.CanonicalManifestDigest)
	if !validGalleryArtifactID(artifact.ArtifactID) ||
		manifest.APIVersion != trainVMEvalExamplesSchema ||
		manifest.RunID != artifact.RunID ||
		manifest.NodeID != artifact.ProducerNodeID ||
		manifest.AttemptID != artifact.ProducerAttemptID ||
		manifest.StepDomain != "optimizer_step" ||
		!validGalleryText(manifest.SeriesID) ||
		!validGalleryDigest(manifest.Checkpoint.ManifestDigest) ||
		!validGalleryDigest(manifest.Evaluator.ComponentDigest) ||
		!validGalleryDigest(manifest.PolicyDigest) ||
		!validGalleryArtifactID(manifest.Checkpoint.ArtifactID) ||
		len(manifest.Evaluator.MetricNames) == 0 ||
		canonicalErr != nil || !canonicalOK || canonical != declaredCanonical ||
		len(manifest.Examples) == 0 ||
		len(manifest.Examples) > trainVMEvalExamplesMaxRecords {
		return trainVMEvalExamplesManifest{},
			fmt.Errorf("eval-examples manifest identity or cardinality is invalid")
	}
	return manifest, nil
}

func evalExamplesSummary(
	artifact trainvmstore.PublishedArtifact, manifest trainVMEvalExamplesManifest,
) trainVMEvalExamplesSummary {
	return trainVMEvalExamplesSummary{
		ArtifactID:               artifact.ArtifactID,
		LogicalName:              artifact.LogicalName,
		Step:                     manifest.OptimizerStep,
		SeriesID:                 manifest.SeriesID,
		NodeID:                   manifest.NodeID,
		AttemptID:                manifest.AttemptID,
		ExampleCount:             len(manifest.Examples),
		MetricNames:              manifest.Evaluator.MetricNames,
		CheckpointArtifactID:     manifest.Checkpoint.ArtifactID,
		CheckpointManifestDigest: manifest.Checkpoint.ManifestDigest,
		PolicyDigest:             manifest.PolicyDigest,
		PublishedAtNS:            artifact.PublishedAtNS,
	}
}

// publishedEvalExamples selects by artifact kind and schema rather than by a
// declared logical name: the plan's observability block names the gallery
// artifact but has no field for this one, and inventing a second hand-maintained
// name is exactly the drift this card exists to fix.
func (s *Server) publishedEvalExamples(
	ctx context.Context, runID string,
) ([]trainVMEvalExamplesSummary, bool, error) {
	journalID, run, _, found, err := trainvmstore.CaptureRunPlanPrefix(ctx, s.trainvm, runID)
	if err != nil {
		return nil, false, err
	}
	if !found {
		return nil, false, os.ErrNotExist
	}
	artifacts, truncated, err := trainvmstore.RecentArtifactsThrough(
		ctx, s.trainvm, runID, run.LastEventSeq, trainVMGalleryMaxEvents,
	)
	if err != nil {
		return nil, false, err
	}
	confirmedJournalID, err := s.trainvm.JournalID(ctx)
	if err != nil {
		return nil, false, err
	}
	if confirmedJournalID != journalID {
		return nil, false, fmt.Errorf(
			"TrainVM journal changed while eval-examples history was being captured")
	}
	result := make([]trainVMEvalExamplesSummary, 0)
	seen := map[string]struct{}{}
	for _, artifact := range artifacts {
		if artifact.Kind != "eval_examples" || artifact.Schema != trainVMEvalExamplesSchema ||
			!artifact.Complete {
			continue
		}
		if _, duplicate := seen[artifact.ArtifactID]; duplicate {
			return nil, false, fmt.Errorf(
				"duplicate published eval-examples artifact ID %q", artifact.ArtifactID)
		}
		seen[artifact.ArtifactID] = struct{}{}
		if len(result) >= trainVMGalleryMaxRevisions {
			truncated = true
			continue
		}
		manifest, loadErr := s.loadEvalExamples(artifact)
		if loadErr != nil {
			// One unreadable revision must not blank the timeline; the milestone
			// simply loses its examples annotation and keeps its other evidence.
			truncated = true
			continue
		}
		result = append(result, evalExamplesSummary(artifact, manifest))
	}
	return result, truncated, nil
}
