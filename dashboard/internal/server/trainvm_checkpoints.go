package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	trainvmstore "trainboard/internal/trainvm"
)

const (
	trainVMCheckpointEnvelopeSchema = "trainvm.checkpoint-snapshot.v1"
	trainVMCheckpointManifestMax    = 64 << 20
	trainVMCheckpointMaxFiles       = 131_072
	trainVMCheckpointDefaultTail    = 250
	trainVMCheckpointMaximumTail    = 1_000
)

type trainVMCheckpointProducer struct {
	RunID     string `json:"run_id"`
	NodeID    string `json:"node_id"`
	AttemptID string `json:"attempt_id"`
}

type trainVMCheckpointObject struct {
	RelativePath string `json:"relative_path"`
	SHA256       string `json:"sha256"`
	SizeBytes    uint64 `json:"size_bytes"`
}

type trainVMCheckpointManifest struct {
	Schema              string                    `json:"schema"`
	CheckpointSchema    string                    `json:"checkpoint_schema"`
	Producer            trainVMCheckpointProducer `json:"producer"`
	OptimizerStep       uint64                    `json:"optimizer_step"`
	ResumeGrade         string                    `json:"resume_grade"`
	StateComponents     []string                  `json:"state_components"`
	PayloadDirectory    string                    `json:"payload_directory"`
	FileCount           uint64                    `json:"file_count"`
	PayloadSizeBytes    uint64                    `json:"payload_size_bytes"`
	Objects             []trainVMCheckpointObject `json:"objects"`
	ParentArtifactIDs   []string                  `json:"parent_artifact_ids"`
	CanonicalTreeDigest string                    `json:"canonical_tree_digest"`
}

type trainVMCheckpointSummary struct {
	Valid               bool     `json:"valid"`
	ValidationError     string   `json:"validation_error,omitempty"`
	Sequence            uint64   `json:"sequence"`
	ArtifactID          string   `json:"artifact_id"`
	LogicalName         string   `json:"logical_name"`
	CheckpointSchema    string   `json:"checkpoint_schema"`
	NodeID              string   `json:"node_id"`
	AttemptID           string   `json:"attempt_id"`
	OptimizerStep       uint64   `json:"optimizer_step"`
	ResumeGrade         string   `json:"resume_grade"`
	StateComponents     []string `json:"state_components"`
	FileCount           uint64   `json:"file_count"`
	PayloadSizeBytes    uint64   `json:"payload_size_bytes"`
	ParentArtifactIDs   []string `json:"parent_artifact_ids"`
	CanonicalTreeDigest string   `json:"canonical_tree_digest"`
	PublishedAtNS       int64    `json:"published_at_ns"`
}

func (s *Server) publishedCheckpoints(
	ctx context.Context, runID string, limit int,
) ([]trainvmstore.PublishedArtifact, bool, error) {
	var result []trainvmstore.PublishedArtifact
	seen := map[string]struct{}{}
	page, found, err := trainvmstore.RecentArtifacts(ctx, s.trainvm, runID, limit)
	if err != nil || !found {
		return nil, found, err
	}
	for _, artifact := range page {
		if artifact.Kind != "checkpoint" {
			continue
		}
		if _, duplicate := seen[artifact.ArtifactID]; duplicate {
			return nil, true, fmt.Errorf("duplicate published checkpoint artifact ID %q", artifact.ArtifactID)
		}
		seen[artifact.ArtifactID] = struct{}{}
		result = append(result, artifact)
	}
	return result, true, nil
}

func validCheckpointRelativePath(value string) bool {
	if value == "" || len(value) > 4096 || filepath.IsAbs(value) || strings.ContainsAny(value, "\x00\r\n") {
		return false
	}
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(value)))
	return clean == value && clean != "." && clean != ".." && !strings.HasPrefix(clean, "../")
}

func validCheckpointStateComponent(value string) bool {
	switch value {
	case "component_composition", "control_revision", "curriculum", "data_cursor",
		"expert_routing", "gradient_scaler", "lr_schedule", "model", "optimizer",
		"optimizer_groups", "plateau_state",
		"parameter_routing", "rng_accelerator", "rng_numpy", "rng_python", "rng_torch",
		"topology", "weight_decay_schedule":
		return true
	default:
		return false
	}
}

func checkpointCanonicalDigest(data []byte) (string, error) {
	var document map[string]json.RawMessage
	if err := strictJSONDocument(data, &document); err != nil {
		return "", err
	}
	if _, present := document["canonical_tree_digest"]; !present {
		return "", fmt.Errorf("checkpoint canonical tree digest is missing")
	}
	delete(document, "canonical_tree_digest")
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(document); err != nil {
		return "", err
	}
	body := bytes.TrimSuffix(canonical.Bytes(), []byte{'\n'})
	digest := sha256.Sum256(body)
	return hex.EncodeToString(digest[:]), nil
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func (s *Server) loadCheckpoint(artifact trainvmstore.PublishedArtifact) (trainVMCheckpointManifest, error) {
	var manifest trainVMCheckpointManifest
	manifestPath, err := fileURIPath(artifact.URI)
	if err != nil {
		return manifest, err
	}
	resolved, ok := s.resolveEvalImage(manifestPath)
	if !ok {
		return manifest, fmt.Errorf("checkpoint manifest is outside allowed data roots")
	}
	data, err := readBoundedFile(resolved, trainVMCheckpointManifestMax)
	if err != nil {
		return manifest, fmt.Errorf("read checkpoint manifest: %w", err)
	}
	expected, digestOK := normalizedSHA256(artifact.Fingerprint)
	actual := sha256.Sum256(data)
	if artifact.FingerprintAlgorithm != "manifest_sha256" || !digestOK || expected != hex.EncodeToString(actual[:]) {
		return manifest, fmt.Errorf("published checkpoint manifest fingerprint mismatch")
	}
	if err := strictJSONDocument(data, &manifest); err != nil {
		return manifest, fmt.Errorf("decode checkpoint manifest: %w", err)
	}
	canonical, canonicalErr := checkpointCanonicalDigest(data)
	declaredCanonical, canonicalOK := normalizedSHA256(manifest.CanonicalTreeDigest)
	if canonicalErr != nil || !canonicalOK || canonical != declaredCanonical ||
		!validGalleryArtifactID(artifact.ArtifactID) || !validGalleryText(manifest.CheckpointSchema) ||
		!validGalleryText(manifest.Producer.RunID) || !validGalleryText(manifest.Producer.NodeID) ||
		!validGalleryText(manifest.Producer.AttemptID) ||
		manifest.Schema != trainVMCheckpointEnvelopeSchema || manifest.CheckpointSchema != artifact.Schema ||
		manifest.Producer.RunID != artifact.RunID || manifest.Producer.NodeID != artifact.ProducerNodeID ||
		manifest.Producer.AttemptID != artifact.ProducerAttemptID || manifest.PayloadDirectory != "payload" ||
		manifest.FileCount == 0 || manifest.FileCount > trainVMCheckpointMaxFiles ||
		manifest.FileCount != uint64(len(manifest.Objects)) || len(manifest.StateComponents) == 0 ||
		manifest.ParentArtifactIDs == nil || !equalStrings(manifest.ParentArtifactIDs, artifact.ParentArtifactIDs) ||
		(manifest.ResumeGrade != "terminal_checkpoint" && manifest.ResumeGrade != "compatible" && manifest.ResumeGrade != "exact") ||
		uint64(len(data)) > artifact.SizeBytes || artifact.SizeBytes-uint64(len(data)) != manifest.PayloadSizeBytes {
		return trainVMCheckpointManifest{}, fmt.Errorf("checkpoint manifest identity or bounds are invalid")
	}
	previousComponent := ""
	for _, component := range manifest.StateComponents {
		if !validCheckpointStateComponent(component) || component <= previousComponent {
			return trainVMCheckpointManifest{}, fmt.Errorf("checkpoint state inventory is invalid")
		}
		previousComponent = component
	}
	var total uint64
	previousPath := ""
	seenPaths := map[string]struct{}{}
	for _, object := range manifest.Objects {
		_, hashOK := normalizedSHA256(object.SHA256)
		if !hashOK || !validCheckpointRelativePath(object.RelativePath) || object.RelativePath <= previousPath {
			return trainVMCheckpointManifest{}, fmt.Errorf("checkpoint object inventory is invalid")
		}
		if _, duplicate := seenPaths[object.RelativePath]; duplicate {
			return trainVMCheckpointManifest{}, fmt.Errorf("checkpoint object inventory is duplicated")
		}
		seenPaths[object.RelativePath] = struct{}{}
		previousPath = object.RelativePath
		if ^uint64(0)-total < object.SizeBytes {
			return trainVMCheckpointManifest{}, fmt.Errorf("checkpoint object sizes overflow")
		}
		total += object.SizeBytes
	}
	if total != manifest.PayloadSizeBytes {
		return trainVMCheckpointManifest{}, fmt.Errorf("checkpoint payload size is inconsistent")
	}
	return manifest, nil
}

func checkpointSummary(artifact trainvmstore.PublishedArtifact, manifest trainVMCheckpointManifest) trainVMCheckpointSummary {
	return trainVMCheckpointSummary{
		Valid:    true,
		Sequence: artifact.Sequence, ArtifactID: artifact.ArtifactID, LogicalName: artifact.LogicalName,
		CheckpointSchema: manifest.CheckpointSchema, NodeID: manifest.Producer.NodeID,
		AttemptID: manifest.Producer.AttemptID, OptimizerStep: manifest.OptimizerStep,
		ResumeGrade: manifest.ResumeGrade, StateComponents: manifest.StateComponents,
		FileCount: manifest.FileCount, PayloadSizeBytes: manifest.PayloadSizeBytes,
		ParentArtifactIDs: manifest.ParentArtifactIDs, CanonicalTreeDigest: manifest.CanonicalTreeDigest,
		PublishedAtNS: artifact.PublishedAtNS,
	}
}

func (s *Server) handleTrainVMCheckpoints(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	limit := trainVMCheckpointDefaultTail
	if raw := r.URL.Query().Get("limit"); raw != "" {
		parsed, parseErr := strconv.Atoi(raw)
		if parseErr != nil || parsed < 1 || parsed > trainVMCheckpointMaximumTail {
			http.Error(w, "checkpoint tail limit must be from 1 through 1000", http.StatusBadRequest)
			return
		}
		limit = parsed
	}
	artifacts, found, err := s.publishedCheckpoints(r.Context(), r.PathValue("run"), limit)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run", http.StatusNotFound)
		return
	}
	result := make([]trainVMCheckpointSummary, 0, len(artifacts))
	for _, artifact := range artifacts {
		manifest, err := s.loadCheckpoint(artifact)
		if err != nil {
			result = append(result, trainVMCheckpointSummary{
				Valid: false, ValidationError: "manifest unavailable or failed immutable verification",
				Sequence: artifact.Sequence, ArtifactID: artifact.ArtifactID,
				LogicalName: artifact.LogicalName, CheckpointSchema: artifact.Schema,
				NodeID: artifact.ProducerNodeID, AttemptID: artifact.ProducerAttemptID,
				ParentArtifactIDs: append([]string(nil), artifact.ParentArtifactIDs...),
				PublishedAtNS:     artifact.PublishedAtNS,
			})
			continue
		}
		result = append(result, checkpointSummary(artifact, manifest))
	}
	sort.SliceStable(result, func(i, j int) bool {
		if result[i].OptimizerStep != result[j].OptimizerStep {
			return result[i].OptimizerStep < result[j].OptimizerStep
		}
		return result[i].Sequence < result[j].Sequence
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(result)
}
