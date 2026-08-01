package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"

	trainvmstore "trainboard/internal/trainvm"
)

const (
	trainVMGPUTraceSchema       = "trainvm.gpu-trace.v1"
	trainVMGPUTraceManifestMax  = 4 << 20
	trainVMGPUTraceRawMax       = int64(2 << 30)
	trainVMGPUTraceMaxOperators = 256
	trainVMGPUTraceMaxEvents    = 10_000
)

type trainVMGPUTraceOperator struct {
	Name              string  `json:"name"`
	Calls             int64   `json:"calls"`
	CPUTimeUS         float64 `json:"cpu_time_us"`
	AcceleratorTimeUS float64 `json:"accelerator_time_us"`
}

type trainVMGPUTraceSummaryValues struct {
	CPUTimeUS                       float64                   `json:"cpu_time_us"`
	AcceleratorTimeUS               float64                   `json:"accelerator_time_us"`
	KernelOrOperatorCount           int64                     `json:"kernel_or_operator_count"`
	TopOperators                    []trainVMGPUTraceOperator `json:"top_operators"`
	AcceleratorLaunchCount          *uint64                   `json:"accelerator_launch_count,omitempty"`
	CapturedStepWallTimeUS          *float64                  `json:"captured_step_wall_time_us,omitempty"`
	GPUActiveRatio                  *float64                  `json:"gpu_active_ratio,omitempty"`
	GPUActiveTimeUS                 *float64                  `json:"gpu_active_time_us,omitempty"`
	InputStallRatio                 *float64                  `json:"input_stall_ratio,omitempty"`
	InputStallTimeUS                *float64                  `json:"input_stall_time_us,omitempty"`
	AllocatorBaselineAllocatedBytes *uint64                   `json:"allocator_baseline_allocated_bytes,omitempty"`
	AllocatorBaselineReservedBytes  *uint64                   `json:"allocator_baseline_reserved_bytes,omitempty"`
	AllocatorPeakAllocatedBytes     *uint64                   `json:"allocator_peak_allocated_bytes,omitempty"`
	AllocatorPeakReservedBytes      *uint64                   `json:"allocator_peak_reserved_bytes,omitempty"`
}

type trainVMGPUTraceOptions struct {
	ProfileMemory bool `json:"profile_memory"`
	RecordShapes  bool `json:"record_shapes"`
	WithStack     bool `json:"with_stack"`
}

type trainVMGPUTraceManifest struct {
	Activities              []string                     `json:"activities"`
	APIVersion              string                       `json:"api_version"`
	AttemptID               string                       `json:"attempt_id"`
	Backend                 string                       `json:"backend"`
	CanonicalManifestDigest string                       `json:"canonical_manifest_digest"`
	CaptureSteps            uint64                       `json:"capture_steps"`
	FirstOptimizerStep      uint64                       `json:"first_optimizer_step"`
	InstrumentedTiming      bool                         `json:"instrumented_timing"`
	InvocationDigest        string                       `json:"invocation_digest"`
	LastOptimizerStep       uint64                       `json:"last_optimizer_step"`
	NodeID                  string                       `json:"node_id"`
	Options                 trainVMGPUTraceOptions       `json:"options"`
	RunID                   string                       `json:"run_id"`
	Sensitivity             string                       `json:"sensitivity"`
	SkipSteps               uint64                       `json:"skip_steps"`
	Summary                 trainVMGPUTraceSummaryValues `json:"summary"`
	TraceSHA256             string                       `json:"trace_sha256"`
	TraceSizeBytes          uint64                       `json:"trace_size_bytes"`
	WarmupSteps             uint64                       `json:"warmup_steps"`
}

type trainVMGPUTraceSummary struct {
	Sequence                uint64                       `json:"sequence"`
	ArtifactID              string                       `json:"artifact_id"`
	LogicalName             string                       `json:"logical_name"`
	NodeID                  string                       `json:"node_id"`
	AttemptID               string                       `json:"attempt_id"`
	Backend                 string                       `json:"backend"`
	FirstOptimizerStep      uint64                       `json:"first_optimizer_step"`
	LastOptimizerStep       uint64                       `json:"last_optimizer_step"`
	CaptureSteps            uint64                       `json:"capture_steps"`
	SkipSteps               uint64                       `json:"skip_steps"`
	WarmupSteps             uint64                       `json:"warmup_steps"`
	Activities              []string                     `json:"activities"`
	Options                 trainVMGPUTraceOptions       `json:"options"`
	Summary                 trainVMGPUTraceSummaryValues `json:"summary"`
	TraceSHA256             string                       `json:"trace_sha256"`
	TraceSizeBytes          uint64                       `json:"trace_size_bytes"`
	Sensitivity             string                       `json:"sensitivity"`
	InvocationDigest        string                       `json:"invocation_digest"`
	CanonicalManifestDigest string                       `json:"canonical_manifest_digest"`
	PublishedAtNS           int64                        `json:"published_at_ns"`
	TraceDownloadURL        string                       `json:"trace_download_url"`
}

func (s *Server) publishedGPUTraces(ctx context.Context, runID string) ([]trainvmstore.PublishedArtifact, error) {
	var result []trainvmstore.PublishedArtifact
	var after uint64
	scanned := 0
	seen := map[string]struct{}{}
	for scanned < trainVMGPUTraceMaxEvents {
		page, err := trainvmstore.Artifacts(ctx, s.trainvm, runID, after, 1000)
		if err != nil {
			return nil, err
		}
		if len(page) == 0 {
			break
		}
		for _, artifact := range page {
			scanned++
			if artifact.Kind == "opaque" && artifact.Schema == trainVMGPUTraceSchema {
				if _, duplicate := seen[artifact.ArtifactID]; duplicate {
					return nil, fmt.Errorf("duplicate published GPU trace artifact ID %q", artifact.ArtifactID)
				}
				seen[artifact.ArtifactID] = struct{}{}
				result = append(result, artifact)
			}
			after = artifact.Sequence
		}
		if len(page) < 1000 {
			break
		}
	}
	if scanned >= trainVMGPUTraceMaxEvents {
		return nil, fmt.Errorf("artifact history exceeds %d events", trainVMGPUTraceMaxEvents)
	}
	return result, nil
}

func findGPUTraceArtifact(artifacts []trainvmstore.PublishedArtifact, artifactID string) (trainvmstore.PublishedArtifact, bool) {
	for _, artifact := range artifacts {
		if artifact.ArtifactID == artifactID {
			return artifact, true
		}
	}
	return trainvmstore.PublishedArtifact{}, false
}

func canonicalDigestWithoutField(data []byte, field string) (string, error) {
	var document map[string]json.RawMessage
	if err := strictJSONDocument(data, &document); err != nil {
		return "", err
	}
	if _, exists := document[field]; !exists {
		return "", fmt.Errorf("canonical digest field is missing")
	}
	delete(document, field)
	body, err := json.Marshal(document)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(body)
	return hex.EncodeToString(digest[:]), nil
}

func finiteNonnegative(value float64) bool {
	return value >= 0 && value < 1.7976931348623157e+308
}

func validRichGPUTraceSummary(summary trainVMGPUTraceSummaryValues) bool {
	present := []bool{
		summary.AcceleratorLaunchCount != nil,
		summary.CapturedStepWallTimeUS != nil,
		summary.GPUActiveRatio != nil,
		summary.GPUActiveTimeUS != nil,
		summary.AllocatorBaselineAllocatedBytes != nil,
		summary.AllocatorBaselineReservedBytes != nil,
		summary.AllocatorPeakAllocatedBytes != nil,
		summary.AllocatorPeakReservedBytes != nil,
	}
	count := 0
	for _, value := range present {
		if value {
			count++
		}
	}
	if count == 0 {
		// Read legacy v1 summaries without inventing unavailable metrics, but do
		// not accept detached input-stall fields without their wall-time basis.
		return summary.InputStallRatio == nil && summary.InputStallTimeUS == nil
	}
	if count != len(present) {
		return false
	}
	if (summary.InputStallRatio == nil) != (summary.InputStallTimeUS == nil) {
		return false
	}
	expectedActiveRatio := *summary.GPUActiveTimeUS / *summary.CapturedStepWallTimeUS
	valid := *summary.AcceleratorLaunchCount <= 1_000_000_000_000 &&
		finiteNonnegative(*summary.CapturedStepWallTimeUS) && *summary.CapturedStepWallTimeUS > 0 &&
		finiteNonnegative(*summary.GPUActiveTimeUS) &&
		*summary.GPUActiveTimeUS <= *summary.CapturedStepWallTimeUS*1.000001 &&
		finiteNonnegative(*summary.GPUActiveRatio) && *summary.GPUActiveRatio <= 1 &&
		math.Abs(*summary.GPUActiveRatio-expectedActiveRatio) <= 0.000001 &&
		*summary.AllocatorBaselineAllocatedBytes <= *summary.AllocatorBaselineReservedBytes &&
		*summary.AllocatorPeakAllocatedBytes <= *summary.AllocatorPeakReservedBytes &&
		*summary.AllocatorBaselineAllocatedBytes <= *summary.AllocatorPeakAllocatedBytes &&
		*summary.AllocatorBaselineReservedBytes <= *summary.AllocatorPeakReservedBytes
	if !valid || summary.InputStallRatio == nil {
		return valid
	}
	expectedInputRatio := *summary.InputStallTimeUS / *summary.CapturedStepWallTimeUS
	return finiteNonnegative(*summary.InputStallTimeUS) &&
		*summary.InputStallTimeUS <= *summary.CapturedStepWallTimeUS*1.000001 &&
		finiteNonnegative(*summary.InputStallRatio) && *summary.InputStallRatio <= 1 &&
		math.Abs(*summary.InputStallRatio-expectedInputRatio) <= 0.000001
}

func (s *Server) loadGPUTrace(artifact trainvmstore.PublishedArtifact) (trainVMGPUTraceManifest, string, error) {
	var manifest trainVMGPUTraceManifest
	manifestPath, err := fileURIPath(artifact.URI)
	if err != nil {
		return manifest, "", err
	}
	resolved, ok := s.resolveEvalImage(manifestPath)
	if !ok {
		return manifest, "", fmt.Errorf("GPU trace manifest is outside allowed data roots")
	}
	data, err := readBoundedFile(resolved, trainVMGPUTraceManifestMax)
	if err != nil {
		return manifest, "", fmt.Errorf("read GPU trace manifest: %w", err)
	}
	expected, ok := normalizedSHA256(artifact.Fingerprint)
	actual := sha256.Sum256(data)
	if artifact.FingerprintAlgorithm != "adapter" || !ok || expected != hex.EncodeToString(actual[:]) {
		return manifest, "", fmt.Errorf("published GPU trace manifest fingerprint mismatch")
	}
	if err := strictJSONDocument(data, &manifest); err != nil {
		return manifest, "", fmt.Errorf("decode GPU trace manifest: %w", err)
	}
	canonical, err := canonicalDigestWithoutField(data, "canonical_manifest_digest")
	declaredCanonical, canonicalOK := normalizedSHA256(manifest.CanonicalManifestDigest)
	_, traceHashOK := normalizedSHA256(manifest.TraceSHA256)
	_, invocationOK := normalizedSHA256(manifest.InvocationDigest)
	if err != nil || !canonicalOK || canonical != declaredCanonical || !traceHashOK || !invocationOK ||
		manifest.APIVersion != trainVMGPUTraceSchema || manifest.RunID != artifact.RunID ||
		manifest.NodeID != artifact.ProducerNodeID || manifest.AttemptID != artifact.ProducerAttemptID ||
		manifest.Backend != "torch" || manifest.Sensitivity != "restricted" || !manifest.InstrumentedTiming ||
		manifest.CaptureSteps == 0 || manifest.CaptureSteps > 128 || manifest.SkipSteps > 256 ||
		manifest.WarmupSteps > 256 || manifest.CaptureSteps+manifest.SkipSteps+manifest.WarmupSteps > 512 ||
		manifest.LastOptimizerStep < manifest.FirstOptimizerStep ||
		manifest.LastOptimizerStep-manifest.FirstOptimizerStep+1 != manifest.CaptureSteps ||
		manifest.TraceSizeBytes == 0 || manifest.TraceSizeBytes > uint64(trainVMGPUTraceRawMax) ||
		uint64(len(data))+manifest.TraceSizeBytes != artifact.SizeBytes ||
		len(manifest.Activities) == 0 || len(manifest.Activities) > 2 ||
		len(manifest.Summary.TopOperators) > trainVMGPUTraceMaxOperators ||
		manifest.Summary.KernelOrOperatorCount < int64(len(manifest.Summary.TopOperators)) ||
		!finiteNonnegative(manifest.Summary.CPUTimeUS) || !finiteNonnegative(manifest.Summary.AcceleratorTimeUS) ||
		!validRichGPUTraceSummary(manifest.Summary) {
		return trainVMGPUTraceManifest{}, "", fmt.Errorf("GPU trace manifest identity or bounds are invalid")
	}
	seenActivities := map[string]struct{}{}
	hasAccelerator := false
	for _, activity := range manifest.Activities {
		if activity != "cpu" && activity != "accelerator" {
			return trainVMGPUTraceManifest{}, "", fmt.Errorf("GPU trace activity is invalid")
		}
		if _, duplicate := seenActivities[activity]; duplicate {
			return trainVMGPUTraceManifest{}, "", fmt.Errorf("GPU trace activity is duplicated")
		}
		seenActivities[activity] = struct{}{}
		hasAccelerator = hasAccelerator || activity == "accelerator"
	}
	if !hasAccelerator {
		return trainVMGPUTraceManifest{}, "", fmt.Errorf("GPU trace has no accelerator activity")
	}
	for _, operator := range manifest.Summary.TopOperators {
		if !validGalleryText(operator.Name) || operator.Calls < 0 ||
			!finiteNonnegative(operator.CPUTimeUS) || !finiteNonnegative(operator.AcceleratorTimeUS) {
			return trainVMGPUTraceManifest{}, "", fmt.Errorf("GPU trace operator summary is invalid")
		}
	}
	tracePath := filepath.Join(filepath.Dir(resolved), "trace.json")
	traceResolved, ok := s.resolveEvalImage(tracePath)
	if !ok {
		return trainVMGPUTraceManifest{}, "", fmt.Errorf("raw GPU trace is outside allowed data roots")
	}
	return manifest, traceResolved, nil
}

func gpuTraceSummary(artifact trainvmstore.PublishedArtifact, manifest trainVMGPUTraceManifest) trainVMGPUTraceSummary {
	traceHash, _ := normalizedSHA256(manifest.TraceSHA256)
	return trainVMGPUTraceSummary{
		Sequence: artifact.Sequence, ArtifactID: artifact.ArtifactID, LogicalName: artifact.LogicalName,
		NodeID: manifest.NodeID, AttemptID: manifest.AttemptID, Backend: manifest.Backend,
		FirstOptimizerStep: manifest.FirstOptimizerStep, LastOptimizerStep: manifest.LastOptimizerStep,
		CaptureSteps: manifest.CaptureSteps, SkipSteps: manifest.SkipSteps, WarmupSteps: manifest.WarmupSteps,
		Activities: manifest.Activities, Options: manifest.Options, Summary: manifest.Summary,
		TraceSHA256: manifest.TraceSHA256, TraceSizeBytes: manifest.TraceSizeBytes,
		Sensitivity: manifest.Sensitivity, InvocationDigest: manifest.InvocationDigest,
		CanonicalManifestDigest: manifest.CanonicalManifestDigest, PublishedAtNS: artifact.PublishedAtNS,
		TraceDownloadURL: fmt.Sprintf("/api/trainvm/runs/%s/profiles/%s/trace?v=%s",
			url.PathEscape(artifact.RunID), url.PathEscape(artifact.ArtifactID), traceHash),
	}
}

func (s *Server) handleTrainVMProfiles(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	artifacts, err := s.publishedGPUTraces(r.Context(), r.PathValue("run"))
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	result := make([]trainVMGPUTraceSummary, 0, len(artifacts))
	for _, artifact := range artifacts {
		manifest, _, err := s.loadGPUTrace(artifact)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		result = append(result, gpuTraceSummary(artifact, manifest))
	}
	sort.SliceStable(result, func(i, j int) bool {
		if result[i].FirstOptimizerStep != result[j].FirstOptimizerStep {
			return result[i].FirstOptimizerStep < result[j].FirstOptimizerStep
		}
		return result[i].Sequence < result[j].Sequence
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(result)
}

func (s *Server) gpuTraceForRequest(r *http.Request) (trainvmstore.PublishedArtifact, trainVMGPUTraceManifest, string, error) {
	artifacts, err := s.publishedGPUTraces(r.Context(), r.PathValue("run"))
	if err != nil {
		return trainvmstore.PublishedArtifact{}, trainVMGPUTraceManifest{}, "", err
	}
	artifact, found := findGPUTraceArtifact(artifacts, r.PathValue("artifact"))
	if !found {
		return trainvmstore.PublishedArtifact{}, trainVMGPUTraceManifest{}, "", os.ErrNotExist
	}
	manifest, tracePath, err := s.loadGPUTrace(artifact)
	return artifact, manifest, tracePath, err
}

func (s *Server) handleTrainVMProfile(w http.ResponseWriter, r *http.Request) {
	artifact, manifest, _, err := s.gpuTraceForRequest(r)
	if err != nil {
		if os.IsNotExist(err) {
			http.Error(w, "published GPU trace not found", http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusBadGateway)
		}
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(gpuTraceSummary(artifact, manifest))
}

func (s *Server) handleTrainVMProfileTrace(w http.ResponseWriter, r *http.Request) {
	artifact, manifest, tracePath, err := s.gpuTraceForRequest(r)
	if err != nil {
		http.Error(w, "published GPU trace not found", http.StatusNotFound)
		return
	}
	expected, ok := normalizedSHA256(manifest.TraceSHA256)
	if !ok || r.URL.Query().Get("v") != expected {
		http.Error(w, "GPU trace identity mismatch", http.StatusConflict)
		return
	}
	file, err := os.Open(tracePath)
	if err != nil {
		http.Error(w, "raw GPU trace is unavailable", http.StatusNotFound)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 ||
		uint64(info.Size()) != manifest.TraceSizeBytes || info.Size() > trainVMGPUTraceRawMax {
		http.Error(w, "raw GPU trace size is invalid", http.StatusBadGateway)
		return
	}
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil || hex.EncodeToString(hasher.Sum(nil)) != expected {
		http.Error(w, "raw GPU trace fingerprint mismatch", http.StatusConflict)
		return
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		http.Error(w, "raw GPU trace is unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", artifact.ArtifactID+".json"))
	w.Header().Set("Cache-Control", "private, no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	http.ServeContent(w, r, filepath.Base(tracePath), info.ModTime(), file)
}
