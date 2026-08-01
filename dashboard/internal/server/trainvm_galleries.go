package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	trainvmstore "trainboard/internal/trainvm"
)

const (
	trainVMGallerySchema    = "rwkv-lab.eval-gallery.v2"
	trainVMGalleryMaxBytes  = 8 << 20
	trainVMGalleryMaxItems  = 512
	trainVMGalleryMaxEvents = 10_000
)

type trainVMGalleryItem struct {
	ItemID                  string            `json:"item_id"`
	HeldoutItemID           string            `json:"heldout_item_id"`
	HeldoutManifestDigest   string            `json:"heldout_manifest_digest"`
	PromptOrConditionDigest string            `json:"prompt_or_condition_digest"`
	GeneratedObjectSHA256   string            `json:"generated_object_sha256"`
	GeneratedObjectURI      string            `json:"generated_object_uri"`
	TargetObjectSHA256      string            `json:"target_object_sha256,omitempty"`
	TargetObjectURI         string            `json:"target_object_uri,omitempty"`
	SourceObjectSHA256      string            `json:"source_object_sha256,omitempty"`
	SourceObjectURI         string            `json:"source_object_uri,omitempty"`
	Seed                    uint64            `json:"seed"`
	SamplingAttributes      map[string]string `json:"sampling_attributes"`
}

type trainVMGalleryManifest struct {
	APIVersion               string               `json:"api_version"`
	RunID                    string               `json:"run_id"`
	NodeID                   string               `json:"node_id"`
	AttemptID                string               `json:"attempt_id"`
	Step                     uint64               `json:"step"`
	StepDomain               string               `json:"step_domain"`
	CheckpointManifestDigest string               `json:"checkpoint_manifest_digest"`
	EvaluatorProfileDigest   string               `json:"evaluator_profile_digest"`
	Items                    []trainVMGalleryItem `json:"items"`
	UsePolicyDigest          string               `json:"use_policy_digest"`
	CanonicalManifestDigest  string               `json:"canonical_manifest_digest"`
}

type trainVMGallerySummary struct {
	Sequence                 uint64 `json:"sequence"`
	ArtifactID               string `json:"artifact_id"`
	LogicalName              string `json:"logical_name"`
	Step                     uint64 `json:"step"`
	StepDomain               string `json:"step_domain"`
	NodeID                   string `json:"node_id"`
	AttemptID                string `json:"attempt_id"`
	ItemCount                int    `json:"item_count"`
	CheckpointManifestDigest string `json:"checkpoint_manifest_digest"`
	EvaluatorProfileDigest   string `json:"evaluator_profile_digest"`
	PublishedAtNS            int64  `json:"published_at_ns"`
}

type trainVMGalleryResponseItem struct {
	ItemID                  string            `json:"item_id"`
	HeldoutItemID           string            `json:"heldout_item_id"`
	HeldoutManifestDigest   string            `json:"heldout_manifest_digest"`
	PromptOrConditionDigest string            `json:"prompt_or_condition_digest"`
	Seed                    uint64            `json:"seed"`
	SamplingAttributes      map[string]string `json:"sampling_attributes"`
	GeneratedImageURL       string            `json:"generated_image_url"`
	TargetImageURL          string            `json:"target_image_url,omitempty"`
	SourceImageURL          string            `json:"source_image_url,omitempty"`
}

type trainVMGalleryResponse struct {
	trainVMGallerySummary
	Schema                  string                       `json:"schema"`
	UsePolicyDigest         string                       `json:"use_policy_digest"`
	CanonicalManifestDigest string                       `json:"canonical_manifest_digest"`
	Items                   []trainVMGalleryResponseItem `json:"items"`
}

func normalizedSHA256(value string) (string, bool) {
	value = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(value)), "sha256:")
	decoded, err := hex.DecodeString(value)
	return value, err == nil && len(decoded) == sha256.Size
}

func fileURIPath(raw string) (string, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "file" || (parsed.Host != "" && parsed.Host != "localhost") ||
		parsed.User != nil || parsed.Opaque != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("only local file URIs without query or fragment are supported")
	}
	path, err := url.PathUnescape(parsed.EscapedPath())
	if err != nil || !filepath.IsAbs(path) {
		return "", fmt.Errorf("file URI must contain an absolute path")
	}
	return filepath.FromSlash(path), nil
}

func readBoundedFile(path string, limit int64) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("file exceeds %d-byte limit", limit)
	}
	return data, nil
}

func (s *Server) publishedGalleries(ctx context.Context, runID string) ([]trainvmstore.PublishedArtifact, error) {
	var result []trainvmstore.PublishedArtifact
	var after uint64
	scanned := 0
	seen := map[string]struct{}{}
	for scanned < trainVMGalleryMaxEvents {
		page, err := trainvmstore.Artifacts(ctx, s.trainvm, runID, after, 1000)
		if err != nil {
			return nil, err
		}
		if len(page) == 0 {
			break
		}
		for _, artifact := range page {
			scanned++
			if artifact.Kind == "image_gallery" && artifact.Schema == trainVMGallerySchema {
				if _, duplicate := seen[artifact.ArtifactID]; duplicate {
					return nil, fmt.Errorf("duplicate published gallery artifact ID %q", artifact.ArtifactID)
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
	if scanned >= trainVMGalleryMaxEvents {
		return nil, fmt.Errorf("artifact history exceeds %d events", trainVMGalleryMaxEvents)
	}
	return result, nil
}

func findGalleryArtifact(artifacts []trainvmstore.PublishedArtifact, artifactID string) (trainvmstore.PublishedArtifact, bool) {
	for _, artifact := range artifacts {
		if artifact.ArtifactID == artifactID {
			return artifact, true
		}
	}
	return trainvmstore.PublishedArtifact{}, false
}

func validGalleryText(value string) bool {
	return value != "" && len(value) <= 512 && !strings.ContainsRune(value, '\x00')
}

func validGalleryDigest(value string) bool {
	_, ok := normalizedSHA256(value)
	return ok
}

func validGalleryArtifactID(value string) bool {
	return validGalleryText(value) && !strings.ContainsAny(value, "/\\")
}

func (s *Server) loadGallery(artifact trainvmstore.PublishedArtifact) (trainVMGalleryManifest, error) {
	var manifest trainVMGalleryManifest
	manifestPath, err := fileURIPath(artifact.URI)
	if err != nil {
		return manifest, err
	}
	resolved, ok := s.resolveEvalImage(manifestPath)
	if !ok {
		return manifest, fmt.Errorf("gallery manifest is outside allowed data roots")
	}
	data, err := readBoundedFile(resolved, trainVMGalleryMaxBytes)
	if err != nil {
		return manifest, fmt.Errorf("read gallery manifest: %w", err)
	}
	if uint64(len(data)) != artifact.SizeBytes {
		return manifest, fmt.Errorf("published gallery manifest size mismatch")
	}
	expected, ok := normalizedSHA256(artifact.Fingerprint)
	actual := sha256.Sum256(data)
	if artifact.FingerprintAlgorithm != "sha256" || !ok || expected != hex.EncodeToString(actual[:]) {
		return manifest, fmt.Errorf("published gallery manifest fingerprint mismatch")
	}
	if err := strictJSONDocument(data, &manifest); err != nil {
		return manifest, fmt.Errorf("decode gallery manifest: %w", err)
	}
	if !validGalleryArtifactID(artifact.ArtifactID) || manifest.APIVersion != trainVMGallerySchema ||
		manifest.RunID != artifact.RunID ||
		manifest.NodeID != artifact.ProducerNodeID || manifest.AttemptID != artifact.ProducerAttemptID ||
		!validGalleryText(manifest.StepDomain) || !validGalleryDigest(manifest.CheckpointManifestDigest) ||
		!validGalleryDigest(manifest.EvaluatorProfileDigest) || !validGalleryDigest(manifest.UsePolicyDigest) ||
		!validGalleryDigest(manifest.CanonicalManifestDigest) || len(manifest.Items) == 0 ||
		len(manifest.Items) > trainVMGalleryMaxItems {
		return trainVMGalleryManifest{}, fmt.Errorf("gallery manifest identity or cardinality is invalid")
	}
	seen := make(map[string]struct{}, len(manifest.Items))
	for _, item := range manifest.Items {
		_, generatedHashOK := normalizedSHA256(item.GeneratedObjectSHA256)
		_, targetHashOK := normalizedSHA256(item.TargetObjectSHA256)
		_, sourceHashOK := normalizedSHA256(item.SourceObjectSHA256)
		if !validGalleryText(item.ItemID) || !validGalleryText(item.HeldoutItemID) ||
			!validGalleryDigest(item.HeldoutManifestDigest) || !validGalleryDigest(item.PromptOrConditionDigest) ||
			!generatedHashOK || item.GeneratedObjectURI == "" || item.SamplingAttributes == nil ||
			(item.TargetObjectURI == "") != (item.TargetObjectSHA256 == "") ||
			(item.TargetObjectURI != "" && !targetHashOK) ||
			(item.SourceObjectURI == "") != (item.SourceObjectSHA256 == "") ||
			(item.SourceObjectURI != "" && !sourceHashOK) {
			return trainVMGalleryManifest{}, fmt.Errorf("gallery item %q is invalid", item.ItemID)
		}
		if _, err := fileURIPath(item.GeneratedObjectURI); err != nil {
			return trainVMGalleryManifest{}, fmt.Errorf("gallery item %q has an invalid generated URI", item.ItemID)
		}
		for _, optionalURI := range []string{item.TargetObjectURI, item.SourceObjectURI} {
			if optionalURI != "" {
				if _, err := fileURIPath(optionalURI); err != nil {
					return trainVMGalleryManifest{}, fmt.Errorf("gallery item %q has an invalid paired URI", item.ItemID)
				}
			}
		}
		if len(item.SamplingAttributes) > 64 {
			return trainVMGalleryManifest{}, fmt.Errorf("gallery item %q has too many sampling attributes", item.ItemID)
		}
		for key, value := range item.SamplingAttributes {
			if !validGalleryText(key) || len(key) > 128 || len(value) > 1024 || strings.ContainsRune(value, '\x00') {
				return trainVMGalleryManifest{}, fmt.Errorf("gallery item %q has an invalid sampling attribute", item.ItemID)
			}
		}
		if _, duplicate := seen[item.ItemID]; duplicate {
			return trainVMGalleryManifest{}, fmt.Errorf("gallery item %q is duplicated", item.ItemID)
		}
		seen[item.ItemID] = struct{}{}
	}
	return manifest, nil
}

func strictJSONDocument(data []byte, output any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(output); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("document contains trailing JSON")
	}
	return nil
}

func gallerySummary(artifact trainvmstore.PublishedArtifact, manifest trainVMGalleryManifest) trainVMGallerySummary {
	return trainVMGallerySummary{
		Sequence: artifact.Sequence, ArtifactID: artifact.ArtifactID, LogicalName: artifact.LogicalName,
		Step: manifest.Step, StepDomain: manifest.StepDomain, NodeID: manifest.NodeID,
		AttemptID: manifest.AttemptID, ItemCount: len(manifest.Items),
		CheckpointManifestDigest: manifest.CheckpointManifestDigest,
		EvaluatorProfileDigest:   manifest.EvaluatorProfileDigest, PublishedAtNS: artifact.PublishedAtNS,
	}
}

func (s *Server) handleTrainVMGalleries(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	artifacts, err := s.publishedGalleries(r.Context(), r.PathValue("run"))
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	result := make([]trainVMGallerySummary, 0, len(artifacts))
	for _, artifact := range artifacts {
		manifest, err := s.loadGallery(artifact)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		result = append(result, gallerySummary(artifact, manifest))
	}
	sort.SliceStable(result, func(i, j int) bool {
		if result[i].Step != result[j].Step {
			return result[i].Step < result[j].Step
		}
		return result[i].Sequence < result[j].Sequence
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(result)
}

func (s *Server) galleryForRequest(r *http.Request) (trainvmstore.PublishedArtifact, trainVMGalleryManifest, int, error) {
	artifacts, err := s.publishedGalleries(r.Context(), r.PathValue("run"))
	if err != nil {
		return trainvmstore.PublishedArtifact{}, trainVMGalleryManifest{}, 0, err
	}
	artifact, ok := findGalleryArtifact(artifacts, r.PathValue("artifact"))
	if !ok {
		return trainvmstore.PublishedArtifact{}, trainVMGalleryManifest{}, 0, os.ErrNotExist
	}
	manifest, err := s.loadGallery(artifact)
	if err != nil {
		return trainvmstore.PublishedArtifact{}, trainVMGalleryManifest{}, 0, err
	}
	index := -1
	if raw := r.PathValue("index"); raw != "" {
		index, err = strconv.Atoi(raw)
		if err != nil || index < 0 || index >= len(manifest.Items) {
			return trainvmstore.PublishedArtifact{}, trainVMGalleryManifest{}, 0, os.ErrNotExist
		}
	}
	return artifact, manifest, index, nil
}

func (s *Server) handleTrainVMGallery(w http.ResponseWriter, r *http.Request) {
	artifact, manifest, _, err := s.galleryForRequest(r)
	if err != nil {
		if os.IsNotExist(err) {
			http.Error(w, "published gallery not found", http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusBadGateway)
		}
		return
	}
	items := make([]trainVMGalleryResponseItem, len(manifest.Items))
	for index, item := range manifest.Items {
		base := fmt.Sprintf("/api/trainvm/runs/%s/galleries/%s/items/%d/",
			url.PathEscape(artifact.RunID), url.PathEscape(artifact.ArtifactID), index)
		generatedHash, _ := normalizedSHA256(item.GeneratedObjectSHA256)
		items[index] = trainVMGalleryResponseItem{
			ItemID: item.ItemID, HeldoutItemID: item.HeldoutItemID,
			HeldoutManifestDigest:   item.HeldoutManifestDigest,
			PromptOrConditionDigest: item.PromptOrConditionDigest,
			Seed:                    item.Seed, SamplingAttributes: item.SamplingAttributes,
			GeneratedImageURL: base + "generated?v=" + generatedHash,
		}
		if item.TargetObjectURI != "" {
			targetHash, _ := normalizedSHA256(item.TargetObjectSHA256)
			items[index].TargetImageURL = base + "target?v=" + targetHash
		}
		if item.SourceObjectURI != "" {
			sourceHash, _ := normalizedSHA256(item.SourceObjectSHA256)
			items[index].SourceImageURL = base + "source?v=" + sourceHash
		}
	}
	response := trainVMGalleryResponse{
		trainVMGallerySummary: gallerySummary(artifact, manifest), Schema: artifact.Schema,
		UsePolicyDigest:         manifest.UsePolicyDigest,
		CanonicalManifestDigest: manifest.CanonicalManifestDigest, Items: items,
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(response)
}

func (s *Server) handleTrainVMGalleryImage(w http.ResponseWriter, r *http.Request) {
	_, manifest, index, err := s.galleryForRequest(r)
	if err != nil {
		http.Error(w, "published gallery image not found", http.StatusNotFound)
		return
	}
	item := manifest.Items[index]
	var rawURI, rawHash string
	switch r.PathValue("role") {
	case "generated":
		rawURI, rawHash = item.GeneratedObjectURI, item.GeneratedObjectSHA256
	case "target":
		rawURI, rawHash = item.TargetObjectURI, item.TargetObjectSHA256
	case "source":
		rawURI, rawHash = item.SourceObjectURI, item.SourceObjectSHA256
	default:
		http.Error(w, "unknown gallery image role", http.StatusBadRequest)
		return
	}
	expected, hashOK := normalizedSHA256(rawHash)
	if rawURI == "" || !hashOK || r.URL.Query().Get("v") != expected {
		http.Error(w, "gallery image identity mismatch", http.StatusConflict)
		return
	}
	path, err := fileURIPath(rawURI)
	if err != nil {
		http.Error(w, "gallery image URI is invalid", http.StatusBadGateway)
		return
	}
	resolved, ok := s.resolveEvalImage(path)
	if !ok {
		http.Error(w, "gallery image is outside allowed data roots", http.StatusForbidden)
		return
	}
	file, err := os.Open(resolved)
	if err != nil {
		http.Error(w, "gallery image is unavailable", http.StatusNotFound)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > 128<<20 {
		http.Error(w, "gallery image is not a bounded regular file", http.StatusBadGateway)
		return
	}
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil || hex.EncodeToString(hasher.Sum(nil)) != expected {
		http.Error(w, "gallery image fingerprint mismatch", http.StatusConflict)
		return
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		http.Error(w, "gallery image is unavailable", http.StatusInternalServerError)
		return
	}
	prefix := make([]byte, 512)
	read, err := file.Read(prefix)
	if err != nil && err != io.EOF {
		http.Error(w, "gallery image is unavailable", http.StatusInternalServerError)
		return
	}
	contentType := http.DetectContentType(prefix[:read])
	switch contentType {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
	default:
		http.Error(w, "gallery object is not a supported raster image", http.StatusUnsupportedMediaType)
		return
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		http.Error(w, "gallery image is unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "private, immutable, max-age=31536000")
	http.ServeContent(w, r, filepath.Base(resolved), info.ModTime(), file)
}
