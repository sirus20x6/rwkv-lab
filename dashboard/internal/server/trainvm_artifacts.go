package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	trainvmstore "trainboard/internal/trainvm"
)

const (
	trainVMGenericArtifactMaxBytes = int64(64 << 20)
)

func (s *Server) publishedArtifactByID(
	r *http.Request, artifactID string, sequence uint64,
) (trainvmstore.PublishedArtifact, bool, error) {
	if !validGalleryArtifactID(artifactID) || sequence == 0 {
		return trainvmstore.PublishedArtifact{}, false, nil
	}
	// The browser carries the immutable publication sequence with the content
	// fingerprint. Resolve that exact journal position in O(1) bounded work;
	// scanning from sequence zero would eventually make recent artifacts
	// unreachable on long-running or high-frequency jobs.
	page, err := trainvmstore.Artifacts(
		r.Context(), s.trainvm, r.PathValue("run"), sequence-1, 1)
	if err != nil {
		return trainvmstore.PublishedArtifact{}, false, err
	}
	if len(page) != 1 || page[0].Sequence != sequence || page[0].ArtifactID != artifactID {
		return trainvmstore.PublishedArtifact{}, false, nil
	}
	return page[0], true, nil
}

func (s *Server) readVerifiedArtifact(
	artifact trainvmstore.PublishedArtifact,
) ([]byte, os.FileInfo, string, error) {
	if artifact.FingerprintAlgorithm != "sha256" &&
		artifact.FingerprintAlgorithm != "manifest_sha256" {
		return nil, nil, "", fmt.Errorf("artifact fingerprint algorithm is not downloadable")
	}
	expected, ok := normalizedSHA256(artifact.Fingerprint)
	if !ok {
		return nil, nil, "", fmt.Errorf("artifact fingerprint is malformed")
	}
	path, err := fileURIPath(artifact.URI)
	if err != nil {
		return nil, nil, "", err
	}
	resolved, ok := s.resolveEvalImage(path)
	if !ok {
		return nil, nil, "", fmt.Errorf("artifact is outside allowed data roots")
	}
	file, err := os.Open(resolved)
	if err != nil {
		return nil, nil, "", err
	}
	fail := func(err error) ([]byte, os.FileInfo, string, error) {
		_ = file.Close()
		return nil, nil, "", err
	}
	// On Linux, re-authorize the already-open file descriptor rather than the
	// pathname used to open it. This closes a symlink-swap window between the
	// allowed-root check and open(2); fingerprint verification alone is not a
	// substitute for path authority.
	if runtime.GOOS == "linux" {
		if _, allowed := s.resolveEvalImage(fmt.Sprintf("/proc/self/fd/%d", file.Fd())); !allowed {
			return fail(fmt.Errorf("opened artifact escaped the allowed data roots"))
		}
	}
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return fail(fmt.Errorf("artifact is not a regular file"))
	}
	if info.Size() < 0 || info.Size() > trainVMGenericArtifactMaxBytes {
		return fail(fmt.Errorf("artifact file exceeds the bounded download size"))
	}
	if artifact.FingerprintAlgorithm == "sha256" &&
		uint64(info.Size()) != artifact.SizeBytes {
		return fail(fmt.Errorf("artifact size disagrees with its publication"))
	}
	if artifact.FingerprintAlgorithm == "manifest_sha256" &&
		uint64(info.Size()) > artifact.SizeBytes {
		return fail(fmt.Errorf("artifact manifest exceeds its published total size"))
	}
	content, err := io.ReadAll(io.LimitReader(file, trainVMGenericArtifactMaxBytes+1))
	if err != nil {
		return fail(fmt.Errorf("read artifact: %w", err))
	}
	if int64(len(content)) != info.Size() {
		return fail(fmt.Errorf("artifact changed while its immutable snapshot was being read"))
	}
	hash := sha256.New()
	if _, err := hash.Write(content); err != nil {
		return fail(fmt.Errorf("hash artifact: %w", err))
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if actual != expected {
		return fail(fmt.Errorf("artifact content failed fingerprint verification"))
	}
	_ = file.Close()
	return content, info, expected, nil
}

func (s *Server) handleTrainVMArtifactContent(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	sequence, parseErr := strconv.ParseUint(strings.TrimSpace(r.URL.Query().Get("s")), 10, 64)
	if parseErr != nil || sequence == 0 {
		http.Error(w, "bounded artifact publication sequence is required", http.StatusBadRequest)
		return
	}
	artifact, found, err := s.publishedArtifactByID(r, r.PathValue("artifact"), sequence)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if !found {
		http.Error(w, "published artifact not found", http.StatusNotFound)
		return
	}
	expected, ok := normalizedSHA256(artifact.Fingerprint)
	if !ok {
		http.Error(w, "published artifact fingerprint is malformed", http.StatusBadGateway)
		return
	}
	if token := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(r.URL.Query().Get("v"))), "sha256:"); token == "" || token != expected {
		http.Error(w, "artifact snapshot changed; refresh the run", http.StatusConflict)
		return
	}
	content, info, fingerprint, err := s.readVerifiedArtifact(artifact)
	if err != nil {
		log.Printf("TrainVM artifact %q at sequence %d failed verification: %v",
			artifact.ArtifactID, artifact.Sequence, err)
		http.Error(w, "published artifact could not be verified", http.StatusBadGateway)
		return
	}
	name := filepath.Base(info.Name())
	w.Header().Set("Cache-Control", "private, immutable")
	w.Header().Set("ETag", `"sha256:`+fingerprint+`"`)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Disposition", "attachment; filename="+strconv.Quote(name))
	if contentType := mime.TypeByExtension(filepath.Ext(name)); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	} else {
		w.Header().Set("Content-Type", "application/octet-stream")
	}
	http.ServeContent(w, r, name, info.ModTime(), bytes.NewReader(content))
}
