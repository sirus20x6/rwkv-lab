package server

import (
	"encoding/json"
	"mime"
	"net/http"
	"strings"
	"unicode"

	trainvmstore "trainboard/internal/trainvm"
)

func (s *Server) handleTrainVMReproducibilityCapsule(
	w http.ResponseWriter, r *http.Request,
) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	runID := r.PathValue("run")
	capsule, found, err := trainvmstore.BuildReproducibilityCapsule(
		r.Context(), s.trainvm, runID,
	)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	if !found {
		http.Error(w, "no such TrainVM run or persisted plan", http.StatusNotFound)
		return
	}
	filename := "trainvm-" + capsuleFilenameComponent(runID) + "-capsule.json"
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", mime.FormatMediaType(
		"attachment", map[string]string{"filename": filename},
	))
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("ETag", `"`+capsule.CapsuleDigest+`"`)
	_ = json.NewEncoder(w).Encode(capsule)
}

func capsuleFilenameComponent(value string) string {
	value = strings.TrimSpace(value)
	var result strings.Builder
	result.Grow(len(value))
	for _, character := range value {
		if unicode.IsLetter(character) || unicode.IsDigit(character) ||
			character == '-' || character == '_' || character == '.' {
			result.WriteRune(character)
		} else {
			result.WriteByte('_')
		}
		if result.Len() >= 128 {
			break
		}
	}
	if result.Len() == 0 {
		return "run"
	}
	return result.String()
}
