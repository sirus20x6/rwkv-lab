package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLegacyExperimentPanelIsReadOnly(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "experiments"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "experiments", "historical.yaml"), []byte("kind: legacy\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	srv := &Server{cfg: Config{RepoRoot: root, RunsDir: filepath.Join(root, "runs")}}
	response := httptest.NewRecorder()
	srv.handleExperiments(response, httptest.NewRequest(http.MethodGet, "/api/experiments", nil))
	body := response.Body.String()
	if !strings.Contains(body, "legacy experiment form · read-only") ||
		!strings.Contains(body, "declarative TrainVM experiment") ||
		!strings.Contains(body, "experiments/historical.yaml") ||
		strings.Contains(body, "/api/experiments/launch") ||
		strings.Contains(body, "/api/experiments/run") {
		t.Fatalf("legacy experiment panel retained mutation authority: %s", body)
	}
}

func TestLegacyMutationHandlersFailClosedWithoutRedirects(t *testing.T) {
	srv := &Server{cfg: Config{RepoRoot: t.TempDir(), RunsDir: t.TempDir()}}
	handlers := map[string]func(http.ResponseWriter, *http.Request){
		"live control":        srv.handleSetControl,
		"experiment launch":   srv.handleLaunchExperiment,
		"config launch":       srv.handleRunConfig,
		"RLVR launch":         srv.handleLaunchRLVR,
		"dataset version":     srv.handleVersionPosttraining,
		"posttrain campaign":  srv.handleLaunchPosttrainingCampaign,
		"paired generation":   srv.handleComparePosttraining,
		"preference mutation": srv.handlePosttrainingFeedback,
		"qualification":       srv.handleRunQualification,
		"checkpoint sample":   srv.handleSample,
	}
	for name, handler := range handlers {
		t.Run(name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler(response, httptest.NewRequest(http.MethodPost, "/legacy-name", nil))
			if response.Code != http.StatusGone || response.Header().Get("Location") != "" {
				t.Fatalf("status=%d location=%q body=%s", response.Code,
					response.Header().Get("Location"), response.Body.String())
			}
		})
	}
}
