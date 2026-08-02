package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLegacyTrainingMutationRoutesAreRetired(t *testing.T) {
	t.Parallel()
	handler := New(Config{}).Handler()
	paths := []string{
		"/api/runs/legacy/stop",
		"/api/runs/legacy/checkpoint",
		"/api/runs/legacy/control",
		"/api/runs/legacy/sample",
		"/api/launch",
		"/api/autostop",
		"/api/convboard/accept",
		"/api/experiments/run",
		"/api/experiments/launch",
		"/api/rlvr/launch",
		"/api/posttraining/version",
		"/api/posttraining/campaign",
		"/api/posttraining/compare",
		"/api/posttraining/feedback",
		"/api/qualification/run",
		"/api/queue/enqueue",
		"/api/queue/start-next",
		"/api/queue/cancel",
		"/api/queue/auto",
	}
	for _, path := range paths {
		path := path
		t.Run(path, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, path, nil)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusNotFound && response.Code != http.StatusMethodNotAllowed {
				t.Fatalf("retired route returned %d; want 404 or 405", response.Code)
			}
		})
	}
}
