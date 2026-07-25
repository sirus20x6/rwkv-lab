package server

import (
	"net/http/httptest"
	"testing"
	"testing/fstest"
)

func TestDashboardShellAndAssetsAreNeverCached(t *testing.T) {
	assets := fstest.MapFS{
		"index.html": {Data: []byte("<!doctype html><title>trainboard</title>")},
		"app.js":     {Data: []byte("// trainboard")},
	}
	srv := New(Config{Static: assets})
	for _, path := range []string{"/", "/static/app.js"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest("GET", path, nil)
			res := httptest.NewRecorder()
			srv.Handler().ServeHTTP(res, req)
			if res.Code != 200 {
				t.Fatalf("status = %d", res.Code)
			}
			if got := res.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
		})
	}
}
