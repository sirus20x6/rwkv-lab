//go:build linux

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestTrainVMGalleryManifestCannotTraverseSymlinkInsideAllowedRoot(t *testing.T) {
	srv, _, manifestPath := trainVMGalleryFixture(t)
	realManifest := filepath.Join(filepath.Dir(manifestPath), "real-gallery.json")
	if err := os.Rename(manifestPath, realManifest); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(realManifest, manifestPath); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries/gallery-75", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("symlinked manifest status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMGalleryImageCannotTraverseSymlinkInsideAllowedRoot(t *testing.T) {
	srv, generatedPath, _ := trainVMGalleryFixture(t)
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries/gallery-75", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var gallery trainVMGalleryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &gallery); err != nil || len(gallery.Items) != 1 {
		t.Fatalf("gallery status=%d body=%s err=%v", response.Code, response.Body.String(), err)
	}
	realImage := filepath.Join(filepath.Dir(generatedPath), "real-generated.png")
	if err := os.Rename(generatedPath, realImage); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(realImage, generatedPath); err != nil {
		t.Fatal(err)
	}

	request = httptest.NewRequest(http.MethodGet, gallery.Items[0].GeneratedImageURL, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("symlinked image status=%d body=%s", response.Code, response.Body.String())
	}
}
