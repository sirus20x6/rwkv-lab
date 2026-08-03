//go:build linux

package server

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestGalleryRootAuthorityPinsConfiguredDirectoryIdentity(t *testing.T) {
	parent := t.TempDir()
	configured := filepath.Join(parent, "configured")
	retired := filepath.Join(parent, "retired")
	if err := os.Mkdir(configured, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configured, "image.png"), []byte("authorized"), 0o600); err != nil {
		t.Fatal(err)
	}
	authority, err := pinGalleryRoot(configured)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()
	if err := os.Rename(configured, retired); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(configured, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configured, "image.png"), []byte("replacement"), 0o600); err != nil {
		t.Fatal(err)
	}

	file, err := authority.Open("image.png")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "authorized" {
		t.Fatalf("pinned root opened replacement content %q", data)
	}
}

func TestGalleryRootAuthorityRejectsSymlinkedDescendants(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.png")
	if err := os.WriteFile(outside, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "linked.png")); err != nil {
		t.Fatal(err)
	}
	authority, err := pinGalleryRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()
	file, err := authority.Open("linked.png")
	if file != nil {
		file.Close()
		t.Fatal("symlinked gallery descendant was opened")
	}
	if !errors.Is(err, unix.ELOOP) {
		t.Fatalf("symlink rejection error = %v, want ELOOP", err)
	}
}
