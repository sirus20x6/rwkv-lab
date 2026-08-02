package server

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var (
	errGalleryPathOutsideRoots = errors.New("gallery path is outside allowed data roots")
	errGalleryPathUnsupported  = errors.New("secure gallery path authority is unsupported")
)

// galleryRootAuthority is a pinned directory descriptor. Implementations must
// resolve relative names beneath that descriptor without following symlinks.
// This keeps both the configured root and the selected object stable across the
// authorization/open boundary.
type galleryRootAuthority interface {
	Open(relative string) (*os.File, error)
	Close() error
}

func configuredImageRoots(cfg Config) []string {
	if len(cfg.ImageRoots) != 0 {
		return cfg.ImageRoots
	}
	return []string{cfg.RunsDir, cfg.RepoRoot}
}

// openGalleryFile opens an absolute artifact path through a pinned configured
// root. The lexical check only selects a candidate authority; openat2 performs
// the security decision atomically relative to the root descriptor.
func (s *Server) openGalleryFile(path string) (*os.File, error) {
	if path == "" || !filepath.IsAbs(path) {
		return nil, errGalleryPathOutsideRoots
	}
	path = filepath.Clean(path)
	var firstOpenErr error
	for _, configuredRoot := range configuredImageRoots(s.cfg) {
		if configuredRoot == "" || !filepath.IsAbs(configuredRoot) {
			continue
		}
		root := filepath.Clean(configuredRoot)
		relative, err := filepath.Rel(root, path)
		if err != nil || relative == "." || !filepath.IsLocal(relative) || relative == ".." ||
			strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			continue
		}
		authority, err := pinGalleryRoot(root)
		if err != nil {
			if errors.Is(err, errGalleryPathUnsupported) {
				return nil, err
			}
			if firstOpenErr == nil {
				firstOpenErr = fmt.Errorf("pin configured gallery root: %w", err)
			}
			continue
		}
		file, openErr := authority.Open(relative)
		closeErr := authority.Close()
		if openErr == nil {
			if closeErr != nil {
				_ = file.Close()
				return nil, fmt.Errorf("close configured gallery root: %w", closeErr)
			}
			return file, nil
		}
		if firstOpenErr == nil {
			firstOpenErr = fmt.Errorf("open gallery object beneath configured root: %w", openErr)
		}
	}
	if firstOpenErr != nil {
		return nil, firstOpenErr
	}
	return nil, errGalleryPathOutsideRoots
}
