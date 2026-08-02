//go:build !linux

package server

import "fmt"

func pinGalleryRoot(string) (galleryRootAuthority, error) {
	return nil, fmt.Errorf("%w: openat2 confinement requires Linux", errGalleryPathUnsupported)
}
