//go:build linux

package server

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

const galleryOpenat2Resolve = unix.RESOLVE_BENEATH | unix.RESOLVE_NO_MAGICLINKS | unix.RESOLVE_NO_SYMLINKS

type linuxGalleryRootAuthority struct {
	fd int
}

func pinGalleryRoot(root string) (galleryRootAuthority, error) {
	how := &unix.OpenHow{
		Flags:   unix.O_PATH | unix.O_DIRECTORY | unix.O_CLOEXEC,
		Resolve: unix.RESOLVE_NO_MAGICLINKS | unix.RESOLVE_NO_SYMLINKS,
	}
	fd, err := unix.Openat2(unix.AT_FDCWD, root, how)
	if err != nil {
		if errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.EINVAL) {
			return nil, fmt.Errorf("%w: Linux openat2 is unavailable: %v", errGalleryPathUnsupported, err)
		}
		return nil, err
	}
	return &linuxGalleryRootAuthority{fd: fd}, nil
}

func (authority *linuxGalleryRootAuthority) Open(relative string) (*os.File, error) {
	if authority == nil || authority.fd < 0 {
		return nil, fmt.Errorf("gallery root authority is closed")
	}
	how := &unix.OpenHow{
		Flags:   unix.O_RDONLY | unix.O_CLOEXEC | unix.O_NOFOLLOW,
		Resolve: galleryOpenat2Resolve,
	}
	fd, err := unix.Openat2(authority.fd, relative, how)
	if err != nil {
		if errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.EINVAL) {
			return nil, fmt.Errorf("%w: Linux openat2 confinement is unavailable: %v", errGalleryPathUnsupported, err)
		}
		return nil, err
	}
	file := os.NewFile(uintptr(fd), relative)
	if file == nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("adopt gallery object descriptor")
	}
	return file, nil
}

func (authority *linuxGalleryRootAuthority) Close() error {
	if authority == nil || authority.fd < 0 {
		return nil
	}
	fd := authority.fd
	authority.fd = -1
	return unix.Close(fd)
}
