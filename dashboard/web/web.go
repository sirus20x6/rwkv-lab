// Package web embeds the static front-end assets so trainboard ships as a
// single self-contained binary. Set TRAINBOARD_STATIC=<dir> to serve from disk
// instead (dev convenience — edit HTML/JS without rebuilding).
package web

import (
	"embed"
	"io/fs"
	"os"
)

//go:embed static
var embedded embed.FS

// Static returns the front-end asset filesystem rooted at the static/ dir.
// Honors the TRAINBOARD_STATIC override for live-editing during development.
func Static() fs.FS {
	if dir := os.Getenv("TRAINBOARD_STATIC"); dir != "" {
		return os.DirFS(dir)
	}
	sub, err := fs.Sub(embedded, "static")
	if err != nil {
		panic(err) // embed path is a compile-time constant; can't fail at runtime
	}
	return sub
}
