package server

import (
	"context"
	"html"
	"net/http"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

// handleArchitecture lazily computes a run's model architecture (layer types,
// param counts, trainable/frozen state) by shelling out to the v1 analyzer
// (legacy/dashboard_v1/architecture.py — safetensors metadata only, no torch
// load) via tools/arch_panel.py, and patches the rendered panel into #arch-body.
// On-demand (the panel's "load" button) so we don't read safetensors per tick.
func (s *Server) handleArchitecture(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	sse := datastar.NewSSE(w, r)
	if name == "" || strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		_ = sse.PatchElements(archError("invalid run name"))
		return
	}
	runDir := filepath.Join(s.cfg.RunsDir, name)
	py := filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python")
	wrapper := filepath.Join(s.cfg.RepoRoot, "dashboard2", "tools", "arch_panel.py")

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, py, wrapper, runDir)
	cmd.Dir = s.cfg.RepoRoot
	out, err := cmd.Output()
	if err != nil {
		_ = sse.PatchElements(archError("architecture unavailable (" + err.Error() + ")"))
		return
	}
	_ = sse.PatchElements(string(out))
}

func archError(msg string) string {
	return `<div id="arch-body" class="arch-body"><div class="empty">` + html.EscapeString(msg) + `</div></div>`
}
