package server

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/starfederation/datastar-go/datastar"

	"trainboard/internal/sysmon"
)

// Control actions are confirm-gated client-side and validated here. They only
// ever signal/spawn allowlisted training processes, never escalate, and never
// kill -9. Every attempt is written to the actions audit table.

func nowTs() float64 { return float64(time.Now().UnixNano()) / 1e9 }

// toast pushes a transient status line to the UI via the $toast signal.
// $toastKind colors it: "" neutral info, "ok" green success, "err" red failure.
func toast(sse *datastar.ServerSentEventGenerator, msg string) { toastKind(sse, "", msg) }

func toastOK(sse *datastar.ServerSentEventGenerator, msg string) { toastKind(sse, "ok", msg) }

func toastErr(sse *datastar.ServerSentEventGenerator, msg string) { toastKind(sse, "err", msg) }

func toastKind(sse *datastar.ServerSentEventGenerator, kind, msg string) {
	_ = sse.MarshalAndPatchSignals(map[string]any{"toast": msg, "toastKind": kind})
}

func (s *Server) procFor(name string) (sysmon.Proc, bool) {
	for _, p := range s.sampler.Latest().Procs {
		if p.RunName == name {
			return p, true
		}
	}
	return sysmon.Proc{}, false
}

// handleStop sends SIGINT (graceful save+exit — the trainers checkpoint on
// interrupt) to the run's live process.
func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	sse := datastar.NewSSE(w, r)
	proc, ok := s.procFor(name)
	if !ok {
		s.db.LogAction(nowTs(), "stop", name, "{}", "no live process", 0)
		toastErr(sse, "stop: no live process for "+name)
		return
	}
	alive, _, _ := sysmon.VerifyTrainingPID(proc.PID)
	if !alive {
		s.db.LogAction(nowTs(), "stop", name, "{}", "pid not a training process", int(proc.PID))
		toastErr(sse, "stop: PID no longer a training process")
		return
	}
	if err := syscall.Kill(int(proc.PID), syscall.SIGINT); err != nil {
		s.db.LogAction(nowTs(), "stop", name, "{}", "kill error: "+err.Error(), int(proc.PID))
		toastErr(sse, "stop failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "stop", name, "{}", "SIGINT sent", int(proc.PID))
	toastOK(sse, fmt.Sprintf("SIGINT sent to PID %d (%s) — it will save & exit", proc.PID, name))
}

// handleCheckpoint sends SIGUSR1 (save-without-exit). Gated to instrumented
// trainers ONLY: SIGUSR1's default disposition is to TERMINATE, so signaling an
// un-instrumented python process would kill it. Phase 6's instrumented copies
// install a SIGUSR1 handler; until one is running this safely refuses.
func (s *Server) handleCheckpoint(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	sse := datastar.NewSSE(w, r)
	proc, ok := s.procFor(name)
	if !ok {
		toastErr(sse, "checkpoint: no live process for "+name)
		return
	}
	alive, _, instrumented := sysmon.VerifyTrainingPID(proc.PID)
	if !alive {
		toastErr(sse, "checkpoint: PID no longer a training process")
		return
	}
	if !instrumented {
		s.db.LogAction(nowTs(), "checkpoint", name, "{}", "refused: not instrumented", int(proc.PID))
		toastErr(sse, "checkpoint-now needs the instrumented trainer (Phase 6) — refused to avoid killing the run")
		return
	}
	if err := syscall.Kill(int(proc.PID), syscall.SIGUSR1); err != nil {
		toastErr(sse, "checkpoint failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "checkpoint", name, "{}", "SIGUSR1 sent", int(proc.PID))
	toastOK(sse, fmt.Sprintf("SIGUSR1 sent to PID %d — checkpoint requested", proc.PID))
}

// handleNotes / handleTags persist user annotations on a run.
func (s *Server) handleNotes(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var sig struct {
		Notes string `json:"notes"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	if err := s.db.SetNotes(name, sig.Notes); err != nil {
		toastErr(sse, "notes save failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "notes", name, "{}", "saved", 0)
	toastOK(sse, "notes saved for "+name)
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var sig struct {
		Tags string `json:"tags"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	// store the raw comma/space list as a JSON array
	tags := splitTags(sig.Tags)
	tagsJSON := "[" + strings.Join(quoteAll(tags), ",") + "]"
	if err := s.db.SetTags(name, tagsJSON); err != nil {
		toastErr(sse, "tags save failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "tags", name, tagsJSON, "saved", 0)
	toastOK(sse, "tags saved for "+name)
}

// handleLaunch spawns an allowlisted training script (detached). Body signals:
// {launchScript, launchArgs}. Args are split on whitespace and passed as
// separate argv (no shell — no metacharacter injection).
func (s *Server) handleLaunch(w http.ResponseWriter, r *http.Request) {
	var sig struct {
		LaunchScript string `json:"launchScript"`
		LaunchArgs   string `json:"launchArgs"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)

	pid, logPath, err := s.spawnTraining(sig.LaunchScript, sig.LaunchArgs)
	if err != nil {
		s.db.LogAction(nowTs(), "launch", "", `{"script":"`+sig.LaunchScript+`"}`, "refused/failed: "+err.Error(), 0)
		toastErr(sse, "launch refused: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "launch", "", fmt.Sprintf(`{"script":%q,"args":%q,"log":%q}`, sig.LaunchScript, sig.LaunchArgs, logPath), "started", pid)
	toastOK(sse, fmt.Sprintf("launched %s (PID %d) → %s", sig.LaunchScript, pid, logPath))
}

// spawnTraining validates + launches an allowlisted training script detached,
// logging to dashboard2/launches/. Shared by manual launch and the queue. No
// shell is used (args are separate argv), so there is no metacharacter injection.
func (s *Server) spawnTraining(script, argStr string) (int, string, error) {
	script = strings.TrimSpace(script)
	if !sysmon.AllowedScript(script) {
		return 0, "", fmt.Errorf("%s is not an allowlisted training script", script)
	}
	scriptPath := filepath.Join(s.cfg.RepoRoot, filepath.Base(script))
	if _, err := os.Stat(scriptPath); err != nil {
		return 0, "", fmt.Errorf("script not found: %s", scriptPath)
	}
	args := strings.Fields(argStr)
	for _, a := range args {
		if strings.ContainsRune(a, 0) {
			return 0, "", fmt.Errorf("invalid argument")
		}
	}
	py := filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python")
	logDir := filepath.Join(s.cfg.RepoRoot, "dashboard2", "launches")
	_ = os.MkdirAll(logDir, 0o755)
	logPath := filepath.Join(logDir, fmt.Sprintf("%s.%d.log", filepath.Base(script), time.Now().Unix()))
	logf, err := os.Create(logPath)
	if err != nil {
		return 0, "", err
	}
	cmd := exec.Command(py, append([]string{filepath.Base(script)}, args...)...)
	cmd.Dir = s.cfg.RepoRoot
	cmd.Stdout = logf
	cmd.Stderr = logf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		logf.Close()
		return 0, "", err
	}
	pid := cmd.Process.Pid
	go func() { _ = cmd.Wait(); logf.Close() }()
	return pid, logPath, nil
}

// ---- helpers ----

func splitTags(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' })
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

func quoteAll(items []string) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = `"` + strings.ReplaceAll(it, `"`, `\"`) + `"`
	}
	return out
}
