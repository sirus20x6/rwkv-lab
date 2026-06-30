package server

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/starfederation/datastar-go/datastar"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

// queueManager reconciles running queue items (marking them done when their PID
// exits) and, ONLY when auto-start is opted in, launches the next queued run as
// soon as the GPU is free. Default is manual-advance so it never fights an
// external supervisor.
func (s *Server) queueManager(ctx context.Context) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.reconcileQueue()
			if s.queueAuto.Load() && s.gpuFree() {
				s.startNextInternal()
			}
		}
	}
}

func (s *Server) reconcileQueue() {
	running, err := s.db.RunningQueue()
	if err != nil {
		return
	}
	for _, q := range running {
		if q.PID == nil || !pidAlive(int(*q.PID)) {
			_ = s.db.MarkFinished(q.ID, "done", nowTs())
		}
	}
}

// gpuFree reports whether no training process is currently running.
func (s *Server) gpuFree() bool {
	return len(s.sampler.Latest().Procs) == 0
}

// startNextInternal launches the highest-priority queued run. Returns whether
// one started.
func (s *Server) startNextInternal() bool {
	next, ok, err := s.db.NextQueued()
	if err != nil || !ok {
		return false
	}
	pid, logPath, err := s.spawnTraining(next.Script, next.Args)
	if err != nil {
		_ = s.db.MarkFinished(next.ID, "failed", nowTs())
		s.db.LogAction(nowTs(), "queue_start", "", fmt.Sprintf(`{"id":%d,"script":%q}`, next.ID, next.Script), "failed: "+err.Error(), 0)
		return false
	}
	_ = s.db.MarkRunning(next.ID, pid, logPath, nowTs())
	s.db.LogAction(nowTs(), "queue_start", "", fmt.Sprintf(`{"id":%d,"script":%q,"args":%q}`, next.ID, next.Script, next.Args), "started", pid)
	return true
}

// ---- handlers ----

func (s *Server) handleEnqueue(w http.ResponseWriter, r *http.Request) {
	var sig struct {
		LaunchScript string `json:"launchScript"`
		LaunchArgs   string `json:"launchArgs"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	script := strings.TrimSpace(sig.LaunchScript)
	if !sysmon.AllowedScript(script) {
		toast(sse, "enqueue refused: "+script+" not allowlisted")
		return
	}
	id, err := s.db.Enqueue(script, sig.LaunchArgs, 0, nowTs())
	if err != nil {
		toast(sse, "enqueue failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "enqueue", "", fmt.Sprintf(`{"id":%d,"script":%q,"args":%q}`, id, script, sig.LaunchArgs), "queued", 0)
	toast(sse, fmt.Sprintf("queued #%d: %s %s", id, script, sig.LaunchArgs))
}

func (s *Server) handleStartNext(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	if !s.gpuFree() {
		toast(sse, "start-next: a training process is already running (GPU busy)")
		return
	}
	if s.startNextInternal() {
		toast(sse, "started next queued run")
	} else {
		toast(sse, "queue empty (nothing to start)")
	}
}

func (s *Server) handleCancelQueue(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)
	sse := datastar.NewSSE(w, r)
	ok, err := s.db.CancelQueued(id)
	if err != nil {
		toast(sse, "cancel failed: "+err.Error())
		return
	}
	if ok {
		s.db.LogAction(nowTs(), "queue_cancel", "", fmt.Sprintf(`{"id":%d}`, id), "canceled", 0)
		toast(sse, fmt.Sprintf("canceled queue #%d", id))
	} else {
		toast(sse, "cancel: item not queued (already running/done?)")
	}
}

func (s *Server) handleQueueAuto(w http.ResponseWriter, r *http.Request) {
	on := r.URL.Query().Get("on") == "1"
	sse := datastar.NewSSE(w, r)
	s.queueAuto.Store(on)
	s.db.LogAction(nowTs(), "queue_auto", "", "", boolStr(on), 0)
	if on {
		toast(sse, "queue auto-start ON — do NOT also run supervisor_night.sh (they'd double-spawn)")
	} else {
		toast(sse, "queue auto-start off (manual-advance)")
	}
}

// renderQueue paints the launch-queue panel: auto toggle + start-next + items.
func renderQueue(items []db.QueueItem, auto, gpuFree bool) string {
	var b strings.Builder
	b.WriteString(`<div id="queue-list">`)
	autoCls, autoLabel, autoTo := "", "auto: off", "1"
	if auto {
		autoCls, autoLabel, autoTo = "on", "auto: ON", "0"
	}
	startAttr := ""
	if !gpuFree {
		startAttr = ` disabled title="GPU busy"`
	}
	fmt.Fprintf(&b,
		`<div class="queue-bar"><button class="autostop %s" data-on:click="@post('/api/queue/auto?on=%s')">%s</button>`+
			`<button class="btn"%s data-on:click="@post('/api/queue/start-next')">▶ start next</button></div>`,
		autoCls, autoTo, autoLabel, startAttr)
	if len(items) == 0 {
		b.WriteString(`<div class="empty">queue empty</div>`)
	}
	for _, q := range items {
		pid := ""
		if q.PID != nil {
			pid = fmt.Sprintf(" · PID %d", *q.PID)
		}
		x := ""
		if q.Status == "queued" {
			x = fmt.Sprintf(`<button class="alert-x" data-on:click="@post('/api/queue/cancel?id=%d')">×</button>`, q.ID)
		}
		fmt.Fprintf(&b,
			`<div class="queue-item %s"><span class="q-status">%s</span>`+
				`<span class="q-cmd">%s %s%s</span>%s</div>`,
			esc(q.Status), esc(q.Status), esc(q.Script), esc(q.Args), pid, x)
	}
	b.WriteString(`</div>`)
	return b.String()
}

// pidAlive reports whether a PID exists (signal 0 probe).
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	return syscall.Kill(pid, 0) == nil
}
