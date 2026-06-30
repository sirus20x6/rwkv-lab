package server

import (
	"log"
	"net/http"
	"path/filepath"
	"time"

	"github.com/starfederation/datastar-go/datastar"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

const streamInterval = time.Second

// handleStream is the long-lived Datastar SSE: it pushes the system header, the
// sidebar run list, and the selected run's header + KPI signals every second.
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	// Read this tab's id BEFORE creating the SSE (ReadSignals consumes the request).
	// Captured once for the connection's lifetime so this viewer's selection is stable.
	var sig struct {
		TabID string `json:"tabId"`
	}
	_ = datastar.ReadSignals(r, &sig)
	tabID := sig.TabID

	sse := datastar.NewSSE(w, r)
	s.pushTick(sse, tabID) // immediate first paint
	t := time.NewTicker(streamInterval)
	defer t.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-t.C:
			if sse.IsClosed() {
				return
			}
			s.pushTick(sse, tabID)
		}
	}
}

func (s *Server) pushTick(sse *datastar.ServerSentEventGenerator, tabID string) {
	now := float64(time.Now().UnixNano()) / 1e9
	snap := s.sampler.Latest()

	summaries, err := s.db.RunSummaries(now)
	if err != nil {
		log.Printf("[stream] run summaries: %v", err)
		return
	}
	procByRun := procIndex(snap.Procs)

	// System header (morph each element by id).
	_ = sse.PatchElements(renderSysGPUs(snap.GPUs))
	_ = sse.PatchElements(renderSysHost(snap.Host))
	_ = sse.PatchElements(renderSysProc(snap.Procs))
	// Sidebar run list.
	_ = sse.PatchElements(renderRunList(summaries, procByRun, now))
	// Global alerts banner (+ auto-stop toggle).
	if active, err := s.db.ActiveAlerts(20); err == nil {
		_ = sse.PatchElements(renderAlerts(active, s.autoStopOn()))
	}
	// Whole-model conversion map.
	_ = sse.PatchElements(renderConvBoard(s.scanBoard(summaries, snap.Procs)))
	// Launch queue.
	if q, err := s.db.ActiveQueue(); err == nil {
		_ = sse.PatchElements(renderQueue(q, s.queueAuto.Load(), len(snap.Procs) == 0))
	}

	// runVersions: name -> latest step (drives Pixi incremental append).
	versions := make(map[string]int64, len(summaries))
	for _, su := range summaries {
		if su.LatestStep != nil {
			versions[su.Name] = *su.LatestStep
		}
	}

	signals := map[string]any{
		"now":         time.Now().Format("15:04:05"),
		"runVersions": versions,
	}

	sel := s.selectedFor(tabID)
	if sel != "" {
		// Live header for the selected run (incl. authoritative best/ checkpoint).
		runDir := filepath.Join(s.cfg.RunsDir, sel)
		if sum, ok := findSummary(summaries, sel); ok {
			var proc *sysmon.Proc
			if p, has := procByRun[sel]; has {
				proc = &p
			}
			_ = sse.PatchElements(renderRunHeader(sum, proc, readBest(runDir), now))
		}
		// LoopedRWKV residual-weight panel (live loop_rw.json).
		if lr, ok := readLoopRW(runDir); ok {
			_ = sse.PatchElements(renderLoopRW(lr))
		} else {
			_ = sse.PatchElements(emptyLoopRW())
		}
		// KPI strip values.
		if k, ok, _ := s.db.RunKPIsByName(sel); ok {
			signals["kpi"] = k
		}
		// Live-tuning overrides (desired vs applied) for the tuning panel.
		if controls, err := s.db.GetControls(sel); err == nil {
			_ = sse.PatchElements(renderControls(controls))
		}
		// Hidden element the Pixi glue observes for (run, version) changes.
		_ = sse.PatchElementf(`<div id="active-run" data-run="%s" data-v="%d" hidden></div>`,
			esc(sel), versions[sel])
	}

	_ = sse.MarshalAndPatchSignals(signals)
}

// handleRunSelect sets the global selected run and reveals the detail panel.
// The ongoing stream then renders that run's header + KPIs live.
func (s *Server) handleRunSelect(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	// Read tabId BEFORE NewSSE — this select binds to the requesting tab only.
	var sig struct {
		TabID string `json:"tabId"`
	}
	_ = datastar.ReadSignals(r, &sig)
	now := float64(time.Now().UnixNano()) / 1e9

	summaries, err := s.db.RunSummaries(now)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	sum, ok := findSummary(summaries, name)
	if !ok {
		http.Error(w, "no such run", http.StatusNotFound)
		return
	}
	s.setSelected(sig.TabID, name)

	sse := datastar.NewSSE(w, r)
	var proc *sysmon.Proc
	if p, has := procIndex(s.sampler.Latest().Procs)[name]; has {
		proc = &p
	}
	_ = sse.PatchElements(renderRunHeader(sum, proc, readBest(filepath.Join(s.cfg.RunsDir, name)), now))
	_ = sse.PatchElementf(`<div id="active-run" data-run="%s" data-v="%d" hidden></div>`,
		esc(name), latestStep(sum))
	notes, tagsJSON := s.db.RunMeta(name)
	_ = sse.MarshalAndPatchSignals(map[string]any{
		"selectedRun": name, "hasSel": true,
		"notes": notes, "tags": tagsCSV(tagsJSON),
	})
}

// ---- helpers ----

func procIndex(procs []sysmon.Proc) map[string]sysmon.Proc {
	m := make(map[string]sysmon.Proc, len(procs))
	for _, p := range procs {
		if p.RunName != "" {
			m[p.RunName] = p
		}
	}
	return m
}

func findSummary(summaries []db.RunSummary, name string) (db.RunSummary, bool) {
	for _, s := range summaries {
		if s.Name == name {
			return s, true
		}
	}
	return db.RunSummary{}, false
}

func latestStep(s db.RunSummary) int64 {
	if s.LatestStep != nil {
		return *s.LatestStep
	}
	return 0
}
