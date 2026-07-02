package server

import (
	"fmt"
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

// pushTick renders one stream tick for one connection. All shared content
// (system header, run list, alerts, conv map, queue) comes from the global
// once-per-second snapshot — the only per-connection DB work is the small
// selected-run queries, so N open tabs no longer multiply the heavy aggregates.
func (s *Server) pushTick(sse *datastar.ServerSentEventGenerator, tabID string) {
	snap := s.latestTick()
	if snap == nil {
		return // refreshLoop hasn't produced the first snapshot yet
	}

	// System header (morph each element by id).
	_ = sse.PatchElements(snap.sysGPUs)
	_ = sse.PatchElements(snap.sysHost)
	_ = sse.PatchElements(snap.sysProc)
	// Sidebar run list.
	_ = sse.PatchElements(snap.runList)
	// Global alerts banner (+ auto-stop toggle).
	if snap.alerts != "" {
		_ = sse.PatchElements(snap.alerts)
	}
	// Whole-model conversion map.
	_ = sse.PatchElements(snap.conv)
	// Launch queue.
	if snap.queue != "" {
		_ = sse.PatchElements(snap.queue)
	}
	// Launch-args history datalist (recent launches/enqueues).
	_ = sse.PatchElements(snap.launchHist)

	signals := map[string]any{
		"now":         time.Now().Format("15:04:05"),
		"runVersions": snap.versions,
	}

	sel := s.selectedFor(tabID)
	if sel != "" {
		// Live header for the selected run (incl. authoritative best/ checkpoint).
		runDir := filepath.Join(s.cfg.RunsDir, sel)
		if sum, ok := findSummary(snap.summaries, sel); ok {
			var proc *sysmon.Proc
			if p, has := snap.procByRun[sel]; has {
				proc = &p
			}
			_ = sse.PatchElements(renderRunHeader(sum, proc, readBest(runDir), snap.ts))
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
			esc(sel), snap.versions[sel])
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

	// Resolve against the shared snapshot (≤1s stale) — selection happens on
	// runs the user can already see, and this keeps clicks off the heavy path.
	snap := s.latestTick()
	if snap == nil {
		http.Error(w, "warming up", http.StatusServiceUnavailable)
		return
	}
	sum, ok := findSummary(snap.summaries, name)
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
	// Reset staged live-tune overrides on run switch (values staged for one run
	// must not silently carry to another) and surface the run's current config
	// values so the tuning inputs show what an override would replace.
	ctlReset := map[string]any{}
	ctlCur := map[string]any{}
	for k := range controlWhitelist {
		ctlReset[k] = ""
		ctlCur[k] = ""
	}
	if cfg := s.readSidecarConfig(name); cfg != nil {
		for k := range controlWhitelist {
			if v, ok := cfg[k]; ok && v != nil {
				ctlCur[k] = fmt.Sprint(v)
			}
		}
	}
	_ = sse.MarshalAndPatchSignals(map[string]any{
		"selectedRun": name, "hasSel": true,
		"notes": notes, "tags": tagsCSV(tagsJSON),
		"ctl": ctlReset, "ctlCur": ctlCur,
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
