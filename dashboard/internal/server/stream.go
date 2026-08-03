package server

import (
	"net/http"
	"path/filepath"
	"time"

	"github.com/starfederation/datastar-go/datastar"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

const streamInterval = time.Second

type streamPatchStamp struct {
	html string
	at   time.Time
}

// streamPatchState is connection-local. Datastar morphing an unchanged region
// still parses all of its HTML and walks its DOM; the loop heatmap alone can be
// thousands of nodes. Suppress identical patches and rate-limit large sidebar
// churn while preserving the one-second scalar signal cadence.
type streamPatchState struct {
	patches map[string]streamPatchStamp
}

func newStreamPatchState() *streamPatchState {
	return &streamPatchState{patches: make(map[string]streamPatchStamp)}
}

func (s *streamPatchState) shouldPatch(key, html string, now time.Time,
	minimumInterval time.Duration) bool {
	previous, exists := s.patches[key]
	if exists && previous.html == html {
		return false
	}
	if exists && minimumInterval > 0 && now.Sub(previous.at) < minimumInterval {
		return false
	}
	s.patches[key] = streamPatchStamp{html: html, at: now}
	return true
}

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
	patches := newStreamPatchState()
	s.pushTick(sse, tabID, patches) // immediate first paint
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
			s.pushTick(sse, tabID, patches)
		}
	}
}

// pushTick renders one stream tick for one connection. All shared content
// (system header, run list, alerts, conversion map) comes from the global
// once-per-second snapshot — the only per-connection DB work is the small
// selected-run queries, so N open tabs no longer multiply the heavy aggregates.
func (s *Server) pushTick(sse *datastar.ServerSentEventGenerator, tabID string,
	patches *streamPatchState) {
	snap := s.latestTick()
	if snap == nil {
		return // refreshLoop hasn't produced the first snapshot yet
	}

	now := time.Now()
	patch := func(key, html string, minimumInterval time.Duration) {
		if patches.shouldPatch(key, html, now, minimumInterval) {
			_ = sse.PatchElements(html)
		}
	}

	// Small telemetry regions may change every second. The full run list embeds
	// status for every historical run and is intentionally capped at 5 seconds.
	patch("sys-gpus", snap.sysGPUs, 0)
	patch("sys-host", snap.sysHost, 0)
	patch("sys-proc", snap.sysProc, 0)
	patch("run-list", snap.runList, 15*time.Second)
	// Global alerts banner.
	if snap.alerts != "" {
		patch("alerts", snap.alerts, 0)
	}
	// Whole-model conversion map.
	patch("conv", snap.conv, 0)
	signals := map[string]any{
		"now":         time.Now().Format("15:04:05"),
		"runVersions": snap.versions,
	}

	// Open the newest run on a tab's first visit.  Previously the stream filled
	// the sidebar but left the entire detail pane hidden until an explicit click,
	// which made a healthy dashboard look empty.  Also send the authoritative
	// selection on every tick so a browser refresh restores that tab's view.
	sel := s.selectedFor(tabID)
	if _, ok := findSummary(snap.summaries, sel); !ok {
		if len(snap.summaries) > 0 {
			sel = snap.summaries[0].Name
			s.setSelected(tabID, sel)
		} else {
			sel = ""
		}
	}
	if sel != "" {
		signals["selectedRun"] = sel
		signals["hasSel"] = true
		// Live header for the selected run (incl. authoritative best/ checkpoint).
		runDir := filepath.Join(s.cfg.RunsDir, sel)
		if sum, ok := findSummary(snap.summaries, sel); ok {
			var proc *sysmon.Proc
			if p, has := snap.procByRun[sel]; has {
				proc = &p
			}
			patch("run-header:"+sel,
				renderRunHeader(sum, proc, snap.bestByRun[sel], snap.ts), 0)
		}
		// LoopedRWKV residual-weight panel (live loop_rw.json).
		if lr, ok := readLoopRW(runDir); ok {
			patch("looprw:"+sel, renderLoopRW(lr), 0)
		} else {
			patch("looprw:"+sel, emptyLoopRW(), 0)
		}
		// KPI strip values.
		if k, ok, _ := s.db.RunKPIsByName(sel); ok {
			applyEvalContractKPIs(&k, snap.bestByRun[sel])
			patch("kpis:"+sel, renderKPIs(k), 0)
			signals["kpi"] = k
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
	best := snap.bestByRun[name]
	_ = sse.PatchElements(renderRunHeader(sum, proc, best, now))
	_ = sse.PatchElementf(`<div id="active-run" data-run="%s" data-v="%d" hidden></div>`,
		esc(name), runVersion(sum))
	notes, tagsJSON := s.db.RunMeta(name)
	signals := map[string]any{
		"selectedRun": name, "hasSel": true,
		"notes": notes, "tags": tagsCSV(tagsJSON),
	}
	// A run click is also a refresh operation. Patch its KPIs in this response
	// instead of leaving the old run's values visible until the next stream tick.
	if k, found, _ := s.db.RunKPIsByName(name); found {
		applyEvalContractKPIs(&k, best)
		_ = sse.PatchElements(renderKPIs(k))
		signals["kpi"] = k
	}
	_ = sse.MarshalAndPatchSignals(signals)
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

func runVersion(s db.RunSummary) int64 {
	return int64(s.LastUpdateTs * 1000)
}
