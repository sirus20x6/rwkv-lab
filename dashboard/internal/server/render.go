package server

import (
	"encoding/json"
	"fmt"
	"html"
	"math"
	"sort"
	"strings"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

// tagsCSV converts a stored JSON tags array into a comma-separated string for
// the tags input field. Falls back to "" on malformed input.
func tagsCSV(tagsJSON string) string {
	var tags []string
	if json.Unmarshal([]byte(tagsJSON), &tags) != nil {
		return ""
	}
	return strings.Join(tags, ", ")
}

// ---- small formatters ----

func esc(s string) string { return html.EscapeString(s) }

func meterCls(pct float64) string {
	switch {
	case pct >= 90:
		return "err"
	case pct >= 75:
		return "warn"
	default:
		return ""
	}
}

func clampPct(p float64) float64 {
	if p < 0 {
		return 0
	}
	if p > 100 {
		return 100
	}
	return p
}

func fmtAge(sec *float64) string {
	if sec == nil {
		return "—"
	}
	s := *sec
	switch {
	case s < 1:
		return "now"
	case s < 60:
		return fmt.Sprintf("%.0fs", s)
	case s < 3600:
		return fmt.Sprintf("%.0fm", s/60)
	case s < 86400:
		return fmt.Sprintf("%.1fh", s/3600)
	default:
		return fmt.Sprintf("%.1fd", s/86400)
	}
}

func fmtDur(s float64) string {
	h := int(s) / 3600
	m := (int(s) % 3600) / 60
	sec := int(s) % 60
	if h > 0 {
		return fmt.Sprintf("%dh%02dm", h, m)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, sec)
	}
	return fmt.Sprintf("%ds", sec)
}

// ---- system header ----

func renderSysGPUs(gpus []sysmon.GPU) string {
	if len(gpus) == 0 {
		return `<div id="sys-gpus" class="sys-block"><span class="muted">no GPU</span></div>`
	}
	var b strings.Builder
	b.WriteString(`<div id="sys-gpus" class="sys-block">`)
	for _, g := range gpus {
		pwr := ""
		if g.PowerW != nil {
			cap := ""
			if g.PowerCapW != nil {
				cap = fmt.Sprintf("/%.0f", *g.PowerCapW)
			}
			pwr = fmt.Sprintf(" · %.0f%sW", *g.PowerW, cap)
		}
		tempCls := ""
		if g.TempC >= 85 {
			tempCls = "err"
		} else if g.TempC >= 75 {
			tempCls = "warn"
		}
		memPct := 0.0
		if g.MemTotalGB > 0 {
			memPct = 100 * g.MemUsedGB / g.MemTotalGB
		}
		fmt.Fprintf(&b,
			`<span class="metric"><span class="lbl">GPU%d</span> <b>%d%%</b>`+
				`<span class="meter"><i class="%s" style="width:%.0f%%"></i></span>`+
				` <span class="lbl">vram</span> <b>%.1f/%.1fGB</b>`+
				`<span class="meter"><i class="%s" style="width:%.0f%%"></i></span>`+
				` <b class="%s">%d°C</b>%s</span>`,
			g.Index, g.UtilPct, meterCls(float64(g.UtilPct)), clampPct(float64(g.UtilPct)),
			g.MemUsedGB, g.MemTotalGB, meterCls(memPct), clampPct(memPct),
			tempCls, g.TempC, pwr)
	}
	b.WriteString(`</div>`)
	return b.String()
}

func renderSysHost(h sysmon.Host) string {
	meter := func(label, val string, pct float64) string {
		return fmt.Sprintf(
			`<span class="metric"><span class="lbl">%s</span> <b>%s</b>`+
				`<span class="meter"><i class="%s" style="width:%.0f%%"></i></span></span>`,
			label, val, meterCls(pct), clampPct(pct))
	}
	cpu := meter("CPU", fmt.Sprintf("%.0f%%", h.CPUPct), h.CPUPct)
	ram := meter("RAM", fmt.Sprintf("%.0f/%.0fGB", h.RAMUsedGB, h.RAMTotalGB), h.RAMPct)
	disk := meter("disk", fmt.Sprintf("%.0f/%.0fGB", h.DiskUsedGB, h.DiskTotalGB), h.DiskPct)
	loadS := fmt.Sprintf(`<span class="metric"><span class="lbl">load</span> <b>%.2f</b></span>`, h.Load1)
	return `<div id="sys-host" class="sys-block">` + cpu + ram + disk + loadS + `</div>`
}

func renderSysProc(procs []sysmon.Proc) string {
	if len(procs) == 0 {
		return `<div id="sys-proc" class="sys-block"><span class="muted">no training processes</span></div>`
	}
	var b strings.Builder
	b.WriteString(`<div id="sys-proc" class="sys-block">`)
	for _, p := range procs {
		// compact chip: liveness + name + runtime + log freshness; PID and RSS
		// stay one hover away in the tooltip
		fmt.Fprintf(&b,
			`<span class="proc-chip" title="PID %d · RSS %.1fGB"><span class="dot %s"></span><b>%s</b>`+
				`<span class="pc-meta">%s · log %s</span></span>`,
			p.PID, p.RSSGB, p.State, esc(p.RunName), fmtDur(p.RuntimeS), fmtAge(p.LogAgeS))
	}
	b.WriteString(`</div>`)
	return b.String()
}

// ---- sidebar run list ----

// effectiveStatus promotes a run to its live process's state when one is attached.
func effectiveStatus(s db.RunSummary, procByRun map[string]sysmon.Proc) string {
	if p, ok := procByRun[s.Name]; ok {
		if p.State == "healthy" || p.State == "stalling" {
			return p.State
		}
	}
	return s.Status
}

// applyEvalContractSummary removes aggregate claims that span an abandoned
// eval contract. Historical rows remain available in charts, but headline
// winner surfaces must follow the trainer's active best artifact.
func applyEvalContractSummary(s *db.RunSummary, best BestInfo) {
	if !best.ContractReset {
		return
	}
	s.BestTop1 = nil
	s.HasHorizons = false
	if !best.Exists {
		s.LatestPPL = nil
		s.LatestTop1 = nil
		s.BestPPL = nil
		s.BestPPLStep = nil
		return
	}
	ppl, step := best.PPL, best.Step
	s.BestPPL, s.BestPPLStep = &ppl, &step
}

func applyEvalContractKPIs(k *db.RunKPIs, best BestInfo) {
	if !best.ContractReset {
		return
	}
	k.BestTop1, k.BestTop1Step = nil, nil
	if !best.Exists {
		k.PPL, k.Top1 = nil, nil
		k.BestPPL, k.BestPPLStep = nil, nil
		return
	}
	ppl, step := best.PPL, best.Step
	k.BestPPL, k.BestPPLStep = &ppl, &step
}

func renderRunList(summaries []db.RunSummary, procByRun map[string]sysmon.Proc, nowTs float64) string {
	// Most-recently-updated first.
	sort.SliceStable(summaries, func(i, j int) bool {
		return summaries[i].LastUpdateTs > summaries[j].LastUpdateTs
	})
	var b strings.Builder
	b.WriteString(`<div id="run-list">`)
	if len(summaries) == 0 {
		b.WriteString(`<div class="empty">no runs ingested yet</div>`)
	}
	for _, s := range summaries {
		state := effectiveStatus(s, procByRun)
		step := "—"
		if s.LatestStep != nil {
			step = fmt.Sprintf("%d", *s.LatestStep)
		}
		age := nowTs - s.LastUpdateTs
		hor := ""
		if s.HasHorizons {
			hor = " · h=1…4"
		}
		// best eval ppl — the number selection decisions are actually made on
		best := ""
		if nz(s.BestPPL) {
			best = fmt.Sprintf(` · <span class="best">%.2f</span>`, *s.BestPPL)
		}
		// data-on:click sets the signal (for highlight) AND fetches the panel.
		// data-show does client-side filtering against the $runFilter signal.
		low := strings.ToLower(s.Name)
		fmt.Fprintf(&b,
			`<div class="run-item" data-run="%s" data-class="{active: $selectedRun==='%s'}" `+
				`data-show="$runFilter==='' || '%s'.includes($runFilter.toLowerCase())" `+
				`data-on:click="$selectedRun='%s'; @get('/api/run/%s')">`+
				`<span class="dot %s"></span>`+
				`<div class="run-info"><div class="run-name">%s</div>`+
				`<div class="run-meta">step %s%s · %s ago%s</div></div></div>`,
			esc(s.Name), jsName(s.Name), jsName(low), jsName(s.Name), urlName(s.Name),
			state, esc(s.Name), step, best, fmtAge(&age), hor)
	}
	b.WriteString(`</div>`)
	return b.String()
}

// ---- run header (one-shot on selection) ----

func renderRunHeader(s db.RunSummary, proc *sysmon.Proc, best BestInfo, nowTs float64) string {
	state := s.Status
	if proc != nil && (proc.State == "healthy" || proc.State == "stalling") {
		state = proc.State
	}
	label := map[string]string{
		"healthy": "ACTIVE", "stalling": "STALLING", "dead": "DEAD",
		"cold": "idle", "no_log": "no log",
	}[state]
	if label == "" {
		label = strings.ToUpper(state)
	}
	pidStr := ""
	progress := ""
	if proc != nil {
		pidStr = fmt.Sprintf(`<span class="pid">PID %d · %s</span>`, proc.PID, fmtDur(proc.RuntimeS))
		if proc.MaxSteps != nil && *proc.MaxSteps > 0 && s.LatestStep != nil {
			pct := clampPct(100 * float64(*s.LatestStep) / float64(*proc.MaxSteps))
			eta := ""
			if proc.RuntimeS > 0 && *s.LatestStep > 0 && pct < 100 {
				spt := proc.RuntimeS / float64(*s.LatestStep)
				eta = " · eta " + fmtDur(spt*float64(int64(*proc.MaxSteps)-*s.LatestStep))
			}
			progress = fmt.Sprintf(
				`<div class="progress-row"><div class="progress-bar"><i style="width:%.1f%%"></i></div>`+
					`<div class="progress-label">step %d / %d (%.1f%%)%s</div></div>`,
				pct, *s.LatestStep, *proc.MaxSteps, pct, eta)
		}
	}
	age := nowTs - s.LastUpdateTs
	bestStr := ""
	// Contract-changing resumes preserve old eval rows for history but quarantine
	// their checkpoint.  In that state the active best artifact is authoritative:
	// an abandoned branch's lower SQLite minimum must never be claimed as the
	// current winner.
	if best.ContractReset {
		if best.Exists {
			bestStr = fmt.Sprintf(` · <span class="best">★ best eval ppl %.3f @ step %d · restartable</span>`, best.PPL, best.Step)
		} else {
			bestStr = ` · <span class="best">eval contract reset · no winner yet</span>`
		}
	}
	// The checkpoint manifest is published durably before its eval log record.
	// If the process/host dies in that narrow interval, the restartable winner
	// must not be hidden indefinitely behind an older SQLite rollup.
	checkpointAhead := !best.ContractReset && best.Exists && (!nz(s.BestPPL) || best.PPL < *s.BestPPL-1e-9)
	if checkpointAhead {
		bestStr = fmt.Sprintf(` · <span class="best">★ checkpoint ppl %.3f @ step %d · restartable</span>`, best.PPL, best.Step)
	} else if !best.ContractReset && nz(s.BestPPL) {
		step := ""
		if s.BestPPLStep != nil {
			step = fmt.Sprintf(" @ step %d", *s.BestPPLStep)
		}
		restartable := ""
		if best.Exists && math.Abs(best.PPL-*s.BestPPL) < 1e-9 &&
			(s.BestPPLStep == nil || best.Step == *s.BestPPLStep) {
			restartable = " · restartable"
		}
		bestStr = fmt.Sprintf(` · <span class="best">★ best eval ppl %.3f%s%s</span>`,
			*s.BestPPL, step, restartable)
	}
	return fmt.Sprintf(
		`<div id="run-header"><div class="run-title-row">`+
			`<span class="dot %s"></span><span class="run-title-main">%s</span>`+
			`<span class="status-pill %s">%s</span>%s</div>`+
			`<div class="sub">%d train · %d eval · %d ckpt · updated %s ago%s</div>%s</div>`,
		state, esc(s.Name), state, label, pidStr,
		s.NTrain, s.NEval, s.NCkpt, fmtAge(&age), bestStr, progress)
}

// jsName escapes a run name for embedding inside a single-quoted JS string in a
// data-* expression. urlName escapes for a URL path segment.
func jsName(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `'`, `\'`, `"`, `&quot;`, "\n", "", "\r", "")
	return r.Replace(s)
}

func urlName(s string) string {
	// run names are filesystem dir names (no spaces/slashes in practice); escape
	// the few URL-significant chars defensively.
	r := strings.NewReplacer("%", "%25", "?", "%3F", "#", "%23", " ", "%20")
	return r.Replace(s)
}

// nz returns true for a finite, present pointer.
func nz(p *float64) bool { return p != nil && !math.IsNaN(*p) && !math.IsInf(*p, 0) }

// renderAlerts paints the global alerts banner (critical + warn) with an
// auto-stop toggle and a dismiss-all. Empty when there's nothing active.
func renderAlerts(active []db.Alert, autoStop bool) string {
	var b strings.Builder
	b.WriteString(`<div id="alerts-banner" class="alerts-banner">`)
	autoCls, autoLabel, autoTo := "", "auto-stop: off", "1"
	if autoStop {
		autoCls, autoLabel, autoTo = "on", "auto-stop: ON", "0"
	}
	// Toggle is always present so the user can arm auto-stop proactively.
	fmt.Fprintf(&b,
		`<div class="alerts-bar"><button class="autostop %s" data-on:click="@post('/api/autostop?on=%s')">%s</button>`,
		autoCls, autoTo, autoLabel)
	if len(active) > 0 {
		b.WriteString(`<button class="btn dismiss" data-on:click="@post('/api/alerts/ack')">dismiss all</button>`)
	}
	b.WriteString(`</div>`)
	for _, a := range active {
		fmt.Fprintf(&b,
			`<div class="alert %s"><span class="alert-kind">%s</span>`+
				`<span class="alert-run">%s</span><span class="alert-msg">%s</span>`+
				`<button class="alert-x" data-on:click="@post('/api/alerts/ack?id=%d')">×</button></div>`,
			esc(a.Severity), esc(a.Kind), esc(a.RunName), esc(a.Message), a.ID)
	}
	b.WriteString(`</div>`)
	return b.String()
}

// renderControls shows the active live-tuning overrides for the selected run,
// each as pending (amber, awaiting the trainer's poll) or applied (green, with
// the step it took effect).
func renderControls(controls []db.Control) string {
	var b strings.Builder
	b.WriteString(`<div id="controls-list" class="controls-list">`)
	if len(controls) == 0 {
		b.WriteString(`<span class="muted">no active overrides</span>`)
	}
	for _, c := range controls {
		if c.Pending {
			fmt.Fprintf(&b, `<span class="ctl-chip pending">%s=%g · pending</span>`, esc(c.Key), c.Value)
		} else {
			step := int64(0)
			if c.AppliedStep != nil {
				step = *c.AppliedStep
			}
			fmt.Fprintf(&b, `<span class="ctl-chip applied">%s=%g · applied@%d</span>`, esc(c.Key), c.Value, step)
		}
	}
	b.WriteString(`</div>`)
	return b.String()
}
