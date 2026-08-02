// Package alerts watches live training runs for divergence/health problems
// (gnorm spikes, NaN/skip storms, codec collapse, ppl regression, throughput
// cliffs, stalls) and records them. It is strictly observational: lifecycle
// and live-control mutations belong to TrainVM.
package alerts

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

const cooldown = 300.0 // seconds before re-raising the same (run,kind)

// Thresholds (deliberately conservative to avoid alert fatigue).
const (
	gnormCritical   = 1000.0
	skipFracWarn    = 0.25
	throughputRatio = 0.5
	codecRelWarn    = 0.40
	pplRegressRatio = 1.5
	stallSeconds    = 180.0
	minRows         = 10

	// grokking diagnostics
	memDeadRMS         = 1e-4 // ROSA/Engram injection RMS below this = path never activated
	memDeadMinStep     = 400  // only flag a dead path once it's had time to grok on
	memDeadMinSamples  = 10   // require persistent evidence, not one atypical batch
	memDeadFrac        = 0.90 // fraction of usable rows that must be near zero
	memDeadMinFinite   = 0.50 // ≥half the reported rows must be finite to judge at all
	codecRelMinSamples = 3    // codec_collapse is critical: never fire off one batch
	pplCollapseRatio   = 1.15 // held-out ppl risen >15% over its own best = collapse
	blockCollapseRatio = 1.15 // held-out block-MSE risen >15% over its own best
	antiGrokLRCool     = 0.5  // conservative operator recommendation on collapse

	// LoopedRWKV loop-gate steering (loop_max_rw rides eval records; the trainer
	// reports loop_lr_mult for a read-only health recommendation).
	loopStallRW      = 1e-3  // max|rw| still below this = the gates never opened
	loopStallMinStep = 800   // give warmup + momentum rebuild time before judging
	loopReleaseRW    = 0.01  // 10x stall threshold: gates clearly moving -> relax the boost
	loopPinRW        = 0.245 // legacy default; trainer now reports loop_pin_thr per run (scales with --loop-gate-cap)
	loopMultCap      = 30.0  // --loop-lr-mult help's fresh-conversion ceiling
)

// memoryPathVerdict is the tri-state health of one recall path's injection
// window. The middle state matters: injection stats are flag-gated
// (--log-grokking-metrics) on a deliberately coarse cadence, so a run can carry
// too few injection rows in its 50-TRAIN-ROW window to judge — a state that
// used to be indistinguishable from "healthy" because both produced silence.
type memoryPathVerdict int

const (
	memoryPathAbsent  memoryPathVerdict = iota // no injection rows at all: run has no such path
	memoryPathUnknown                          // rows exist but are too few / too non-finite to judge
	memoryPathAlive
	memoryPathDead
)

// memoryPathEvidence carries the verdict plus the counts behind it, so an
// insufficient-evidence alert can state exactly what it was missing.
type memoryPathEvidence struct {
	verdict memoryPathVerdict
	rows    int // rows in the window that reported an injection RMS
	finite  int // of those, rows whose value is a real number
	dead    int // of the finite rows, rows below memDeadRMS
}

// classifyMemoryPath judges a ROSA/Engram injection window.
//
// Non-finite values are dropped rather than counted as dead (injection_rms
// returns a/b, so NaN is reachable). Dropping alone is not enough: a window of
// 45 NaNs and 10 zeros would otherwise raise "persistently ~0" on 10/50 rows.
// So the finite rows must also be a majority of what was reported; below that
// the window is unknown, not dead.
func classifyMemoryPath(samples []float64) memoryPathEvidence {
	e := memoryPathEvidence{rows: len(samples)}
	if e.rows == 0 {
		return e // memoryPathAbsent
	}
	for _, value := range samples {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			continue
		}
		e.finite++
		if value < memDeadRMS {
			e.dead++
		}
	}
	if e.finite < memDeadMinSamples ||
		float64(e.finite) < memDeadMinFinite*float64(e.rows) {
		e.verdict = memoryPathUnknown
		return e
	}
	e.verdict = memoryPathAlive
	if float64(e.dead)/float64(e.finite) >= memDeadFrac {
		e.verdict = memoryPathDead
	}
	return e
}

// robustCodecRel reduces the codec_rel window to its median. codec_collapse is
// a CRITICAL stats-driven alert, so it must not fire off one atypical batch —
// the same failure the memory-path
// checks were hardened against. Reports ok=false when the window is too thin to
// be robust; the alert then stays silent rather than trusting a single row.
func robustCodecRel(window []float64) (float64, bool) {
	finite := make([]float64, 0, len(window))
	for _, value := range window {
		if !math.IsNaN(value) && !math.IsInf(value, 0) {
			finite = append(finite, value)
		}
	}
	if len(finite) < codecRelMinSamples {
		return 0, false
	}
	sort.Float64s(finite)
	return finite[len(finite)/2], true
}

type Detector struct {
	db       *db.DB
	sampler  *sysmon.Sampler
	runsDir  string
	interval time.Duration

	baselinePPL float64

	mu         sync.Mutex
	lastRaised map[string]float64
	suspended  map[string]string // run -> cause while stats-driven monitoring is suspended
}

// evalContractReset is the trainer's durable statement that rows at or before
// Step (and any older abandoned-branch rows still in SQLite) cannot be used to
// judge the active model contract. PublishedTS is the receipt's filesystem
// publication time; ingested train/eval rows carry their source log mtime.
type evalContractReset struct {
	Step        int64
	PublishedTS float64
}

func New(database *db.DB, sampler *sysmon.Sampler, runsDir string, interval time.Duration) *Detector {
	if interval <= 0 {
		interval = 10 * time.Second
	}
	d := &Detector{
		db: database, sampler: sampler, runsDir: runsDir, interval: interval,
		lastRaised: map[string]float64{},
		suspended:  map[string]string{},
	}
	d.baselinePPL = loadBaselinePPL(runsDir)
	return d
}

func (d *Detector) Run(ctx context.Context) {
	t := time.NewTicker(d.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			d.scan()
		}
	}
}

func (d *Detector) scan() {
	procs := d.sampler.Latest().Procs
	names := make([]string, 0, len(procs))
	for _, p := range procs {
		if p.RunName != "" {
			names = append(names, p.RunName)
		}
	}
	_ = d.db.ResolveInactiveStalls(names)
	statsByRun, err := d.db.RecentTrainStatsByName(names, 50)
	if err != nil {
		return
	}
	pplByRun := map[string]*float64{}
	if summaries, err := d.db.RunSummaries(float64(time.Now().UnixNano()) / 1e9); err == nil {
		for _, summary := range summaries {
			pplByRun[summary.Name] = summary.LatestPPL
		}
	}
	for _, p := range procs {
		if p.RunName == "" {
			continue
		}
		row, ok := statsByRun[p.RunName]
		// Stall detection is driven purely by filesystem evidence (p.LogAgeS), so
		// it must run before the ingested-row evidence gate: a trainer that hangs
		// before its first log append (CUDA init deadlock etc.) is exactly the
		// failure the stall alert exists for, yet it produces zero rows.
		lastStep := int64(0)
		if ok {
			lastStep = row.Stats.LastStep
		}
		d.checkStall(p, lastStep)
		if !ok {
			continue
		}
		stats, reset, ready, cause := d.currentTrainStats(p, row.RunID, row.Stats)
		d.noteMonitoringGate(p, cause)
		if !ready {
			// A newly started process can be visible before its first log append,
			// while SQLite still describes the prior process/branch. Health actions
			// must fail closed until this process has published current evidence.
			continue
		}
		d.scanRun(p, row.RunID, stats, pplByRun[p.RunName], reset)
	}
}

// checkStall raises the stall alert for a live process whose log has gone
// quiet. Runs independently of the ingested-evidence gate (see scan).
func (d *Detector) checkStall(p sysmon.Proc, step int64) {
	if p.LogAgeS != nil && *p.LogAgeS > stallSeconds {
		d.raise(p, "stall", "warn", step,
			fmt.Sprintf("no log update for %.0fs while process alive (possible hang)", *p.LogAgeS))
		return
	}
	// Stall alerts describe a live condition, not a permanent warning. Once the
	// same trainer resumes appending, retire its banner entry automatically; the
	// historical event remains available in the run timeline.
	_ = d.db.ResolveAlerts(p.RunName, "stall")
}

// noteMonitoringGate makes fail-closed monitoring failures visible. A malformed
// receipt or a stats query error suppresses all stats-driven alerts for the run
// (fail closed), which would otherwise be silent and indefinite — so entering
// that state raises a one-shot "monitoring_suspended" alert (once per run per
// cause) and recovering clears it so a later re-entry alerts again. The normal
// no-evidence-yet gate passes cause "" and never alerts.
func (d *Detector) noteMonitoringGate(p sysmon.Proc, cause string) {
	d.mu.Lock()
	if d.suspended == nil {
		d.suspended = map[string]string{}
	}
	prev := d.suspended[p.RunName]
	if cause == "" {
		delete(d.suspended, p.RunName)
		if prev != "" {
			delete(d.lastRaised, p.RunName+"|monitoring_suspended")
		}
	} else {
		d.suspended[p.RunName] = cause
	}
	d.mu.Unlock()
	if cause != "" && cause != prev {
		d.raise(p, "monitoring_suspended", "warn", 0,
			"health alerts suspended (fail closed): "+cause)
	}
}

// currentTrainStats returns only rows known to belong to the live process and,
// when present, the active eval contract. A malformed present receipt is
// authoritative but unusable, so it suppresses alerts rather than failing open
// onto retained history. suspendCause is non-empty for the abnormal fail-closed
// paths (malformed receipt, stats query error) that should be surfaced to the
// user; the ordinary no-current-evidence gate leaves it empty.
func (d *Detector) currentTrainStats(p sysmon.Proc, runID int64, fallback db.TrainStats) (stats db.TrainStats, reset *evalContractReset, ready bool, suspendCause string) {
	receipt, present, valid := readEvalContractReset(filepath.Join(d.runsDir, p.RunName))
	if present && !valid {
		return db.TrainStats{}, nil, false, "eval_contract_reset.json present but malformed/unreadable"
	}

	afterStep, afterTS := int64(-1), -1.0
	if valid {
		afterStep, afterTS = receipt.Step, receipt.PublishedTS
		reset = &receipt
	}
	// A receipt persists across ordinary process restarts. Independently fence
	// stale rows from a previous PID until the current PID appends to the log.
	// Known limitation: ingested rows carry their source log's batch-wide mtime,
	// so a watcher-backlog flush can stamp rows with a later ts than the append
	// that produced them — the ts fence here is only as precise as ingestion.
	if p.StartedTS > afterTS {
		afterTS = p.StartedTS
	}

	stats = fallback
	var err error
	if afterStep >= 0 || afterTS >= 0 {
		stats, err = d.db.RecentTrainStatsSince(runID, afterStep, afterTS, 50)
		if err != nil {
			return db.TrainStats{}, reset, false, "train stats query failed: " + err.Error()
		}
	}
	if stats.N == 0 || stats.LastStep <= afterStep || stats.LastTS <= afterTS {
		return db.TrainStats{}, reset, false, ""
	}
	return stats, reset, true, ""
}

func readEvalContractReset(runDir string) (receipt evalContractReset, present, valid bool) {
	path := filepath.Join(runDir, "eval_contract_reset.json")
	// Read first, then stat, so PublishedTS describes the content actually read
	// (a stat-then-read window lets the trainer's atomic replace pair an old
	// mtime with new content).
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return evalContractReset{}, false, false
		}
		return evalContractReset{}, true, false
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return evalContractReset{}, true, false
	}
	var raw struct {
		Schema  int      `json:"schema"`
		Reset   *bool    `json:"reset"`
		Step    *int64   `json:"step"`
		Reasons []string `json:"reasons"`
	}
	if json.Unmarshal(data, &raw) != nil || raw.Schema != 1 || raw.Reset == nil ||
		raw.Step == nil || *raw.Step < 0 || !*raw.Reset || len(raw.Reasons) == 0 {
		return evalContractReset{}, true, false
	}
	for _, reason := range raw.Reasons {
		if strings.TrimSpace(reason) == "" {
			return evalContractReset{}, true, false
		}
	}
	return evalContractReset{
		Step:        *raw.Step,
		PublishedTS: float64(info.ModTime().UnixNano()) / 1e9,
	}, true, true
}

func (d *Detector) evalStats(runID int64, reset *evalContractReset) (db.EvalStats, error) {
	if reset == nil {
		return d.db.RecentEvalStats(runID, 30)
	}
	return d.db.RecentEvalStatsSince(runID, reset.Step, reset.PublishedTS, 30)
}

func (d *Detector) scanRun(p sysmon.Proc, runID int64, stats db.TrainStats, latestPPL *float64, reset *evalContractReset) {
	// stall is checked in scan(), before the evidence gate — it must fire even
	// for a process that has never appended a row.
	if stats.N < minRows {
		return // not enough data for the rate-based checks yet
	}

	if stats.MaxGnorm > gnormCritical {
		d.raise(p, "gnorm_spike", "critical", stats.LastStep,
			fmt.Sprintf("gradient norm spiked to %.0f (last %d steps)", stats.MaxGnorm, stats.N))
	}
	if stats.SkipFrac > skipFracWarn {
		d.raise(p, "nan_rate", "warn", stats.LastStep,
			fmt.Sprintf("%.0f%% of recent steps skipped (non-finite loss)", 100*stats.SkipFrac))
	}
	if stats.MedTokPerSec > 0 && stats.LastTokPerSec > 0 &&
		stats.LastTokPerSec < throughputRatio*stats.MedTokPerSec {
		d.raise(p, "throughput_drop", "warn", stats.LastStep,
			fmt.Sprintf("throughput fell to %.0f tok/s (median %.0f)", stats.LastTokPerSec, stats.MedTokPerSec))
	}
	if median, ok := robustCodecRel(stats.CodecRelWindow); ok && median > codecRelWarn {
		d.raise(p, "codec_collapse", "critical", stats.LastStep,
			fmt.Sprintf("codec rel_rmse median %.3f over %d rows > %.2f — SMT/DMT targets likely garbage",
				median, len(stats.CodecRelWindow), codecRelWarn))
	}
	// memory_path_dead: a ROSA/Engram recall path that never activated (injection
	// RMS still ~0 well past warmup). Only fires when the run actually emits the
	// field — runs without ROSA/Engram report zero rows and are skipped.
	// memory_path_unmeasured is its explicit counterpart: rows exist, but too few
	// to judge, which must be visible rather than silently absent.
	if stats.LastStep > memDeadMinStep {
		var dead, unknown []string
		for _, path := range []struct {
			name   string
			window []float64
		}{{"ROSA", stats.RosaInjWindow}, {"Engram", stats.EngramInjWindow}} {
			evidence := classifyMemoryPath(path.window)
			switch evidence.verdict {
			case memoryPathDead:
				dead = append(dead, path.name)
			case memoryPathUnknown:
				unknown = append(unknown, fmt.Sprintf("%s (%d finite of %d rows)",
					path.name, evidence.finite, evidence.rows))
			}
		}
		if len(dead) > 0 {
			d.raise(p, "memory_path_dead", "warn", stats.LastStep,
				fmt.Sprintf("%s injection persistently ~0 on usable rows (≥%.0f%% RMS < %.0e) at step %d — recall path hasn't grokked on",
					strings.Join(dead, " & "), 100*memDeadFrac, memDeadRMS, stats.LastStep))
		}
		if len(unknown) > 0 {
			d.raise(p, "memory_path_unmeasured", "info", stats.LastStep,
				fmt.Sprintf("%s injection health is UNKNOWN, not healthy: need ≥%d finite rows (and ≥%.0f%% finite) in the last %d train rows — raise the --log-grokking-metrics cadence",
					strings.Join(unknown, " & "), memDeadMinSamples, 100*memDeadMinFinite, stats.N))
		} else {
			// Insufficient evidence describes a live condition, like stall:
			// once the window carries enough rows to judge, retire the banner.
			_ = d.db.ResolveAlerts(p.RunName, "memory_path_unmeasured")
		}
	}

	es, eerr := d.evalStats(runID, reset)
	// RunSummaries intentionally retains historical minima. After a contract
	// reset its latest PPL can still be from the abandoned branch until watcher
	// ingestion catches up, so only a scoped eval row may drive regression.
	if reset != nil {
		latestPPL = nil
		if eerr == nil && es.N > 0 && es.LastPPL > 0 {
			latestPPL = &es.LastPPL
		}
	}
	if d.baselinePPL > 0 && latestPPL != nil && *latestPPL > pplRegressRatio*d.baselinePPL {
		d.raise(p, "ppl_regress", "warn", stats.LastStep,
			fmt.Sprintf("eval ppl %.1f is %.1fx the original baseline %.1f", *latestPPL, *latestPPL/d.baselinePPL, d.baselinePPL))
	}

	// Read-only loop-gate diagnostics (LoopedRWKV residual_weight, surfaced as
	// loop_max_rw on eval records). The detector reports when an operator should
	// consider boosting, releasing, or cooling loop_lr_mult, but never writes a
	// trainer control. loop_live=0 means the multiplier is baked into the group
	// LR, so a live-control recommendation would be misleading and is omitted.
	if eerr == nil && es.N >= 1 && es.LastMaxRW != nil && (es.LastLoopLive == nil || *es.LastLoopLive != 0) {
		// The trainer-reported value is observational evidence only. An absent
		// value is treated as the neutral multiplier for message qualification.
		fallback := 1.0
		if es.LastLoopMult != nil {
			fallback = *es.LastLoopMult
		}
		cur := fallback
		pinThr := loopPinRW // legacy default; trainer reports the cap-scaled threshold
		if es.LastPinThr != nil {
			pinThr = *es.LastPinThr
		}
		// --loop-anneal-rw: the trainer cools the boost itself on a deterministic
		// schedule. Its ownership is surfaced in the diagnostic so the operator
		// does not fight that schedule with a second controller.
		annealed := es.LastLoopAnn != nil && *es.LastLoopAnn != 0
		if stats.LastStep > loopStallMinStep && *es.LastMaxRW < loopStallRW {
			next := math.Min(math.Max(cur, 1.0)*10.0, loopMultCap)
			d.raise(p, "loop_stall", "warn", stats.LastStep,
				fmt.Sprintf("loop gates still ~0 (max|rw| %.1e) at step %d — review the schedule and consider loop_lr_mult %.3g→%.3g through TrainVM",
					*es.LastMaxRW, stats.LastStep, cur, next))
		} else if *es.LastMaxRW >= pinThr && annealed {
			d.raise(p, "loop_pinned", "warn", stats.LastStep,
				fmt.Sprintf("loop gates beyond healthy regime (max|rw| %.3f ≥ %.3f); trainer anneal owns cooling (mult %.3g)",
					*es.LastMaxRW, pinThr, cur))
		} else if *es.LastMaxRW >= pinThr && cur > 1.0 {
			next := math.Max(cur*0.5, 1.0)
			d.raise(p, "loop_pinned", "warn", stats.LastStep,
				fmt.Sprintf("loop gates pinned (max|rw| %.3f ≥ %.3f) — consider loop_lr_mult %.3g→%.3g through TrainVM",
					*es.LastMaxRW, pinThr, cur, next))
		} else if *es.LastMaxRW >= loopReleaseRW && !annealed && cur > 1.0 {
			next := math.Max(cur*0.5, 1.0)
			d.raise(p, "loop_release", "info", stats.LastStep,
				fmt.Sprintf("loop gates moving (max|rw| %.3f ≥ %.2g) — consider releasing loop_lr_mult %.3g→%.3g through TrainVM",
					*es.LastMaxRW, loopReleaseRW, cur, next))
		}
	}

	// anti_grokking_collapse: a held-out metric regressing from its own best while
	// training keeps improving — late-stage "un-grokking" (distinct from ppl_regress,
	// which compares to the original-model baseline, not the run's own minimum).
	if eerr == nil && es.N >= 3 {
		trainImproving := stats.OldestLoss > 0 && stats.LastLoss < stats.OldestLoss
		pplCollapse := es.MinPPL > 0 && es.LastPPL > pplCollapseRatio*es.MinPPL
		blockCollapse := es.LastBlockVal != nil && es.MinBlockVal != nil &&
			*es.MinBlockVal > 0 && *es.LastBlockVal > blockCollapseRatio*(*es.MinBlockVal)
		if trainImproving && (pplCollapse || blockCollapse) {
			what, cur, best := "held-out ppl", es.LastPPL, es.MinPPL
			if blockCollapse {
				what, cur, best = "held-out block-MSE", *es.LastBlockVal, *es.MinBlockVal
			}
			// Surface a conservative recommendation without competing with the
			// trainer's autopilot or TrainVM's authoritative control boundary.
			d.raise(p, "anti_grokking_collapse", "warn", stats.LastStep,
				fmt.Sprintf("%s rose to %.4g (%.0f%% over best %.4g) while train still falling — consider lr_scale=%.2f through TrainVM; autopilot handles restore-best/reg",
					what, cur, 100*(cur/best-1), best, antiGrokLRCool))
		}
	}
}

// raise records an alert subject to cooldown. It never signals a process or
// writes a trainer control.
func (d *Detector) raise(p sysmon.Proc, kind, severity string, step int64, msg string) bool {
	key := p.RunName + "|" + kind
	now := float64(time.Now().UnixNano()) / 1e9
	d.mu.Lock()
	if last, ok := d.lastRaised[key]; ok && now-last < cooldown {
		d.mu.Unlock()
		return false
	}
	d.lastRaised[key] = now
	d.mu.Unlock()

	_, _ = d.db.InsertAlert(db.Alert{
		Ts: now, RunName: p.RunName, Kind: kind, Severity: severity, Message: msg, Step: step,
	})
	return true
}

func loadBaselinePPL(runsDir string) float64 {
	data, err := os.ReadFile(filepath.Join(runsDir, "_baseline.json"))
	if err != nil {
		return 0
	}
	var raw map[string]any
	if json.Unmarshal(data, &raw) != nil {
		return 0
	}
	if v, ok := raw["ppl"].(float64); ok {
		return v
	}
	return 0
}
