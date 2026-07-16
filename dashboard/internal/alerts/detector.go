// Package alerts watches live training runs for divergence/health problems
// (gnorm spikes, NaN/skip storms, codec collapse, ppl regression, throughput
// cliffs, stalls) and records them. Optionally auto-stops a run on a critical
// alert (opt-in — never kills a run unless the user enabled it).
package alerts

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
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
	pplCollapseRatio   = 1.15 // held-out ppl risen >15% over its own best = collapse
	blockCollapseRatio = 1.15 // held-out block-MSE risen >15% over its own best
	antiGrokLRCool     = 0.5  // lr_scale written on collapse (cool in place, don't kill)

	// LoopedRWKV loop-gate steering (loop_max_rw rides eval records; the trainer
	// applies loop_lr_mult to the "rwkv_loop" param group per step).
	loopStallRW      = 1e-3  // max|rw| still below this = the gates never opened
	loopStallMinStep = 800   // give warmup + momentum rebuild time before judging
	loopReleaseRW    = 0.01  // 10x stall threshold: gates clearly moving -> relax the boost
	loopPinRW        = 0.245 // legacy default; trainer now reports loop_pin_thr per run (scales with --loop-gate-cap)
	loopMultCap      = 30.0  // --loop-lr-mult help's fresh-conversion ceiling
)

type Detector struct {
	db       *db.DB
	sampler  *sysmon.Sampler
	runsDir  string
	interval time.Duration

	baselinePPL float64
	autoStop    atomic.Bool

	mu         sync.Mutex
	lastRaised map[string]float64
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
	}
	d.baselinePPL = loadBaselinePPL(runsDir)
	return d
}

func (d *Detector) SetAutoStop(v bool) { d.autoStop.Store(v) }
func (d *Detector) AutoStop() bool     { return d.autoStop.Load() }

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
		if !ok {
			continue
		}
		stats, reset, ready := d.currentTrainStats(p, row.RunID, row.Stats)
		if !ready {
			// A newly started process can be visible before its first log append,
			// while SQLite still describes the prior process/branch. Health actions
			// must fail closed until this process has published current evidence.
			continue
		}
		d.scanRun(p, row.RunID, stats, pplByRun[p.RunName], reset)
	}
}

// currentTrainStats returns only rows known to belong to the live process and,
// when present, the active eval contract. A malformed present receipt is
// authoritative but unusable, so it suppresses alerts rather than failing open
// onto retained history.
func (d *Detector) currentTrainStats(p sysmon.Proc, runID int64, fallback db.TrainStats) (db.TrainStats, *evalContractReset, bool) {
	receipt, present, valid := readEvalContractReset(filepath.Join(d.runsDir, p.RunName))
	if present && !valid {
		return db.TrainStats{}, nil, false
	}

	afterStep, afterTS := int64(-1), -1.0
	var reset *evalContractReset
	if valid {
		afterStep, afterTS = receipt.Step, receipt.PublishedTS
		reset = &receipt
	}
	// A receipt persists across ordinary process restarts. Independently fence
	// stale rows from a previous PID until the current PID appends to the log.
	if p.StartedTS > afterTS {
		afterTS = p.StartedTS
	}

	stats := fallback
	var err error
	if afterStep >= 0 || afterTS >= 0 {
		stats, err = d.db.RecentTrainStatsSince(runID, afterStep, afterTS, 50)
		if err != nil {
			return db.TrainStats{}, reset, false
		}
	}
	if stats.N == 0 || stats.LastStep <= afterStep || stats.LastTS <= afterTS {
		return db.TrainStats{}, reset, false
	}
	return stats, reset, true
}

func readEvalContractReset(runDir string) (receipt evalContractReset, present, valid bool) {
	path := filepath.Join(runDir, "eval_contract_reset.json")
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return evalContractReset{}, false, false
		}
		return evalContractReset{}, true, false
	}
	if !info.Mode().IsRegular() {
		return evalContractReset{}, true, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
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

	// stall: process alive but log gone quiet
	if p.LogAgeS != nil && *p.LogAgeS > stallSeconds {
		d.raise(p, "stall", "warn", stats.LastStep,
			fmt.Sprintf("no log update for %.0fs while process alive (possible hang)", *p.LogAgeS))
	}
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
	if stats.CodecRel != nil && *stats.CodecRel > codecRelWarn {
		d.raise(p, "codec_collapse", "critical", stats.LastStep,
			fmt.Sprintf("codec rel_rmse %.3f > %.2f — SMT/DMT targets likely garbage", *stats.CodecRel, codecRelWarn))
	}
	// memory_path_dead: a ROSA/Engram recall path that never activated (injection
	// RMS still ~0 well past warmup). Only fires when the run actually emits the
	// field — runs without ROSA/Engram leave it nil and are skipped.
	if stats.LastStep > memDeadMinStep {
		var dead []string
		if stats.RosaInjRMS != nil && *stats.RosaInjRMS < memDeadRMS {
			dead = append(dead, "ROSA")
		}
		if stats.EngramInjRMS != nil && *stats.EngramInjRMS < memDeadRMS {
			dead = append(dead, "Engram")
		}
		if len(dead) > 0 {
			d.raise(p, "memory_path_dead", "warn", stats.LastStep,
				fmt.Sprintf("%s injection still ~0 (RMS < %.0e) at step %d — recall path hasn't grokked on",
					strings.Join(dead, " & "), memDeadRMS, stats.LastStep))
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

	// loop-gate steering (LoopedRWKV residual_weight, surfaced as loop_max_rw on eval
	// records). Full lifecycle: stalled ~0 well past warmup -> boost the live
	// loop_lr_mult so the zero-init gates get off the floor; clearly moving -> release
	// the boost back toward 1 (it exists for escape velocity, not steady state);
	// pinned at the cap -> cool it. residual_weight is UNBOUNDED in looped_rwkv, so
	// the boost must never be left hot longer than the escape window needs.
	// loop_live=0 (schedulefree) means the mult is baked into the group lr — live
	// steering is a no-op there, so write nothing rather than alert forever.
	if eerr == nil && es.N >= 1 && es.LastMaxRW != nil && (es.LastLoopLive == nil || *es.LastLoopLive != 0) {
		// Current effective mult: an explicit control row wins (it's what the trainer
		// polls next); else the trainer-reported value, which folds in the LAUNCH
		// --loop-lr-mult this side can't otherwise see — a run started at 30x must not
		// be "boosted" down to 10x, and its release/cool rules must still engage.
		fallback := 1.0
		if es.LastLoopMult != nil {
			fallback = *es.LastLoopMult
		}
		cur := d.currentControl(p.RunName, "loop_lr_mult", fallback)
		pinThr := loopPinRW // legacy default; trainer reports the cap-scaled threshold
		if es.LastPinThr != nil {
			pinThr = *es.LastPinThr
		}
		// --loop-anneal-rw: the trainer cools the boost itself on a deterministic
		// schedule. Round-1 gate A/B showed detector-side cooling is ingest/sampler-
		// laggy (one arm cooled at max|rw| 0.303, the other at 2.1) — so when the
		// trainer owns cooling, this side never writes loop_lr_mult controls for
		// pin/release; the stall BOOST still applies (the trainer folds a control
		// override into its anneal formula), and pin degrades to a watermark alert.
		annealed := es.LastLoopAnn != nil && *es.LastLoopAnn != 0
		if stats.LastStep > loopStallMinStep && *es.LastMaxRW < loopStallRW {
			next := math.Min(math.Max(cur, 1.0)*10.0, loopMultCap)
			if next > cur {
				if d.raise(p, "loop_stall", "warn", stats.LastStep,
					fmt.Sprintf("loop gates still ~0 (max|rw| %.1e) at step %d — boosting loop_lr_mult %.3g→%.3g",
						*es.LastMaxRW, stats.LastStep, cur, next)) {
					now := float64(time.Now().UnixNano()) / 1e9
					_ = d.db.SetControls(p.RunName, map[string]float64{"loop_lr_mult": next}, now)
				}
			}
		} else if *es.LastMaxRW >= pinThr && annealed {
			// watermark only: gates beyond the healthy regime, but cooling is the
			// trainer's job. Surfaces on the dashboard without fighting the anneal.
			d.raise(p, "loop_pinned", "warn", stats.LastStep,
				fmt.Sprintf("loop gates beyond healthy regime (max|rw| %.3f ≥ %.3f); trainer anneal owns cooling (mult %.3g)",
					*es.LastMaxRW, pinThr, cur))
		} else if *es.LastMaxRW >= pinThr && cur > 1.0 {
			next := math.Max(cur*0.5, 1.0)
			if d.raise(p, "loop_pinned", "warn", stats.LastStep,
				fmt.Sprintf("loop gates pinned (max|rw| %.3f ≥ %.3f) — cooling loop_lr_mult %.3g→%.3g",
					*es.LastMaxRW, pinThr, cur, next)) {
				now := float64(time.Now().UnixNano()) / 1e9
				_ = d.db.SetControls(p.RunName, map[string]float64{"loop_lr_mult": next}, now)
			}
		} else if *es.LastMaxRW >= loopReleaseRW && !annealed && cur > 1.0 {
			next := math.Max(cur*0.5, 1.0)
			if d.raise(p, "loop_release", "info", stats.LastStep,
				fmt.Sprintf("loop gates moving (max|rw| %.3f ≥ %.2g) — releasing loop_lr_mult %.3g→%.3g",
					*es.LastMaxRW, loopReleaseRW, cur, next)) {
				now := float64(time.Now().UnixNano()) / 1e9
				_ = d.db.SetControls(p.RunName, map[string]float64{"loop_lr_mult": next}, now)
			}
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
			// Recover, don't kill: warn (so auto-stop won't SIGINT) and write an
			// in-place LR cool to the control table. The trainer's --grok-autopilot
			// owns the structural recovery (restore-best + reg escalation).
			if d.raise(p, "anti_grokking_collapse", "warn", stats.LastStep,
				fmt.Sprintf("%s rose to %.4g (%.0f%% over best %.4g) while train still falling — auto-cooling lr_scale=%.2f; autopilot handles restore-best/reg",
					what, cur, 100*(cur/best-1), best, antiGrokLRCool)) {
				now := float64(time.Now().UnixNano()) / 1e9
				_ = d.db.SetControls(p.RunName, map[string]float64{"lr_scale": antiGrokLRCool}, now)
			}
		}
	}
}

// currentControl reads the current desired value of a live-tune key for a run
// (the steering target the detector escalates from), defaulting when unset.
func (d *Detector) currentControl(run, key string, def float64) float64 {
	cs, err := d.db.GetControls(run)
	if err != nil {
		return def
	}
	for _, c := range cs {
		if c.Key == key {
			return c.Value
		}
	}
	return def
}

// raise records an alert (subject to cooldown) and, for criticals with auto-stop
// enabled, SIGINTs the run's process.
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

	if severity == "critical" && d.autoStop.Load() && p.PID > 0 {
		if alive, _, _ := sysmon.VerifyTrainingPID(p.PID); alive {
			if err := syscall.Kill(int(p.PID), syscall.SIGINT); err == nil {
				d.db.LogAction(now, "auto_stop", p.RunName, `{"trigger":"`+kind+`"}`, "SIGINT sent (auto-stop)", int(p.PID))
				_, _ = d.db.InsertAlert(db.Alert{
					Ts: now, RunName: p.RunName, Kind: "auto_stop", Severity: "critical", Step: step,
					Message: fmt.Sprintf("auto-stop: SIGINT sent to PID %d after %s", p.PID, kind),
				})
			}
		}
	}
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
