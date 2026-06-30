// Package alerts watches live training runs for divergence/health problems
// (gnorm spikes, NaN/skip storms, codec collapse, ppl regression, throughput
// cliffs, stalls) and records them. Optionally auto-stops a run on a critical
// alert (opt-in — never kills a run unless the user enabled it).
package alerts

import (
	"context"
	"encoding/json"
	"fmt"
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
)

type Detector struct {
	db       *db.DB
	sampler  *sysmon.Sampler
	runsDir  string
	interval time.Duration

	baselinePPL float64
	autoStop     atomic.Bool

	mu         sync.Mutex
	lastRaised map[string]float64
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
	for _, p := range procs {
		if p.RunName == "" {
			continue
		}
		d.scanRun(p)
	}
}

func (d *Detector) scanRun(p sysmon.Proc) {
	runID, ok, err := d.db.RunID(p.RunName)
	if err != nil || !ok {
		return
	}
	stats, err := d.db.RecentTrainStats(runID, 50)
	if err != nil {
		return
	}

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
	if d.baselinePPL > 0 {
		if k, ok, _ := d.db.RunKPIsByName(p.RunName); ok && k.PPL != nil && *k.PPL > pplRegressRatio*d.baselinePPL {
			d.raise(p, "ppl_regress", "warn", stats.LastStep,
				fmt.Sprintf("eval ppl %.1f is %.1fx the original baseline %.1f", *k.PPL, *k.PPL/d.baselinePPL, d.baselinePPL))
		}
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

	// anti_grokking_collapse: a held-out metric regressing from its own best while
	// training keeps improving — late-stage "un-grokking" (distinct from ppl_regress,
	// which compares to the original-model baseline, not the run's own minimum).
	if es, eerr := d.db.RecentEvalStats(runID, 30); eerr == nil && es.N >= 3 {
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
