package sysmon

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v4/process"
)

// Training entrypoints we recognize. Any python process whose cmdline names one
// of these is treated as a training job (mirrors v1, broadened to the current
// set of entrypoints).
var trainingScripts = []string{
	"convert_train.py", "distill_consolidate.py", "drive_isolation.py",
	"train_mla.py", "train_mla_engram.py", "rlvr_train.py", "rlvr_campaign.py",
	"recursive_improve.py", "adapter_recursive.py", "posttrain_train.py", "posttrain_campaign.py",
	"vision_train.py", "vision_cache.py",
}

var trainingModules = map[string]string{
	"convert_train.py":       "rwkv_lab.convert_train",
	"distill_consolidate.py": "rwkv_lab.distill_consolidate",
	"drive_isolation.py":     "rwkv_lab.drive_isolation",
	"train_mla.py":           "rwkv_lab.train_mla",
	"train_mla_engram.py":    "rwkv_lab.train_mla_engram",
	"rlvr_train.py":          "rwkv_lab.rlvr_train",
	"rlvr_campaign.py":       "rwkv_lab.rlvr_campaign",
	"recursive_improve.py":   "rwkv_lab.recursive_improve",
	"posttrain_train.py":     "rwkv_lab.posttrain_train",
	"posttrain_campaign.py":  "rwkv_lab.posttrain_campaign",
	"adapter_recursive.py":   "rwkv_lab.adapter_recursive",
	"vision_train.py":        "rwkv_lab.vision_train",
	"vision_cache.py":        "rwkv_lab.vision_cache",
}

// AllowedScript reports whether basename(path) is a recognized training
// entrypoint (the launch allowlist).
func AllowedScript(path string) bool {
	base := filepath.Base(path)
	for _, s := range trainingScripts {
		if base == s {
			return true
		}
	}
	return false
}

// ModuleForScript returns the Python module launched for an allowlisted script.
func ModuleForScript(path string) (string, bool) {
	mod, ok := trainingModules[filepath.Base(path)]
	return mod, ok
}

func matchedScript(cmdline []string) string {
	for _, a := range cmdline {
		base := filepath.Base(a)
		for _, s := range trainingScripts {
			if base == s || a == trainingModules[s] {
				return s
			}
		}
	}
	return ""
}

// argValue returns the value of --flag or --flag=value (first match).
func argValue(cmdline []string, names ...string) (string, bool) {
	want := map[string]bool{}
	for _, n := range names {
		want[n] = true
	}
	for i, a := range cmdline {
		if want[a] && i+1 < len(cmdline) {
			return cmdline[i+1], true
		}
		for n := range want {
			if strings.HasPrefix(a, n+"=") {
				return strings.TrimPrefix(a, n+"="), true
			}
		}
	}
	return "", false
}

// readProcs enumerates training processes and annotates them with run name +
// log-age liveness.
func readProcs(runsDir string) []Proc {
	procs, err := process.Processes()
	if err != nil {
		return nil
	}
	now := float64(time.Now().UnixNano()) / 1e9
	var out []Proc
	for _, p := range procs {
		cmdline, err := p.CmdlineSlice()
		if err != nil || len(cmdline) == 0 {
			continue
		}
		script := matchedScript(cmdline)
		if script == "" {
			continue
		}
		// Must actually be a python invocation (avoid matching an editor/grep).
		if !looksPython(cmdline) {
			continue
		}

		pr := Proc{PID: p.Pid, Script: script}

		if v, ok := argValue(cmdline, "--out-dir", "--out", "--output"); ok {
			pr.RunName = filepath.Base(v)
		}
		if script == "vision_cache.py" {
			if v, ok := argValue(cmdline, "--cache"); ok {
				pr.RunName = "cache: " + filepath.Base(v)
			} else {
				pr.RunName = "vision cache"
			}
		}
		if v, ok := argValue(cmdline, "--max-steps", "--steps"); ok {
			if n, err := strconv.Atoi(v); err == nil {
				pr.MaxSteps = &n
			}
		}
		if ct, err := p.CreateTime(); err == nil {
			pr.StartedTS = float64(ct) / 1000.0
			pr.RuntimeS = now - pr.StartedTS
		}
		if cp, err := p.CPUPercent(); err == nil {
			pr.CPUPct = cp
		}
		if mi, err := p.MemoryInfo(); err == nil && mi != nil {
			pr.RSSGB = float64(mi.RSS) / gb
		}
		if nt, err := p.NumThreads(); err == nil {
			pr.NumThreads = nt
		}

		if script == "vision_cache.py" {
			// Cache prefill has no train.jsonl, but it is still a live GPU job and
			// must appear in the header (and keep the launch queue from competing).
			pr.State = "healthy"
		} else {
			pr.LogAgeS, pr.State = liveness(runsDir, pr.RunName, now)
		}
		out = append(out, pr)
	}
	return out
}

// VerifyTrainingPID re-reads a live PID and confirms it is still a python
// training process (guards against PID reuse before we signal it). Returns the
// matched script and whether the cmdline references an instrumented copy.
func VerifyTrainingPID(pid int32) (ok bool, script string, instrumented bool) {
	p, err := process.NewProcess(pid)
	if err != nil {
		return false, "", false
	}
	cmdline, err := p.CmdlineSlice()
	if err != nil || len(cmdline) == 0 {
		return false, "", false
	}
	script = matchedScript(cmdline)
	if script == "" || !looksPython(cmdline) {
		return false, "", false
	}
	// "instrumented" == carries the SIGUSR1/SIGINT handlers (checkpoint-now / stop),
	// so signaling it won't kill the run. convert_train.py carries them natively now:
	// the dashboard/instrumented/ copy was reconciled into the root canonical trainer
	// (2026-06-30), so it's signal-capable regardless of path. Other trainers are only
	// signal-capable when run from a dashboard/instrumented/ copy.
	if script == "convert_train.py" {
		instrumented = true
	}
	for _, a := range cmdline {
		if strings.Contains(a, "dashboard/instrumented/") {
			instrumented = true
			break
		}
	}
	return true, script, instrumented
}

func looksPython(cmdline []string) bool {
	// Check the first couple of argv entries for a python interpreter.
	n := len(cmdline)
	if n > 2 {
		n = 2
	}
	for _, a := range cmdline[:n] {
		if strings.Contains(strings.ToLower(filepath.Base(a)), "python") {
			return true
		}
	}
	return false
}

// liveness returns age/state from the newest trainer heartbeat. Vision eval can
// spend minutes decoding at one step, so status.json keeps it visibly alive.
func liveness(runsDir, runName string, now float64) (*float64, string) {
	if runName == "" {
		return nil, "unknown"
	}
	newest := float64(0)
	for _, name := range []string{"train.jsonl", "status.json"} {
		if st, err := os.Stat(filepath.Join(runsDir, runName, name)); err == nil {
			modified := float64(st.ModTime().UnixNano()) / 1e9
			if modified > newest {
				newest = modified
			}
		}
	}
	if newest == 0 {
		return nil, "unknown"
	}
	age := now - newest
	switch {
	case age < HealthyWindow:
		return &age, "healthy"
	case age < StaleWindow:
		return &age, "stalling"
	default:
		return &age, "dead"
	}
}
