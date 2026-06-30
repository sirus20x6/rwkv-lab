// Package convboard builds a whole-model layer-conversion map: which layers are
// accepted into converted_layers_lib, which are actively converting, which have
// attempted runs, and which are still pending — with each layer's latest ppl /
// codec_rel. Read-only; the "accept to lib" promotion is surfaced as a command
// (the lib format is produced by the user's own assemble_looped.py surgery).
package convboard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"sync"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

// LayerStatus is one layer's row in the conversion map.
type LayerStatus struct {
	Layer     int      `json:"layer"`
	Status    string   `json:"status"` // accepted|converting|attempted|pending
	RunName   string   `json:"run_name"`
	PPL       *float64 `json:"ppl"`
	CodecRel  *float64 `json:"codec_rel"`
	LibSizeGB float64  `json:"lib_size_gb"`
	LibMtime  float64  `json:"lib_mtime"`
}

var libRe = regexp.MustCompile(`^L(\d+)\.pt$`)

// sidecar layer cache: run dir's latest config.json path -> (mtime, layer).
var (
	layerCacheMu sync.Mutex
	layerCache   = map[string]struct {
		mtime float64
		layer int
		ok    bool
	}{}
)

// Scan builds the per-layer map for nLayers layers. summaries are the run
// summaries the caller already computed (avoids a second RunSummaries query).
func Scan(database *db.DB, libDir, runsDir string, summaries []db.RunSummary, procs []sysmon.Proc, nLayers int) []LayerStatus {
	accepted := scanLib(libDir)                          // layer -> (sizeGB, mtime)
	runByLayer := scanRuns(runsDir, summaries)           // layer -> run name (latest)

	// layers actively converting (live proc whose run maps to a layer)
	converting := map[int]string{}
	for _, p := range procs {
		if p.RunName == "" {
			continue
		}
		if L, ok := runLayer(filepath.Join(runsDir, p.RunName)); ok {
			converting[L] = p.RunName
		}
	}

	out := make([]LayerStatus, 0, nLayers)
	for L := 0; L < nLayers; L++ {
		ls := LayerStatus{Layer: L, Status: "pending"}
		if run, ok := runByLayer[L]; ok {
			ls.RunName = run
			ls.Status = "attempted"
			if k, ok, _ := database.RunKPIsByName(run); ok {
				// prefer the run's best (min) eval ppl — the layer's quality, and
				// what best/best.json records — over the latest noisy eval.
				if k.BestPPL != nil {
					ls.PPL = k.BestPPL
				} else {
					ls.PPL = k.PPL
				}
			}
			if rid, ok, _ := database.RunID(run); ok {
				if st, err := database.RecentTrainStats(rid, 50); err == nil {
					ls.CodecRel = st.CodecRel
				}
			}
		}
		if lib, ok := accepted[L]; ok {
			ls.Status = "accepted"
			ls.LibSizeGB = lib.sizeGB
			ls.LibMtime = lib.mtime
		}
		if run, ok := converting[L]; ok {
			ls.Status = "converting"
			ls.RunName = run
		}
		out = append(out, ls)
	}
	return out
}

type libInfo struct {
	sizeGB float64
	mtime  float64
}

func scanLib(libDir string) map[int]libInfo {
	out := map[int]libInfo{}
	entries, err := os.ReadDir(libDir)
	if err != nil {
		return out
	}
	for _, e := range entries {
		m := libRe.FindStringSubmatch(e.Name())
		if m == nil {
			continue
		}
		L, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		out[L] = libInfo{
			sizeGB: float64(info.Size()) / (1024 * 1024 * 1024),
			mtime:  float64(info.ModTime().UnixNano()) / 1e9,
		}
	}
	return out
}

// scanRuns maps each layer to its most-recently-updated run.
func scanRuns(runsDir string, summaries []db.RunSummary) map[int]string {
	out := map[int]string{}
	best := map[int]float64{} // layer -> run last_update_ts
	for _, s := range summaries {
		L, ok := runLayer(filepath.Join(runsDir, s.Name))
		if !ok {
			continue
		}
		if s.LastUpdateTs >= best[L] {
			best[L] = s.LastUpdateTs
			out[L] = s.Name
		}
	}
	return out
}

// RunLayer is the exported form of runLayer (used by the leaderboard).
func RunLayer(runDir string) (int, bool) { return runLayer(runDir) }

// runLayer reads a run's latest sidecar config.json to find the converted layer
// (config.train_rwkv8_layers, a single int for convert_train runs). Cached by mtime.
func runLayer(runDir string) (int, bool) {
	cfgPath, mtime := latestSidecar(runDir)
	if cfgPath == "" {
		return 0, false
	}
	layerCacheMu.Lock()
	if c, ok := layerCache[cfgPath]; ok && c.mtime == mtime {
		layerCacheMu.Unlock()
		return c.layer, c.ok
	}
	layerCacheMu.Unlock()

	L, ok := parseSidecarLayer(cfgPath)
	layerCacheMu.Lock()
	layerCache[cfgPath] = struct {
		mtime float64
		layer int
		ok    bool
	}{mtime, L, ok}
	layerCacheMu.Unlock()
	return L, ok
}

func latestSidecar(runDir string) (string, float64) {
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return "", 0
	}
	var best string
	var bestMtime float64
	for _, e := range entries {
		if !e.IsDir() || len(e.Name()) < 5 || e.Name()[:5] != "step_" {
			continue
		}
		p := filepath.Join(runDir, e.Name(), "config.json")
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		mt := float64(info.ModTime().UnixNano()) / 1e9
		if mt >= bestMtime {
			bestMtime = mt
			best = p
		}
	}
	return best, bestMtime
}

func parseSidecarLayer(cfgPath string) (int, bool) {
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return 0, false
	}
	var sc struct {
		Config struct {
			TrainRWKV8Layers string `json:"train_rwkv8_layers"`
		} `json:"config"`
	}
	if json.Unmarshal(data, &sc) != nil {
		return 0, false
	}
	s := sc.Config.TrainRWKV8Layers
	if s == "" {
		return 0, false
	}
	// single int (convert_train writes str(args.layer)); take the first token if csv
	if i := indexByte(s, ','); i >= 0 {
		s = s[:i]
	}
	L, err := strconv.Atoi(s)
	if err != nil {
		return 0, false
	}
	return L, true
}

func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}
