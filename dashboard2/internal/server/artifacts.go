package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// BestInfo is the authoritative best checkpoint a run restarts from
// (runs/<run>/best/best.json, written atomically by convert_train on improve).
type BestInfo struct {
	Step   int64
	PPL    float64
	Exists bool
}

func readBest(runDir string) BestInfo {
	data, err := os.ReadFile(filepath.Join(runDir, "best", "best.json"))
	if err != nil {
		return BestInfo{}
	}
	var b struct {
		Step int64   `json:"step"`
		PPL  float64 `json:"ppl"`
	}
	if json.Unmarshal(data, &b) != nil {
		return BestInfo{}
	}
	// require the actual checkpoint too, so we only claim "restartable" when it is
	if _, err := os.Stat(filepath.Join(runDir, "best", "ckpt.pt")); err != nil {
		return BestInfo{}
	}
	return BestInfo{Step: b.Step, PPL: b.PPL, Exists: true}
}

// LoopRW mirrors runs/<run>/loop_rw.json (LoopedRWKV residual weights).
type LoopRW struct {
	LoopCount int     `json:"loop_count"`
	NLayers   int     `json:"n_layers"`
	NPinned   int     `json:"n_pinned"`
	MeanMaxRW float64 `json:"mean_max_rw"`
	Layers    []struct {
		Layer int       `json:"layer"`
		MaxRW float64   `json:"max_rw"`
		RW    []float64 `json:"rw"`
	} `json:"layers"`
}

func readLoopRW(runDir string) (LoopRW, bool) {
	data, err := os.ReadFile(filepath.Join(runDir, "loop_rw.json"))
	if err != nil {
		return LoopRW{}, false
	}
	var lr LoopRW
	if json.Unmarshal(data, &lr) != nil || len(lr.Layers) == 0 {
		return LoopRW{}, false
	}
	return lr, true
}

// renderLoopRW paints the LoopedRWKV residual-weight panel: one bar per converted
// layer (max|residual_weight|; 0 = the loop collapsed to a single pass), amber
// when pinned near the bf16 0.25 grid (well-used loop), blue otherwise.
func renderLoopRW(lr LoopRW) string {
	var b strings.Builder
	b.WriteString(`<div id="looprw-panel" class="panel"><div class="panel-title">loop usage · residual_weight per layer ` +
		fmt.Sprintf(`<span class="sub">mean %.3f · %d/%d pinned ~0.25 · %d passes</span></div>`,
			lr.MeanMaxRW, lr.NPinned, lr.NLayers, lr.LoopCount))
	rows := append([]struct {
		Layer int       `json:"layer"`
		MaxRW float64   `json:"max_rw"`
		RW    []float64 `json:"rw"`
	}{}, lr.Layers...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Layer < rows[j].Layer })
	b.WriteString(`<div class="looprw-bars">`)
	for _, r := range rows {
		// scale to a 0..0.3 visual range (residual weights live near 0..0.25)
		pct := r.MaxRW / 0.30 * 100
		if pct > 100 {
			pct = 100
		}
		cls := "lo"
		if r.MaxRW >= 0.245 {
			cls = "pinned"
		}
		passes := make([]string, len(r.RW))
		for i, v := range r.RW {
			passes[i] = fmt.Sprintf("%.3f", v)
		}
		fmt.Fprintf(&b,
			`<div class="looprw-row" title="rw=[%s]"><span class="lrw-l">L%d</span>`+
				`<span class="lrw-track"><i class="%s" style="width:%.1f%%"></i></span>`+
				`<span class="lrw-v">%.4f</span></div>`,
			strings.Join(passes, ", "), r.Layer, cls, pct, r.MaxRW)
	}
	b.WriteString(`</div></div>`)
	return b.String()
}

// emptyLoopRW hides the panel when a run has no loop_rw.json.
func emptyLoopRW() string {
	return `<div id="looprw-panel" class="panel hidden"></div>`
}
