package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/starfederation/datastar-go/datastar"

	"trainboard/internal/convboard"
	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

func (s *Server) libDir() string {
	if s.cfg.LibDir != "" {
		return s.cfg.LibDir
	}
	return filepath.Join(s.cfg.RepoRoot, "Qwen3.5-9B-RWKV", "converted_layers_lib")
}

// nLayers returns the base-model layer count (config override → base config →
// default 32).
func (s *Server) nLayers() int {
	if s.cfg.NLayers > 0 {
		return s.cfg.NLayers
	}
	data, err := os.ReadFile(filepath.Join(s.cfg.RepoRoot, "Qwen3.5-9B-Base", "config.json"))
	if err == nil {
		var c struct {
			NumHiddenLayers int `json:"num_hidden_layers"`
			TextConfig      struct {
				NumHiddenLayers int `json:"num_hidden_layers"`
			} `json:"text_config"`
		}
		if json.Unmarshal(data, &c) == nil {
			if c.TextConfig.NumHiddenLayers > 0 {
				return c.TextConfig.NumHiddenLayers
			}
			if c.NumHiddenLayers > 0 {
				return c.NumHiddenLayers
			}
		}
	}
	return 32
}

// renderConvBoard paints the whole-model layer-conversion strip.
func renderConvBoard(layers []convboard.LayerStatus) string {
	var b strings.Builder
	counts := map[string]int{}
	for _, l := range layers {
		counts[l.Status]++
	}
	b.WriteString(`<div id="conv-board" class="conv-board"><div class="conv-head">conversion map · ` +
		fmt.Sprintf(`<span class="ok">%d accepted</span> · <span class="warn">%d converting</span> · %d attempted · %d pending`,
			counts["accepted"], counts["converting"], counts["attempted"], counts["pending"]) +
		`</div><div class="conv-cells">`)
	for _, l := range layers {
		ppl := "—"
		if l.PPL != nil {
			ppl = fmt.Sprintf("%.2f", *l.PPL)
		}
		codec := ""
		if l.CodecRel != nil {
			codec = fmt.Sprintf(" codec %.3f", *l.CodecRel)
		}
		title := fmt.Sprintf("L%d · %s · run %s · ppl %s%s", l.Layer, l.Status, l.RunName, ppl, codec)
		click := ""
		if l.RunName != "" {
			click = fmt.Sprintf(`data-on:click="$selectedRun='%s'; @get('/api/run/%s')"`, jsName(l.RunName), urlName(l.RunName))
		}
		fmt.Fprintf(&b, `<div class="conv-cell %s" title="%s" %s><span class="cl">L%d</span><span class="cp">%s</span></div>`,
			esc(l.Status), esc(title), click, l.Layer, ppl)
	}
	b.WriteString(`</div></div>`)
	return b.String()
}

// scanBoard computes the conversion map using the caller's summaries/procs.
func (s *Server) scanBoard(summaries []db.RunSummary, procs []sysmon.Proc) []convboard.LayerStatus {
	return convboard.Scan(s.db, s.libDir(), s.cfg.RunsDir, summaries, procs, s.nLayers())
}

// handleAcceptLayer records a layer-promotion candidate (provenance) and surfaces
// the source checkpoint. It does NOT write into converted_layers_lib — the lib
// format is produced by assemble_looped.py's state-dict surgery, so promotion
// stays a deliberate, user-run step.
func (s *Server) handleAcceptLayer(w http.ResponseWriter, r *http.Request) {
	layer, _ := strconv.Atoi(r.URL.Query().Get("layer"))
	run := r.URL.Query().Get("run")
	sse := datastar.NewSSE(w, r)
	if run == "" {
		toast(sse, "accept: no run for this layer")
		return
	}
	ckpt, step := latestCkpt(filepath.Join(s.cfg.RunsDir, run))
	var ppl *float64
	if k, ok, _ := s.db.RunKPIsByName(run); ok {
		ppl = k.PPL
	}
	libPath := filepath.Join(s.libDir(), fmt.Sprintf("L%02d.pt", layer))
	if err := s.db.AcceptLayer(layer, run, step, libPath, ppl, nowTs()); err != nil {
		toast(sse, "accept failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "accept_layer", run, fmt.Sprintf(`{"layer":%d,"ckpt":%q}`, layer, ckpt), "candidate recorded", 0)
	toast(sse, fmt.Sprintf("L%d candidate = %s → promote with assemble_looped.py (source: %s)", layer, run, ckpt))
}

// latestCkpt returns the best (preferred) or newest checkpoint path + step.
// convert_train now saves runs/<run>/best/ atomically on each eval improvement
// (resolve_best_ckpt), so prefer that when present.
func latestCkpt(runDir string) (string, int64) {
	if p := filepath.Join(runDir, "best", "ckpt.pt"); fileExists(p) {
		return p, -1 // step unknown for best/; provenance still records the path
	}
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return "", 0
	}
	var best string
	var bestStep int64 = -1
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "step_") {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimPrefix(e.Name(), "step_"), 10, 64)
		if err != nil {
			continue
		}
		p := filepath.Join(runDir, e.Name(), "ckpt.pt")
		if _, err := os.Stat(p); err != nil {
			continue
		}
		if n > bestStep {
			bestStep = n
			best = p
		}
	}
	if bestStep < 0 {
		return "", 0
	}
	return best, bestStep
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
