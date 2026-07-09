package server

// Experiments panel: reads the rwkv-lab results registry (experiments.db, written by
// experiment.py / config.py) and renders a per-task ranked table (acc mean±std + significance,
// loop-gate engagement, params, FLOP/token, length-gen) plus an interactive builder — task
// dropdown, model number-pickers, seeds/steps, and lever checkboxes — that launches
// experiment.py with those parameters. Also lists experiments/*.yaml with one-click run.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/starfederation/datastar-go/datastar"
)

// known lever combos (mirrors experiment.LEVERS) — the builder's checkbox table.
type leverDef struct{ Name, Desc string }

var knownLevers = []leverDef{
	{"baseline", "single pass — no recurrent depth (the control)"},
	{"loop2", "recurrent depth ×2 — 2 weight-tied refinement passes"},
	{"loop3", "recurrent depth ×3 — 3 weight-tied refinement passes"},
	{"loop4", "recurrent depth ×4 — 4 weight-tied refinement passes"},
	{"loop3_hyper", "loop ×3 + hyper-connection lanes (extra loop capacity)"},
	{"loop3_cart", "loop ×3 + CART contractive LTI anchor (bounds the deep loop)"},
	{"loop3_deq", "loop ×3 + DEQ 1-step gradient (O(1) memory)"},
	{"loop3_factor", "loop ×3 + factored head×channel gate"},
	{"nextlat", "next-latent prediction aux — predicts h[t+d] in-sequence (training-only, no inference cost)"},
	{"loop3_nextlat", "loop ×3 + next-latent prediction"},
}

type taskDef struct{ Name, Desc string }

var knownTasks = []taskDef{
	{"recall", "associative retrieval (k→v)"},
	{"copy", "state capacity (reproduce a sequence)"},
	{"induction", "in-context pattern (A B … A → B)"},
}

func validTask(name string) bool {
	for _, t := range knownTasks {
		if t.Name == name {
			return true
		}
	}
	return false
}

type expRow struct {
	Config                        string
	Mean, Std                     float64 // acc
	Gate, ParamsM, FlopTok, Acc2x float64
	Seeds                         int
	Sha                           string
}

func m0(m map[string][]float64, k string) float64 {
	if v, ok := m[k]; ok && len(v) > 0 {
		return v[0]
	}
	return math.NaN()
}

func (s *Server) readRegistry() (map[string][]expRow, error) {
	path := filepath.Join(s.cfg.RepoRoot, "experiments.db")
	if _, err := os.Stat(path); err != nil {
		return nil, nil
	}
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro&_pragma=busy_timeout(2000)")
	if err != nil {
		return nil, err
	}
	defer db.Close()
	rows, err := db.Query(`SELECT task, config, seeds, git_sha, metrics_json, MAX(ts)
		FROM results GROUP BY task, config ORDER BY task`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string][]expRow{}
	for rows.Next() {
		var task, config, sha, mj string
		var seeds int
		var ts float64
		if rows.Scan(&task, &config, &seeds, &sha, &mj, &ts) != nil {
			continue
		}
		var m map[string][]float64
		if json.Unmarshal([]byte(mj), &m) != nil || len(m["acc"]) < 2 {
			continue
		}
		out[task] = append(out[task], expRow{Config: config, Mean: m["acc"][0], Std: m["acc"][1],
			Gate: m0(m, "gate"), ParamsM: m0(m, "params_m"), FlopTok: m0(m, "flop_per_tok"),
			Acc2x: m0(m, "acc_2x"), Seeds: seeds, Sha: sha})
	}
	return out, nil
}

func (s *Server) listConfigs() []string {
	var files []string
	entries, _ := os.ReadDir(filepath.Join(s.cfg.RepoRoot, "experiments"))
	for _, e := range entries {
		if n := e.Name(); strings.HasSuffix(n, ".yaml") || strings.HasSuffix(n, ".yml") || strings.HasSuffix(n, ".json") {
			files = append(files, "experiments/"+n)
		}
	}
	sort.Strings(files)
	return files
}

func fnum(v float64, dec int) string {
	if math.IsNaN(v) {
		return "—"
	}
	return strconv.FormatFloat(v, 'f', dec, 64)
}

func (s *Server) handleExperiments(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	reg, err := s.readRegistry()
	if err != nil {
		_ = sse.PatchElements(`<div id="experiments-body"><div class="empty">registry error: ` + esc(err.Error()) + `</div></div>`)
		return
	}
	var b strings.Builder
	b.WriteString(`<div id="experiments-body" class="exp-body">`)

	// --- interactive builder: task dropdown, model pickers, seeds/steps, lever checkboxes ---
	b.WriteString(`<div class="exp-build"><div class="exp-h">new experiment</div>`)
	b.WriteString(`<table class="field-tbl"><tr><td class="f-l">task</td><td><select data-bind-task>`)
	for _, t := range knownTasks {
		fmt.Fprintf(&b, `<option value="%s">%s — %s</option>`, t.Name, t.Name, esc(t.Desc))
	}
	b.WriteString(`</select></td><td class="f-d">the capability to probe</td></tr>`)
	numField := func(label, sig, hint string, def int) {
		fmt.Fprintf(&b, `<tr><td class="f-l">%s</td><td><input type="number" data-bind-%s value="%d" min="1"></td>`+
			`<td class="f-d">%s</td></tr>`, label, sig, def, esc(hint))
	}
	numField("len / pairs", "tasklen", "task difficulty — pairs (recall) / sequence length (copy·induction)", 16)
	numField("d_model", "dmodel", "model width", 256)
	numField("layers", "nlayers", "model depth", 4)
	numField("seeds", "seeds", "runs averaged → error bars + significance", 3)
	numField("steps", "steps", "training steps per run", 3000)
	b.WriteString(`</table>`)
	b.WriteString(`<div class="exp-levs"><div class="lev-h">configs to compare</div><table class="lev-tbl">`)
	for _, lv := range knownLevers {
		if lv.Name == "baseline" {
			// baseline is the significance reference — locked on, not an optional arm.
			fmt.Fprintf(&b, `<tr><td class="lev-c"><input type="checkbox" checked disabled></td>`+
				`<td class="lev-n"><code>%s</code></td>`+
				`<td class="lev-d">%s <span class="lev-ref">reference — always run</span></td></tr>`, lv.Name, esc(lv.Desc))
			continue
		}
		chk := ""
		if lv.Name == "loop3" {
			chk = " checked"
		}
		fmt.Fprintf(&b, `<tr><td class="lev-c"><input type="checkbox" id="lev_%s" data-bind-lev_%s%s></td>`+
			`<td class="lev-n"><label for="lev_%s"><code>%s</code></label></td>`+
			`<td class="lev-d">%s</td></tr>`, lv.Name, lv.Name, chk, lv.Name, lv.Name, esc(lv.Desc))
	}
	b.WriteString(`</table></div>`)
	b.WriteString(`<button class="btn" data-on:click="@post('/api/experiments/launch')">▶ run experiment</button>`)
	b.WriteString(`</div></div>`)

	// --- launchable config files ---
	b.WriteString(`<div class="exp-configs"><div class="exp-h">config files</div>`)
	for _, c := range s.listConfigs() {
		fmt.Fprintf(&b, `<div class="exp-cfg"><code>%s</code>`+
			`<button class="btn sm" data-on:click="@post('/api/experiments/run?cfg=%s')">run</button></div>`, esc(c), esc(c))
	}
	b.WriteString(`</div>`)

	// --- results by task: acc + gate + params + FLOP + length-gen + significance ---
	b.WriteString(`<div class="exp-results"><div class="exp-h">results (latest per config)</div>`)
	tasks := make([]string, 0, len(reg))
	for t := range reg {
		tasks = append(tasks, t)
	}
	sort.Strings(tasks)
	if len(tasks) == 0 {
		b.WriteString(`<div class="empty">no results yet — build one above or run a config</div>`)
	}
	for _, task := range tasks {
		rowsT := reg[task]
		sort.Slice(rowsT, func(i, j int) bool { return rowsT[i].Mean > rowsT[j].Mean })
		var base *expRow
		for i := range rowsT {
			if rowsT[i].Config == "baseline" {
				base = &rowsT[i]
			}
		}
		fmt.Fprintf(&b, `<div class="exp-task"><div class="exp-tname">%s</div><table class="exp-tbl">`, esc(task))
		b.WriteString(`<tr class="exp-hd"><td>config</td><td>acc</td><td>len-gen</td><td>gate</td><td>params</td><td>MF/tok</td><td>vs base</td></tr>`)
		for _, rw := range rowsT {
			delta := `<td></td>`
			if base != nil && rw.Config != "baseline" {
				d := rw.Mean - base.Mean
				sig := math.Abs(d) > (rw.Std + base.Std)
				cls := map[bool]string{true: "sig", false: "ns"}[sig]
				delta = fmt.Sprintf(`<td class="%s">Δ%+.3f %s</td>`, cls, d, map[bool]string{true: "✓", false: "·"}[sig])
			}
			gate := fnum(rw.Gate, 3)
			if !math.IsNaN(rw.Gate) && rw.Gate < 0.02 && rw.Config != "baseline" {
				gate += " ⚠"
			}
			mf := "—"
			if !math.IsNaN(rw.FlopTok) {
				mf = fnum(rw.FlopTok/1e6, 1)
			}
			fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td>%.3f±%.3f</td><td class="dim">%s</td>`+
				`<td class="dim">%s</td><td class="dim">%sM</td><td class="dim">%s</td>%s</tr>`,
				esc(rw.Config), rw.Mean, rw.Std, fnum(rw.Acc2x, 3), gate, fnum(rw.ParamsM, 2), mf, delta)
		}
		b.WriteString(`</table></div>`)
	}
	b.WriteString(`</div></div>`)
	_ = sse.PatchElements(b.String())
}

// handleLaunchExperiment reads the builder's Datastar signals and spawns experiment.py with them.
func (s *Server) handleLaunchExperiment(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	var sig map[string]any
	if json.NewDecoder(r.Body).Decode(&sig) != nil {
		toastErr(sse, "launch: bad request")
		return
	}
	str := func(k, def string) string {
		if v, ok := sig[k]; ok {
			return fmt.Sprintf("%v", v)
		}
		return def
	}
	task := str("task", "recall")
	if !validTask(task) {
		toastErr(sse, "launch: unknown task")
		return
	}
	configs := []string{"baseline"} // significance reference — always run
	for _, lv := range knownLevers {
		if lv.Name == "baseline" {
			continue
		}
		if b, _ := sig["lev_"+lv.Name].(bool); b {
			configs = append(configs, lv.Name)
		}
	}
	if len(configs) < 2 {
		toastErr(sse, "launch: check at least one lever to compare against baseline")
		return
	}
	args := []string{"-m", "rwkv_lab.experiment",
		"--task", task + ":" + str("tasklen", "16"),
		"--configs", strings.Join(configs, ","),
		"--seeds", str("seeds", "3"), "--steps", str("steps", "3000"),
		"--d-model", str("dmodel", "256"), "--n-layers", str("nlayers", "4")}
	logPath := filepath.Join(s.cfg.RunsDir, "exp_"+task+".log")
	lf, err := os.Create(logPath)
	if err != nil {
		toastErr(sse, "launch: "+err.Error())
		return
	}
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"), args...)
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
	cmd.Stdout, cmd.Stderr = lf, lf
	if err := cmd.Start(); err != nil {
		lf.Close()
		toastErr(sse, "launch failed: "+err.Error())
		return
	}
	go func() { _ = cmd.Wait(); lf.Close() }()
	toast(sse, fmt.Sprintf("launched %s [%s] (pid %d) — expand again to see results", task, strings.Join(configs, ","), cmd.Process.Pid))
}

func contains(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

// handleRunConfig launches `python -m rwkv_lab.config run <cfg>` (a config file under experiments/).
func (s *Server) handleRunConfig(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	cfg := r.URL.Query().Get("cfg")
	if cfg == "" || strings.Contains(cfg, "..") || !strings.HasPrefix(cfg, "experiments/") ||
		!(strings.HasSuffix(cfg, ".yaml") || strings.HasSuffix(cfg, ".yml") || strings.HasSuffix(cfg, ".json")) {
		toastErr(sse, "run refused: cfg must be experiments/*.yaml")
		return
	}
	if _, err := os.Stat(filepath.Join(s.cfg.RepoRoot, cfg)); err != nil {
		toastErr(sse, "run refused: "+cfg+" not found")
		return
	}
	base := strings.TrimSuffix(filepath.Base(cfg), filepath.Ext(cfg))
	logPath := filepath.Join(s.cfg.RunsDir, "config_"+base+".log")
	lf, err := os.Create(logPath)
	if err != nil {
		toastErr(sse, "run refused: "+err.Error())
		return
	}
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"), "-m", "rwkv_lab.config", "run", cfg)
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
	cmd.Stdout, cmd.Stderr = lf, lf
	if err := cmd.Start(); err != nil {
		lf.Close()
		toastErr(sse, "run failed: "+err.Error())
		return
	}
	go func() { _ = cmd.Wait(); lf.Close() }()
	toast(sse, fmt.Sprintf("launched %s (pid %d) — log %s", base, cmd.Process.Pid, filepath.Base(logPath)))
}
