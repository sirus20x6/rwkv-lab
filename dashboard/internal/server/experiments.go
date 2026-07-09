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

// known lever combos (mirrors experiment.LEVERS) — the builder's checkbox table. LMOnly levers need a
// real token future, so they're only selectable when an LM corpus (not a synthetic task) is chosen.
type leverDef struct {
	Name, Desc string
	LMOnly     bool
}

var knownLevers = []leverDef{
	{"baseline", "single pass — no recurrent depth (the control)", false},
	{"loop2", "recurrent depth ×2 — 2 weight-tied refinement passes", false},
	{"loop3", "recurrent depth ×3 — 3 weight-tied refinement passes", false},
	{"loop4", "recurrent depth ×4 — 4 weight-tied refinement passes", false},
	{"loop3_hyper", "loop ×3 + hyper-connection lanes (extra loop capacity)", false},
	{"loop3_cart", "loop ×3 + CART contractive LTI anchor (bounds the deep loop)", false},
	{"loop3_deq", "loop ×3 + DEQ 1-step gradient (O(1) memory)", false},
	{"loop3_factor", "loop ×3 + factored head×channel gate", false},
	{"nextlat", "next-latent prediction aux — predicts h[t+d] in-sequence (training-only, no inference cost)", false},
	{"loop3_nextlat", "loop ×3 + next-latent prediction", false},
	{"top", "token-order prediction — lookahead window (LM only)", true},
	{"lmtp", "leap multi-token prediction (LM only)", true},
	{"bst", "belief-state forward+backward objective (LM only)", true},
	{"jtp", "joint multi-token prediction (LM only)", true},
}

const lmTask = "local-lm"

type taskDef struct{ Name, Desc string }

var knownTasks = []taskDef{
	{"recall", "associative retrieval (k→v)"},
	{"copy", "state capacity (reproduce a sequence)"},
	{"induction", "in-context pattern (A B … A → B)"},
	{lmTask, "LM: local code+docs corpus (enables top/lmtp/bst/jtp → leaderboard)"},
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
	// init / start-from: scratch, continue a pretrained model, or hand off to conversion
	b.WriteString(`<tr><td class="f-l">init</td><td><select data-bind-init>` +
		`<option value="scratch">from scratch</option>` +
		`<option value="g1g">continue · pretrained g1g</option>` +
		`<option value="resume">resume · saved run</option>` +
		`<option value="convert">convert · GDN→RWKV</option>` +
		`</select></td><td class="f-d">random init · continue a pretrained model · convert (opens in the conversion board)</td></tr>`)
	// resume checkpoint path — relevant only when init = resume (dimmed otherwise)
	b.WriteString(`<tr><td class="f-l">resume from</td>` +
		`<td><input type="text" data-bind-resume data-attr-disabled="$init !== 'resume'" placeholder="runs/&lt;name&gt;/ckpt.pt"></td>` +
		`<td class="f-d">checkpoint to continue [resume only]</td></tr>`)
	// budget: fixed step count or fixed wall-clock time (Karpathy-style rounds)
	b.WriteString(`<tr><td class="f-l">budget</td><td><select data-bind-budget>` +
		`<option value="steps">steps</option><option value="minutes">wall-clock (min)</option>` +
		`</select></td><td class="f-d">fixed step count, or fixed wall-clock time per run</td></tr>`)
	numField := func(label, sig, hint string, def int) {
		fmt.Fprintf(&b, `<tr><td class="f-l">%s</td><td><input type="number" data-bind-%s value="%d" min="1"></td>`+
			`<td class="f-d">%s</td></tr>`, label, sig, def, esc(hint))
	}
	// defaults target a ~350M from-scratch LM research run (d1024 · L18 · h64 ≈ 362M @ vocab 65536);
	// dial d_model/layers down for quick synthetic diagnostics.
	numField("amount", "amount", "step count — or minutes when budget = wall-clock", 20000)
	numField("len / pairs", "tasklen", "task difficulty — pairs (recall) / length (copy·induction) [synthetic]", 16)
	numField("context len", "ctxlen", "sequence length [LM mode]", 1024)
	numField("d_model", "dmodel", "model width (1024 ≈ 350M-class)", 1024)
	numField("layers", "nlayers", "model depth", 18)
	numField("head size", "headsize", "attention head width (1024/64 = 16 heads)", 64)
	// live model-size estimate: params ≈ 2·V·d + L·12.1·d² (V=65536), matched to our arch within ~0.5%
	b.WriteString(`<tr><td class="f-l">model size</td><td colspan="2" class="model-size" ` +
		`data-text="$init==='g1g' ? 'g1g ≈ 1.5B (dims fixed)' : ($task==='` + lmTask + `' ` +
		`? ((2*65536*(+$dmodel)+(+$nlayers)*12.1*(+$dmodel)**2)/1e6).toFixed(0)+'M params  (' + ((+$nlayers)*12.1*(+$dmodel)**2/1e6).toFixed(0)+'M non-embed · vocab 65536)' ` +
		`: ((+$nlayers)*12.1*(+$dmodel)**2/1e6).toFixed(0)+'M params  (synthetic · tiny vocab)')"></td></tr>`)
	numField("batch", "batch", "sequences per step", 16)
	// optimizer settings
	b.WriteString(`<tr><td class="f-l">optimizer</td><td><select data-bind-optimizer>` +
		`<option value="adamw">AdamW</option>` +
		`<option value="adamw8bit">AdamW 8-bit</option>` +
		`<option value="paged-adamw8bit">AdamW 8-bit (paged)</option>` +
		`<option value="muon">Muon (spectral)</option>` +
		`</select></td><td class="f-d">AdamW · 8-bit variants quantize the Adam moment states (bitsandbytes, ~75% ` +
		`less optimizer memory, ~fp32 quality; paged rides out OOM spikes) · Muon = Newton–Schulz on 2D weights, AdamW on embeds/norms</td></tr>`)
	strField := func(label, sig, hint, def string) {
		fmt.Fprintf(&b, `<tr><td class="f-l">%s</td><td><input type="text" data-bind-%s value="%s"></td>`+
			`<td class="f-d">%s</td></tr>`, label, sig, def, esc(hint))
	}
	strField("lr", "lr", "learning rate — AdamW ~6e-4 · Muon matrix ~0.02 (units differ)", "6e-4")
	strField("weight decay", "wd", "decoupled weight decay", "0.1")
	strField("warmup", "warmup", "warmup steps (0 = auto ≈5%)", "0")
	// fp8 compute — orthogonal to the optimizer choice, so always shown (not gated on Muon)
	b.WriteString(`<tr><td class="f-l"><label for="fp8"><code>fp8</code></label></td>` +
		`<td><input type="checkbox" id="fp8" data-bind-fp8></td>` +
		`<td class="f-d">run eligible Linear GEMMs in fp8 on the Blackwell/Hopper tensor cores ` +
		`(torchao Float8Linear; bf16 master weights kept, so the optimizer is unchanged). Throughput ` +
		`gain needs torch.compile; eager fp8 still trains correctly.</td></tr>`)
	b.WriteString(`<tr><td class="f-l"><label for="docompile"><code>compile</code></label></td>` +
		`<td><input type="checkbox" id="docompile" data-bind-docompile></td>` +
		`<td class="f-d">torch.compile the training forward — fuses the fp8 cast+GEMM for the real ` +
		`~2× speedup on Blackwell (one-time ~40s compile at step 0). Best paired with fp8.</td></tr>`)
	numField("gen block", "genblock", "synthetic: batches generated per launch — amortizes gen kernel launches (1 = off)", 1)
	// Muon variants — rows shown only when optimizer = muon; each variant is its own checkbox row
	moff := ` data-class-muon-off="$optimizer !== 'muon'"`
	mvRow := func(sig, name, desc string) {
		fmt.Fprintf(&b, `<tr%s><td class="f-l"><label for="%s"><code>%s</code></label></td>`+
			`<td><input type="checkbox" id="%s" data-bind-%s></td><td class="f-d">%s</td></tr>`,
			moff, sig, name, sig, sig, esc(desc))
	}
	mvRow("sm_mona", "Muon²", "MONA — momentum-orthogonalized adaptive update")
	mvRow("sm_second_moment", "Aurora", "Adam-style second moment on the orthogonalized update")
	mvRow("sm_rsav", "RSAV", "gradient-energy variance gate (two-pass, ξ-clamped scalar)")
	mvRow("sm_da_muon", "DA-Muon", "distance-aware adaptive step from the update radius")
	mvRow("sm_aro", "ARO", "approximate row orthogonalization (Sinkhorn iterations)")
	msf := func(label, sig, hint, def string) {
		fmt.Fprintf(&b, `<tr%s><td class="f-l">%s</td><td><input type="text" data-bind-%s value="%s"></td>`+
			`<td class="f-d">%s</td></tr>`, moff, label, sig, def, esc(hint))
	}
	msf("scale", "sm_scale", "Newton–Schulz update scale (0.4·√d amplifier — see LR units)", "0.4")
	msf("Muon^p", "sm_spectral_power", "spectral power p (0 = plain Muon)", "0.0")
	msf("DDC", "sm_ddc_strength", "dual-descent correction strength (0 = off)", "0.0")
	msf("NS steps", "sm_ns_steps", "Newton–Schulz iterations", "5")
	numField("seeds", "seeds", "runs averaged → error bars (1 for big-model research; ↑ for cheap synthetic A/Bs)", 1)
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
		gate := "" // LM-only levers: checkbox disabled (not selectable) unless the LM corpus is chosen
		if lv.LMOnly {
			gate = ` data-attr-disabled="$task !== '` + lmTask + `'"`
		}
		fmt.Fprintf(&b, `<tr><td class="lev-c"><input type="checkbox" id="lev_%s" data-bind-lev_%s%s%s></td>`+
			`<td class="lev-n"><label for="lev_%s"><code>%s</code></label></td>`+
			`<td class="lev-d">%s</td></tr>`, lv.Name, lv.Name, chk, gate, lv.Name, lv.Name, esc(lv.Desc))
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
	var sig map[string]any
	_ = datastar.ReadSignals(r, &sig) // MUST read the body before creating the SSE (it closes r.Body)
	if sig == nil {
		sig = map[string]any{}
	}
	sse := datastar.NewSSE(w, r)
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
	lmOnlyPicked := false
	for _, lv := range knownLevers {
		if lv.Name == "baseline" {
			continue
		}
		if b, _ := sig["lev_"+lv.Name].(bool); b {
			configs = append(configs, lv.Name)
			lmOnlyPicked = lmOnlyPicked || lv.LMOnly
		}
	}
	if len(configs) < 2 {
		toastErr(sse, "launch: check at least one lever to compare against baseline")
		return
	}
	budget := []string{"--steps", str("amount", "20000")} // fixed steps, or wall-clock minutes
	if str("budget", "steps") == "minutes" {
		budget = []string{"--minutes", str("amount", "10")}
	}
	optArgs := []string{"--optimizer", str("optimizer", "adamw"), "--lr", str("lr", "6e-4"),
		"--weight-decay", str("wd", "0.1")}
	if wu := str("warmup", "0"); wu != "" && wu != "0" {
		optArgs = append(optArgs, "--warmup", wu)
	}
	if str("optimizer", "adamw") == "muon" { // Muon-variant flags
		optArgs = append(optArgs, "--sm-scale", str("sm_scale", "0.4"),
			"--sm-spectral-power", str("sm_spectral_power", "0.0"),
			"--sm-ddc-strength", str("sm_ddc_strength", "0.0"), "--sm-ns-steps", str("sm_ns_steps", "5"))
		for _, t := range []struct{ Sig, Flag string }{
			{"sm_mona", "--sm-mona"}, {"sm_second_moment", "--sm-second-moment"},
			{"sm_rsav", "--sm-rsav"}, {"sm_da_muon", "--sm-da-muon"}, {"sm_aro", "--sm-aro"}} {
			if on, _ := sig[t.Sig].(bool); on {
				optArgs = append(optArgs, t.Flag, "1")
			}
		}
	}
	if on, _ := sig["fp8"].(bool); on { // fp8 compute — orthogonal to the optimizer
		optArgs = append(optArgs, "--fp8")
	}
	if on, _ := sig["docompile"].(bool); on { // torch.compile the train forward
		optArgs = append(optArgs, "--compile")
	}
	init := str("init", "scratch")
	if init == "convert" { // per-layer GDN→RWKV distillation — not a config sweep
		toast(sse, "conversion (GDN→RWKV) runs in the conversion board / queue — it's per-layer teacher distillation, not a compare-configs sweep")
		return
	}
	// LM path: an LM corpus task, OR a continuation (g1g / resume) which is inherently an LM.
	if task == lmTask || init == "g1g" || init == "resume" {
		args := append([]string{"-m", "rwkv_lab.config", "run-lm", "--levers", strings.Join(configs, ","),
			"--d-model", str("dmodel", "1024"), "--n-layers", str("nlayers", "18"), "--head-size", str("headsize", "64"),
			"--batch", str("batch", "16"), "--seq-len", str("ctxlen", "1024")}, budget...)
		args = append(args, optArgs...)
		note := "from scratch"
		if init == "g1g" {
			args = append(args, "--init-g1g", "models/rwkv7-g1g-1.5b.pth")
			note = "continued from g1g"
		} else if init == "resume" {
			rp := str("resume", "")
			if rp == "" {
				toastErr(sse, "resume: enter a checkpoint path (runs/<name>/ckpt.pt)")
				return
			}
			args = append(args, "--resume", rp)
			note = "resumed from " + rp
		}
		pid, err := s.spawnPy(args, "lm_experiment.log")
		if err != nil {
			toastErr(sse, "launch failed: "+err.Error())
			return
		}
		toast(sse, fmt.Sprintf("launched LM sweep [%s] · %s (pid %d) — runs appear in the leaderboard", strings.Join(configs, ","), note, pid))
		return
	}
	if lmOnlyPicked { // synthetic task can't supply the token future these objectives need
		toastErr(sse, "launch: top/lmtp/bst/jtp need the LM corpus — pick 'local-lm' as the task")
		return
	}
	args := append([]string{"-m", "rwkv_lab.experiment",
		"--task", task + ":" + str("tasklen", "16"), "--configs", strings.Join(configs, ","),
		"--seeds", str("seeds", "1"), "--d-model", str("dmodel", "1024"), "--n-layers", str("nlayers", "18"),
		"--head-size", str("headsize", "64"), "--batch", str("batch", "16")}, budget...)
	if gb := str("genblock", "1"); gb != "" && gb != "1" { // synthetic-only: amortize gen launches
		args = append(args, "--gen-block", gb)
	}
	args = append(args, optArgs...)
	pid, err := s.spawnPy(args, "exp_"+task+".log")
	if err != nil {
		toastErr(sse, "launch: "+err.Error())
		return
	}
	toast(sse, fmt.Sprintf("launched %s [%s] (pid %d) — expand again to see results", task, strings.Join(configs, ","), pid))
}

// spawnPy launches `python -m <module> …` detached, logging to runs/<logName>, and returns the pid.
func (s *Server) spawnPy(args []string, logName string) (int, error) {
	lf, err := os.Create(filepath.Join(s.cfg.RunsDir, logName))
	if err != nil {
		return 0, err
	}
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"), args...)
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
	cmd.Stdout, cmd.Stderr = lf, lf
	if err := cmd.Start(); err != nil {
		lf.Close()
		return 0, err
	}
	go func() { _ = cmd.Wait(); lf.Close() }()
	return cmd.Process.Pid, nil
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
