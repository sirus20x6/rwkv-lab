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
	{"seedchain", "Future-Seed — layer L's state scan starts from layer L−1's final state (no loops)", false},
	{"engram", "Engram LMB — token-SAM recall reads a learned table; gated injection + copy head (CPU recall per step)", false},
	{"deepembed", "DeepEmbed (RWKV-8) — per-layer per-token FFN gate, 1 + emb(ids); big sparse tables, ~free lookup", false},
	{"de_hidden", "DeepEmbed exact (BlinkDL v7a) — input-dependent bilinear gate on the FFN hidden, per-token r×r matrix", false},
	{"de_shift", "DeepEmbed exact + separate gate token-shift — the variant BlinkDL reported as 'very large'", false},
	{"de_full", "DeepEmbed exact + gate token-shift + emb-residual (global token embedding folded into the gate matrix)", false},
	{"top", "token-order prediction — lookahead window (LM only)", true},
	{"lmtp", "leap multi-token prediction (LM only)", true},
	{"bst", "belief-state forward+backward objective (LM only)", true},
	{"jtp", "joint multi-token prediction (LM only)", true},
}

const lmTask = "local-lm"
const blendTask = "blend-lm"
const blendMixTask = "blendmix-lm"

type taskDef struct{ Name, Desc string }

var knownTasks = []taskDef{
	{"recall", "associative retrieval (k→v)"},
	{"copy", "state capacity (reproduce a sequence)"},
	{"induction", "in-context pattern (A B … A → B)"},
	{lmTask, "LM: local code+docs corpus, 1.9M tok (enables top/lmtp/bst/jtp → leaderboard)"},
	{blendTask, "LM: Open-PerfectBlend chat/math/code, 388M tok flat windows (first launch builds the cache)"},
	{blendMixTask, "LM: Open-PerfectBlend packed into 512…32k context buckets — mixed-context training, batch scales 1/ctx (batch × context len = token budget/step)"},
}

// LM-corpus tasks all end in "-lm" (the Datastar expressions rely on the same suffix).
func isLMTask(name string) bool { return strings.HasSuffix(name, "-lm") }

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

type campaignArm struct {
	Name                                     string
	Seeds                                    int
	Acc, Std, TokPS, PeakMB, StepMS, EnergyJ float64
	Delta, CILo, CIHi, PAdj                  float64
	Significant, Confirmed, Pareto           bool
	SeriesJSONs                              []string
	accs, toks, peaks, steps, energy         []float64
}

type campaignRow struct {
	ID, ParentID      int64
	Name, Task, Phase string
	Status, Sha       string
	Created           float64
	Arms              []*campaignArm
}

func meanStd(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return math.NaN(), math.NaN()
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	m := sum / float64(len(xs))
	if len(xs) == 1 {
		return m, 0
	}
	var q float64
	for _, x := range xs {
		q += (x - m) * (x - m)
	}
	return m, math.Sqrt(q / float64(len(xs)-1))
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

func (s *Server) readCampaigns() ([]*campaignRow, error) {
	path := filepath.Join(s.cfg.RepoRoot, "experiments.db")
	if _, err := os.Stat(path); err != nil {
		return nil, nil
	}
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro&_pragma=busy_timeout(2000)")
	if err != nil {
		return nil, err
	}
	defer db.Close()
	rows, err := db.Query(`SELECT id,COALESCE(parent_id,0),COALESCE(name,''),task,phase,status,
		COALESCE(git_sha,''),created_ts FROM campaigns ORDER BY created_ts DESC LIMIT 20`)
	if err != nil {
		if strings.Contains(err.Error(), "no such table") {
			return nil, nil
		}
		return nil, err
	}
	byID, order := map[int64]*campaignRow{}, []*campaignRow{}
	for rows.Next() {
		c := &campaignRow{}
		if rows.Scan(&c.ID, &c.ParentID, &c.Name, &c.Task, &c.Phase, &c.Status, &c.Sha, &c.Created) == nil {
			byID[c.ID] = c
			order = append(order, c)
		}
	}
	rows.Close()
	if len(order) == 0 {
		return order, nil
	}

	tr, err := db.Query(`SELECT t.campaign_id,a.name,t.seed,t.metrics_json,COALESCE(t.series_json,'[]')
		FROM trials t JOIN arms a ON a.id=t.arm_id
		WHERE t.status='complete' AND t.phase=(SELECT phase FROM campaigns WHERE id=t.campaign_id)
		AND t.rung=(SELECT max(t2.rung) FROM trials t2 WHERE t2.campaign_id=t.campaign_id
		  AND t2.arm_id=t.arm_id AND t2.phase=t.phase AND t2.status='complete')
		ORDER BY t.campaign_id,a.name,t.seed`)
	if err != nil {
		return nil, err
	}
	type armKey struct {
		cid  int64
		name string
	}
	arms := map[armKey]*campaignArm{}
	for tr.Next() {
		var cid int64
		var name, mj, sj string
		var seed int
		if tr.Scan(&cid, &name, &seed, &mj, &sj) != nil || byID[cid] == nil {
			continue
		}
		key := armKey{cid, name}
		a := arms[key]
		if a == nil {
			a = &campaignArm{Name: name, Delta: math.NaN(), CILo: math.NaN(), CIHi: math.NaN(), PAdj: math.NaN()}
			arms[key] = a
			byID[cid].Arms = append(byID[cid].Arms, a)
		}
		var m map[string]float64
		if json.Unmarshal([]byte(mj), &m) != nil {
			continue
		}
		a.accs = append(a.accs, m["acc"])
		a.toks = append(a.toks, m["tok_per_sec"])
		a.peaks = append(a.peaks, m["peak_alloc_mb"])
		a.steps = append(a.steps, m["step_ms_p50"])
		a.energy = append(a.energy, m["energy_joules"])
		a.Seeds++
		a.SeriesJSONs = append(a.SeriesJSONs, sj)
	}
	tr.Close()
	cr, err := db.Query(`SELECT p.campaign_id,a.name,p.delta,p.ci_low,p.ci_high,p.p_adjusted,
		p.significant,p.confirmed FROM comparisons p JOIN arms a ON a.id=p.arm_id WHERE p.metric='acc'`)
	if err == nil {
		for cr.Next() {
			var cid int64
			var name string
			var d, lo, hi, p sql.NullFloat64
			var sig, conf int
			if cr.Scan(&cid, &name, &d, &lo, &hi, &p, &sig, &conf) == nil {
				if a := arms[armKey{cid, name}]; a != nil {
					if d.Valid {
						a.Delta = d.Float64
					}
					if lo.Valid {
						a.CILo = lo.Float64
					}
					if hi.Valid {
						a.CIHi = hi.Float64
					}
					if p.Valid {
						a.PAdj = p.Float64
					}
					a.Significant = sig != 0
					a.Confirmed = conf != 0
				}
			}
		}
		cr.Close()
	}
	for _, c := range order {
		for _, a := range c.Arms {
			a.Acc, a.Std = meanStd(a.accs)
			a.TokPS, _ = meanStd(a.toks)
			a.PeakMB, _ = meanStd(a.peaks)
			a.StepMS, _ = meanStd(a.steps)
			a.EnergyJ, _ = meanStd(a.energy)
		}
		for i, a := range c.Arms {
			dominated := false
			for j, b := range c.Arms {
				if i == j {
					continue
				}
				if b.Acc >= a.Acc && b.StepMS <= a.StepMS && b.PeakMB <= a.PeakMB && (b.Acc > a.Acc || b.StepMS < a.StepMS || b.PeakMB < a.PeakMB) {
					dominated = true
					break
				}
			}
			a.Pareto = !dominated
		}
		sort.Slice(c.Arms, func(i, j int) bool { return c.Arms[i].Acc > c.Arms[j].Acc })
	}
	return order, nil
}

func lossSparkline(raws []string) string {
	var pts []struct {
		Step int     `json:"step"`
		Loss float64 `json:"loss"`
	}
	var curves [][]float64
	for _, raw := range raws {
		if json.Unmarshal([]byte(raw), &pts) == nil && len(pts) >= 2 {
			x := make([]float64, len(pts))
			for i, p := range pts {
				x[i] = p.Loss
			}
			curves = append(curves, x)
		}
	}
	if len(curves) == 0 {
		return "—"
	}
	n := len(curves[0])
	means, stds := make([]float64, n), make([]float64, n)
	for i := 0; i < n; i++ {
		var xs []float64
		for _, c := range curves {
			if len(c) == n {
				xs = append(xs, c[i])
			}
		}
		means[i], stds[i] = meanStd(xs)
	}
	lo, hi := means[0]-stds[0], means[0]+stds[0]
	for i := range means {
		lo = math.Min(lo, means[i]-stds[i])
		hi = math.Max(hi, means[i]+stds[i])
	}
	coord := func(i int, v float64) string {
		x := float64(i)*92/float64(n-1) + 2
		y := 18 - (v-lo)/math.Max(hi-lo, 1e-9)*16
		return fmt.Sprintf("%.1f,%.1f", x, y)
	}
	var xy, band []string
	for i, v := range means {
		xy = append(xy, coord(i, v))
		band = append(band, coord(i, v+stds[i]))
	}
	for i := n - 1; i >= 0; i-- {
		band = append(band, coord(i, means[i]-stds[i]))
	}
	return `<svg viewBox="0 0 96 20" width="96" height="20"><polygon fill="currentColor" opacity=".15" points="` +
		strings.Join(band, " ") + `"/><polyline fill="none" stroke="currentColor" stroke-width="1.5" points="` +
		strings.Join(xy, " ") + `"/></svg>`
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
	campaigns, cerr := s.readCampaigns()
	if cerr != nil {
		campaigns = nil
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
		`data-text="$init==='g1g' ? 'g1g ≈ 1.5B (dims fixed)' : ($task.endsWith('-lm') ` +
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
	numField("grad accum", "gradaccum", "LM: micro-batches per optimizer step — effective batch = batch × N (1 = off)", 1)
	strField("ema", "ema", "LM: EMA decay for shadow weights — eval + checkpoint carry them (0.999 typical, 0 = off)", "0")
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
	b.WriteString(`<tr><td class="f-l">successive halving</td><td><input type="checkbox" checked data-bind-halving></td>` +
		`<td class="f-d">promote arms through 2% → 10% → 30% → 100% confidence-adjusted budget rungs</td></tr>`)
	b.WriteString(`<tr><td class="f-l">factorial interactions</td><td><input type="checkbox" data-bind-factorial></td>` +
		`<td class="f-d">add compatible pairwise combinations to estimate synergy/interference</td></tr>`)
	numField("confirm seeds", "confirmseeds", "fresh, previously unused seeds for the winning exploratory arm (0 disables)", 8)
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
			gate = ` data-attr-disabled="!$task.endsWith('-lm')"`
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

	// --- normalized campaigns: lineage, seed uncertainty, cost Pareto, confirmation ---
	b.WriteString(`<div class="exp-results"><div class="exp-h">campaigns · paired trials + cost Pareto</div>`)
	if len(campaigns) == 0 {
		b.WriteString(`<div class="empty">no normalized campaigns yet</div>`)
	}
	for _, c := range campaigns {
		lineage := ""
		if c.ParentID > 0 {
			lineage = fmt.Sprintf(` · child of #%d`, c.ParentID)
		}
		name := c.Name
		if name == "" {
			name = c.Task
		}
		fmt.Fprintf(&b, `<div class="exp-task"><div class="exp-tname">#%d %s <span class="dim">%s · %s · @%s%s</span></div>`, c.ID, esc(name), esc(c.Phase), esc(c.Status), esc(c.Sha), lineage)
		b.WriteString(`<table class="exp-tbl"><tr class="exp-hd"><td>arm</td><td>seed acc</td><td>paired evidence</td><td>loss</td><td>tok/s</td><td>step p50</td><td>peak</td><td>energy</td><td>decision</td></tr>`)
		for _, a := range c.Arms {
			evidence := "—"
			if !math.IsNaN(a.Delta) {
				evidence = fmt.Sprintf(`Δ%+.3f · CI[%+.3f,%+.3f] · p<sub>H</sub>=%.3g`, a.Delta, a.CILo, a.CIHi, a.PAdj)
			}
			decision := ""
			if a.Pareto {
				decision += `<span class="sig">PARETO</span> `
			}
			if a.Confirmed {
				decision += `<span class="sig">CONFIRMED ✓</span>`
			} else if a.Significant {
				decision += `<span class="sig">evidence ✓</span>`
			}
			fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td>%.3f±%.3f <span class="dim">n=%d</span></td><td>%s</td><td>%s</td>`+
				`<td>%s</td><td>%sms</td><td>%sMB</td><td>%sJ</td><td>%s</td></tr>`, esc(a.Name), a.Acc, a.Std, a.Seeds, evidence, lossSparkline(a.SeriesJSONs),
				fnum(a.TokPS, 0), fnum(a.StepMS, 1), fnum(a.PeakMB, 0), fnum(a.EnergyJ, 0), decision)
		}
		b.WriteString(`</table></div>`)
	}
	b.WriteString(`</div>`)

	// --- backward-compatible aggregate rows written before normalized campaigns ---
	b.WriteString(`<div class="exp-results"><div class="exp-h">legacy aggregates (unpaired)</div>`)
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
				delta = fmt.Sprintf(`<td class="dim">Δ%+.3f · unpaired</td>`, d)
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
	if isLMTask(task) || init == "g1g" || init == "resume" {
		args := append([]string{"-m", "rwkv_lab.config", "run-lm", "--levers", strings.Join(configs, ","),
			"--d-model", str("dmodel", "1024"), "--n-layers", str("nlayers", "18"), "--head-size", str("headsize", "64"),
			"--batch", str("batch", "16"), "--seq-len", str("ctxlen", "1024")}, budget...)
		if task == blendTask {
			args = append(args, "--corpus", "blend")
		} else if task == blendMixTask {
			args = append(args, "--corpus", "blend-mix")
		}
		args = append(args, optArgs...)
		if ga := str("gradaccum", "1"); ga != "" && ga != "1" { // effective batch = batch × N
			args = append(args, "--grad-accum", ga)
		}
		if em := str("ema", "0"); em != "" && em != "0" { // EMA shadow weights (eval + ckpt)
			args = append(args, "--ema", em)
		}
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
		toastErr(sse, "launch: top/lmtp/bst/jtp need an LM corpus — pick 'local-lm' or 'blend-lm' as the task")
		return
	}
	args := append([]string{"-m", "rwkv_lab.experiment",
		"--task", task + ":" + str("tasklen", "16"), "--configs", strings.Join(configs, ","),
		"--seeds", str("seeds", "1"), "--d-model", str("dmodel", "1024"), "--n-layers", str("nlayers", "18"),
		"--head-size", str("headsize", "64"), "--batch", str("batch", "16")}, budget...)
	if gb := str("genblock", "1"); gb != "" && gb != "1" { // synthetic-only: amortize gen launches
		args = append(args, "--gen-block", gb)
	}
	if on, ok := sig["halving"].(bool); ok && !on {
		args = append(args, "--no-halving")
	}
	if on, _ := sig["factorial"].(bool); on {
		args = append(args, "--factorial")
	}
	args = append(args, "--confirm-seeds", str("confirmseeds", "8"))
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
