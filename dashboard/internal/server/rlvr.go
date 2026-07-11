package server

// RLVR panel: launches the versioned rwkv_lab.rlvr_campaign contract and
// renders held-out / promotion evidence from campaign.json. The policy
// algorithms are sourced from Dr.GRPO (https://arxiv.org/abs/2503.20783),
// DAPO (https://arxiv.org/abs/2503.14476), and GSPO
// (https://arxiv.org/abs/2507.18071). Bounded recursive proposal lineage follows
// Absolute Zero (https://arxiv.org/abs/2505.03335); generated code verification
// remains an external Adamaton sandbox boundary.

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

type rlvrSummary struct {
	Runs            int     `json:"runs"`
	HeldoutMean     float64 `json:"heldout_mean"`
	HeldoutStd      float64 `json:"heldout_std"`
	BaselineMean    float64 `json:"baseline_mean"`
	DeltaMean       float64 `json:"delta_mean"`
	DeltaStd        float64 `json:"delta_std"`
	Promotions      int     `json:"promotions"`
	UpdatesApplied  int     `json:"updates_applied"`
	SFTUpdates      int     `json:"sft_updates"`
	PreflightPasses int     `json:"preflight_passes"`
	RolloutTokens   int     `json:"rollout_tokens"`
}

type rlvrCampaign struct {
	Path          string                 `json:"-"`
	Status        string                 `json:"status"`
	Checkpoint    string                 `json:"checkpoint"`
	Tasks         string                 `json:"tasks"`
	Algorithms    []string               `json:"algorithms"`
	Seeds         []int                  `json:"seeds"`
	Steps         int                    `json:"steps"`
	GroupSize     int                    `json:"group_size"`
	CompletedArms int                    `json:"completed_arms"`
	FailedArms    int                    `json:"failed_arms"`
	Created       float64                `json:"created_ts"`
	Elapsed       float64                `json:"elapsed_seconds"`
	Summary       map[string]rlvrSummary `json:"summary"`
}

type recursiveLoop struct {
	Path               string  `json:"-"`
	Status             string  `json:"status"`
	CurrentCheckpoint  string  `json:"current_checkpoint"`
	CompletedRounds    int     `json:"completed_rounds"`
	Promotions         int     `json:"promotions"`
	TotalRolloutTokens int     `json:"total_rollout_tokens"`
	Created            float64 `json:"created_ts"`
}

type rlvrDiscovery struct {
	loops     []recursiveLoop
	campaigns []rlvrCampaign
}

func (s *Server) readRLVRDiscovery() rlvrDiscovery {
	return s.cachedDiscovery("rlvr-campaigns", 2*time.Second, func() any {
		value := rlvrDiscovery{}
		_ = filepath.WalkDir(s.cfg.RunsDir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || (entry.Name() != "loop.json" && entry.Name() != "campaign.json") {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return nil
			}
			rel, _ := filepath.Rel(s.cfg.RunsDir, filepath.Dir(path))
			if entry.Name() == "loop.json" {
				var row recursiveLoop
				if json.Unmarshal(data, &row) == nil && row.Status != "" {
					row.Path = filepath.ToSlash(rel)
					value.loops = append(value.loops, row)
				}
			} else {
				var row rlvrCampaign
				if json.Unmarshal(data, &row) == nil &&
					(strings.HasPrefix(row.Status, "run") || row.Status == "complete" || row.Status == "failed") {
					row.Path = filepath.ToSlash(rel)
					value.campaigns = append(value.campaigns, row)
				}
			}
			return nil
		})
		sort.Slice(value.loops, func(i, j int) bool { return value.loops[i].Created > value.loops[j].Created })
		sort.Slice(value.campaigns, func(i, j int) bool { return value.campaigns[i].Created > value.campaigns[j].Created })
		if len(value.campaigns) > 20 {
			value.campaigns = value.campaigns[:20]
		}
		return value
	}).(rlvrDiscovery)
}

func (s *Server) readRecursiveLoops() []recursiveLoop { return s.readRLVRDiscovery().loops }

func (s *Server) readRLVRCampaigns() []rlvrCampaign { return s.readRLVRDiscovery().campaigns }

func (s *Server) handleRLVR(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	var b strings.Builder
	b.WriteString(`<div id="rlvr-body" class="rlvr-body">`)
	b.WriteString(`<div class="rlvr-build"><div class="exp-h">new verifiable-reward campaign</div>`)
	b.WriteString(`<table class="field-tbl">`)
	row := func(label, control, hint string) {
		fmt.Fprintf(&b, `<tr><td class="f-l">%s</td><td>%s</td><td class="f-d">%s</td></tr>`,
			esc(label), control, esc(hint))
	}
	row("parent checkpoint", `<input type="text" data-bind-rlvrckpt value="runs/gen_smoke/ckpt.pt">`,
		"self-describing rwkv_pretrain checkpoint; the parent is never overwritten")
	row("output", `<input type="text" data-bind-rlvrout value="runs/rlvr-campaign">`,
		"campaign directory under runs/")
	row("task JSONL", `<input type="text" data-bind-rlvrtasks value="">`,
		"empty = generated arithmetic; or experiments/rlvr_arithmetic.example.jsonl")
	row("held-out JSONL", `<input type="text" data-bind-rlvrheldout value="">`,
		"optional separate eval-only file; never shared with a proposal process")
	row("algorithms", `<input type="text" data-bind-rlvralgorithms value="gspo,dr_grpo,dapo">`,
		"comma-separated comparison arms")
	row("seeds", `<input type="text" data-bind-rlvrseeds value="0,1,2">`,
		"paired task/sampling seeds (maximum 8)")
	row("steps", `<input type="number" min="1" max="10000" data-bind-rlvrsteps value="20">`,
		"policy-update steps per algorithm and seed")
	row("prompts / step", `<input type="number" min="1" max="64" data-bind-rlvrprompts value="2">`,
		"distinct task groups per update")
	row("group size", `<input type="number" min="2" max="64" data-bind-rlvrgroup value="8">`,
		"rollouts per prompt; group-relative objectives require at least two")
	row("max new", `<input type="number" min="1" max="512" data-bind-rlvrmaxnew value="32">`,
		"maximum response policy tokens")
	row("rollout engine", `<select data-bind-rlvrengine><option value="auto">auto</option>`+
		`<option value="recurrent">recurrent only</option><option value="batched">batched prefix</option></select>`,
		"auto uses constant-state RWKV decoding with an exact lever-safe fallback")
	row("rollout devices", `<input type="text" data-bind-rlvrdevices value="" placeholder="cuda:0,cuda:1">`,
		"optional inference replicas; first entry must match the policy device")
	row("curriculum", `<input type="text" data-bind-rlvrcurriculum value="1,2">`,
		"comma-separated difficulty stages")
	row("SFT warm-start", `<input type="number" min="0" max="10000" data-bind-rlvrsft value="16">`,
		"trusted-answer cold-start updates before RL")
	row("preflight prompts", `<input type="number" min="0" max="512" data-bind-rlvrpreflight value="8">`,
		"require mixed verifier rewards before relative policy updates")
	row("learning rate", `<input type="text" data-bind-rlvrlr value="1e-6">`, "AdamW policy LR")
	row("KL coefficient", `<input type="text" data-bind-rlvrkl value="0.01">`,
		"non-negative reference-policy penalty")
	row("reference", `<select data-bind-rlvrreference><option value="rollout">rollout policy</option>`+
		`<option value="initial">fixed parent (2× model memory)</option><option value="none">none</option></select>`,
		"rollout is the memory-light proximal reference")
	row("eval every", `<input type="number" min="1" max="1000" data-bind-rlvrevalevery value="5">`,
		"held-out evaluation interval")
	row("eval prompts", `<input type="number" min="1" max="512" data-bind-rlvrevalprompts value="16">`,
		"fixed hidden prompts scored before and after RL")
	row("promotion delta", `<input type="text" data-bind-rlvrmindelta value="0.01">`,
		"absolute held-out reward gain required for eligibility")
	row("confidence", `<input type="text" data-bind-rlvrconfidence value="0.95">`,
		"paired-bootstrap confidence level; lower bound must clear the delta")
	row("family regression", `<input type="text" data-bind-rlvrfamilyreg value="0">`,
		"maximum tolerated held-out regression in any task family")
	row("rollout budget", `<input type="number" min="0" data-bind-rlvrtokenbudget value="0">`,
		"hard candidate rollout-token cap; 0 = unlimited")
	row("time budget", `<input type="text" data-bind-rlvrtimebudget value="0">`,
		"hard candidate seconds cap; 0 = unlimited")
	row("device", `<select data-bind-rlvrdevice><option value="cuda">cuda</option><option value="cpu">cpu</option></select>`,
		"CUDA is required for practical campaigns")
	b.WriteString(`</table><button class="btn" data-on:click="@post('/api/rlvr/launch')">▶ run RLVR campaign</button></div>`)
	b.WriteString(`<div class="rlvr-results"><div class="exp-h">campaign evidence ` +
		`<button class="btn sm" data-on:click="@get('/api/rlvr')">refresh</button></div>`)
	rows := s.readRLVRCampaigns()
	if len(rows) == 0 {
		b.WriteString(`<div class="empty">no RLVR campaigns yet</div>`)
	}
	for _, campaign := range rows {
		created := time.Unix(int64(campaign.Created), 0).Format("Jan 02 15:04")
		fmt.Fprintf(&b, `<div class="rlvr-campaign"><div class="exp-tname"><code>%s</code> `+
			`<span class="rlvr-status %s">%s</span> <span class="dim">%s · %d steps · group %d · %.1fm</span></div>`,
			esc(campaign.Path), esc(campaign.Status), esc(campaign.Status), created,
			campaign.Steps, campaign.GroupSize, campaign.Elapsed/60)
		b.WriteString(`<table class="exp-tbl"><tr class="exp-hd"><td>algorithm</td><td>runs</td>` +
			`<td>baseline</td><td>held-out</td><td>Δ reward</td><td>RL/SFT</td>` +
			`<td>preflight</td><td>tokens</td><td>promotions</td></tr>`)
		algorithms := append([]string(nil), campaign.Algorithms...)
		sort.Strings(algorithms)
		for _, algorithm := range algorithms {
			summary, ok := campaign.Summary[algorithm]
			if !ok {
				fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td colspan="8" class="dim">pending</td></tr>`, esc(algorithm))
				continue
			}
			decision := fmt.Sprintf("%d/%d", summary.Promotions, summary.Runs)
			fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td>%d</td><td>%.3f</td>`+
				`<td>%.3f ± %.3f</td><td class="%s">%+.3f ± %.3f</td><td>%d/%d</td>`+
				`<td>%d/%d</td><td>%d</td><td>%s</td></tr>`,
				esc(algorithm), summary.Runs, summary.BaselineMean, summary.HeldoutMean,
				summary.HeldoutStd, map[bool]string{true: "sig", false: "ns"}[summary.DeltaMean > 0],
				summary.DeltaMean, summary.DeltaStd, summary.UpdatesApplied, summary.SFTUpdates,
				summary.PreflightPasses, summary.Runs, summary.RolloutTokens, decision)
		}
		b.WriteString(`</table></div>`)
	}
	loops := s.readRecursiveLoops()
	if len(loops) > 0 {
		b.WriteString(`<div class="exp-h">recursive improvement lineage</div>`)
		b.WriteString(`<table class="exp-tbl"><tr class="exp-hd"><td>loop</td><td>status</td>` +
			`<td>rounds</td><td>promotions</td><td>rollout tokens</td><td>current checkpoint</td></tr>`)
		for _, loop := range loops {
			fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td>%s</td><td>%d</td>`+
				`<td>%d</td><td>%d</td><td><code>%s</code></td></tr>`,
				esc(loop.Path), esc(loop.Status), loop.CompletedRounds, loop.Promotions,
				loop.TotalRolloutTokens, esc(filepath.Base(loop.CurrentCheckpoint)))
		}
		b.WriteString(`</table>`)
	}
	b.WriteString(`</div></div>`)
	_ = sse.PatchElements(b.String())
}

func boundedInt(value string, fallback, lo, hi int) (string, error) {
	value = strings.TrimSpace(value)
	n := fallback
	var err error
	if value != "" {
		n, err = strconv.Atoi(value)
	}
	if err != nil {
		return "", fmt.Errorf("%q is not an integer", value)
	}
	if value == "" {
		n = fallback
	}
	if n < lo || n > hi {
		return "", fmt.Errorf("value %d outside [%d,%d]", n, lo, hi)
	}
	return strconv.Itoa(n), nil
}

func (s *Server) pathUnderRepo(value string, mustExist bool) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", nil
	}
	path := value
	if !filepath.IsAbs(path) {
		path = filepath.Join(s.cfg.RepoRoot, path)
	}
	path = filepath.Clean(path)
	rel, err := filepath.Rel(s.cfg.RepoRoot, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path must stay under repository root")
	}
	if mustExist {
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			return "", fmt.Errorf("file does not exist: %s", value)
		}
	}
	return path, nil
}

func (s *Server) handleLaunchRLVR(w http.ResponseWriter, r *http.Request) {
	var sig map[string]any
	_ = datastar.ReadSignals(r, &sig)
	if sig == nil {
		sig = map[string]any{}
	}
	sse := datastar.NewSSE(w, r)
	str := func(key, fallback string) string {
		if value, ok := sig[key]; ok {
			return fmt.Sprintf("%v", value)
		}
		return fallback
	}
	ckpt, err := s.pathUnderRepo(str("rlvrckpt", "runs/gen_smoke/ckpt.pt"), true)
	if err != nil {
		toastErr(sse, "RLVR checkpoint: "+err.Error())
		return
	}
	if ckpt == "" {
		toastErr(sse, "RLVR checkpoint is required")
		return
	}
	tasks, err := s.pathUnderRepo(str("rlvrtasks", ""), true)
	if err != nil {
		toastErr(sse, "RLVR tasks: "+err.Error())
		return
	}
	heldout, err := s.pathUnderRepo(str("rlvrheldout", ""), true)
	if err != nil {
		toastErr(sse, "RLVR held-out tasks: "+err.Error())
		return
	}
	outValue := strings.TrimSpace(str("rlvrout", "runs/rlvr-campaign"))
	if outValue == "" {
		toastErr(sse, "RLVR output is required")
		return
	}
	out, err := s.pathUnderRepo(outValue, false)
	if err != nil {
		toastErr(sse, "RLVR output: "+err.Error())
		return
	}
	relRuns, _ := filepath.Rel(s.cfg.RunsDir, out)
	if relRuns == "." || relRuns == ".." || strings.HasPrefix(relRuns, ".."+string(filepath.Separator)) {
		toastErr(sse, "RLVR output must be under runs/")
		return
	}
	algorithms := strings.Split(str("rlvralgorithms", "gspo,dr_grpo,dapo"), ",")
	allowed := map[string]bool{"gspo": true, "dr_grpo": true, "dapo": true}
	seenAlgorithms := map[string]bool{}
	for _, algorithm := range algorithms {
		algorithm = strings.TrimSpace(algorithm)
		if !allowed[algorithm] || seenAlgorithms[algorithm] {
			toastErr(sse, "RLVR algorithms must be gspo, dr_grpo, or dapo")
			return
		}
		seenAlgorithms[algorithm] = true
	}
	seeds := strings.Split(str("rlvrseeds", "0,1,2"), ",")
	if len(seeds) == 0 || len(seeds) > 8 {
		toastErr(sse, "RLVR requires 1–8 seeds")
		return
	}
	seenSeeds := map[int]bool{}
	for _, seed := range seeds {
		value, err := strconv.Atoi(strings.TrimSpace(seed))
		if err != nil || seenSeeds[value] {
			toastErr(sse, "RLVR seeds must be comma-separated integers")
			return
		}
		seenSeeds[value] = true
	}
	steps, err := boundedInt(str("rlvrsteps", "20"), 20, 1, 10000)
	if err != nil {
		toastErr(sse, "RLVR steps: "+err.Error())
		return
	}
	prompts, err := boundedInt(str("rlvrprompts", "2"), 2, 1, 64)
	if err != nil {
		toastErr(sse, "RLVR prompts: "+err.Error())
		return
	}
	group, err := boundedInt(str("rlvrgroup", "8"), 8, 2, 64)
	if err != nil {
		toastErr(sse, "RLVR group: "+err.Error())
		return
	}
	maxNew, err := boundedInt(str("rlvrmaxnew", "32"), 32, 1, 512)
	if err != nil {
		toastErr(sse, "RLVR max-new: "+err.Error())
		return
	}
	sftSteps, err := boundedInt(str("rlvrsft", "16"), 16, 0, 10000)
	if err != nil {
		toastErr(sse, "RLVR SFT steps: "+err.Error())
		return
	}
	preflightPrompts, err := boundedInt(str("rlvrpreflight", "8"), 8, 0, 512)
	if err != nil {
		toastErr(sse, "RLVR preflight: "+err.Error())
		return
	}
	tokenBudget, err := boundedInt(str("rlvrtokenbudget", "0"), 0, 0, 2_000_000_000)
	if err != nil {
		toastErr(sse, "RLVR rollout budget: "+err.Error())
		return
	}
	evalEvery, err := boundedInt(str("rlvrevalevery", "5"), 5, 1, 1000)
	if err != nil {
		toastErr(sse, "RLVR eval interval: "+err.Error())
		return
	}
	evalPrompts, err := boundedInt(str("rlvrevalprompts", "16"), 16, 1, 512)
	if err != nil {
		toastErr(sse, "RLVR eval prompts: "+err.Error())
		return
	}
	for name, value := range map[string]string{"learning rate": str("rlvrlr", "1e-6"),
		"KL coefficient": str("rlvrkl", "0.01"), "promotion delta": str("rlvrmindelta", "0.01"),
		"family regression": str("rlvrfamilyreg", "0"), "time budget": str("rlvrtimebudget", "0")} {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) || parsed < 0 {
			toastErr(sse, "RLVR "+name+" must be a non-negative number")
			return
		}
	}
	confidence, err := strconv.ParseFloat(strings.TrimSpace(str("rlvrconfidence", "0.95")), 64)
	if err != nil || math.IsNaN(confidence) || math.IsInf(confidence, 0) || confidence <= 0 || confidence >= 1 {
		toastErr(sse, "RLVR confidence must be between zero and one")
		return
	}
	curriculum := strings.TrimSpace(str("rlvrcurriculum", "1,2"))
	for _, stage := range strings.Split(curriculum, ",") {
		value, err := strconv.Atoi(strings.TrimSpace(stage))
		if err != nil || value < 1 || value > 4 {
			toastErr(sse, "RLVR curriculum stages must be comma-separated integers in [1,4]")
			return
		}
	}
	engine := str("rlvrengine", "auto")
	if engine != "auto" && engine != "recurrent" && engine != "batched" {
		toastErr(sse, "RLVR rollout engine must be auto, recurrent, or batched")
		return
	}
	reference := str("rlvrreference", "rollout")
	if reference != "rollout" && reference != "initial" && reference != "none" {
		toastErr(sse, "RLVR reference must be rollout, initial, or none")
		return
	}
	device := str("rlvrdevice", "cuda")
	if device != "cuda" && device != "cpu" {
		toastErr(sse, "RLVR device must be cuda or cpu")
		return
	}
	rolloutDevices := strings.TrimSpace(str("rlvrdevices", ""))
	if rolloutDevices != "" {
		for _, raw := range strings.Split(rolloutDevices, ",") {
			value := strings.TrimSpace(raw)
			index, err := strconv.Atoi(strings.TrimPrefix(value, "cuda:"))
			if !strings.HasPrefix(value, "cuda:") || err != nil || index < 0 {
				toastErr(sse, "RLVR rollout devices must be comma-separated cuda:N values")
				return
			}
		}
	}
	args := []string{"-m", "rwkv_lab.rlvr_campaign", "--ckpt", ckpt, "--out", out,
		"--algorithms", str("rlvralgorithms", "gspo,dr_grpo,dapo"),
		"--seeds", str("rlvrseeds", "0,1,2"), "--steps", steps,
		"--prompts-per-step", prompts, "--group-size", group, "--max-new", maxNew,
		"--rollout-engine", engine, "--curriculum-stages", curriculum,
		"--sft-steps", sftSteps, "--preflight-prompts", preflightPrompts,
		"--lr", str("rlvrlr", "1e-6"), "--kl-coef", str("rlvrkl", "0.01"),
		"--reference", reference, "--eval-every", evalEvery, "--eval-prompts", evalPrompts,
		"--min-heldout-delta", str("rlvrmindelta", "0.01"),
		"--confidence", str("rlvrconfidence", "0.95"), "--require-confidence",
		"--max-family-regression", str("rlvrfamilyreg", "0"),
		"--max-rollout-tokens", tokenBudget, "--max-train-seconds", str("rlvrtimebudget", "0"),
		"--device", device}
	if rolloutDevices != "" {
		args = append(args, "--rollout-devices", rolloutDevices)
	}
	if tasks != "" {
		args = append(args, "--tasks", tasks)
	}
	if heldout != "" {
		args = append(args, "--heldout-tasks", heldout)
	}
	pid, err := s.spawnPy(args, fmt.Sprintf("rlvr_campaign_%d.log", time.Now().Unix()))
	if err != nil {
		toastErr(sse, "RLVR launch failed: "+err.Error())
		return
	}
	toastOK(sse, fmt.Sprintf("launched RLVR campaign (%d arms, pid %d)", len(algorithms)*len(seeds), pid))
}
