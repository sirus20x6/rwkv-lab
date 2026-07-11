package server

// Post-training data + behavior panel. Dataset paths are repository-confined and validation is
// delegated to rwkv_lab.posttrain_data. Paired generation uses the same prompt, seed, temperature,
// and token budget for both checkpoints. Explicit operator choices append training preferences;
// they never enter held-out evaluation data and never trigger training or publication.

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

type posttrainPreview struct {
	ID       string `json:"id"`
	Kind     string `json:"kind"`
	Split    string `json:"split"`
	Variants map[string]struct {
		Text       string `json:"text"`
		TrainChars int    `json:"train_chars"`
	} `json:"variants"`
}

type posttrainInspection struct {
	Schema        string             `json:"schema"`
	Path          string             `json:"path"`
	SHA256        string             `json:"sha256"`
	Bytes         int64              `json:"bytes"`
	Examples      int                `json:"examples"`
	Kinds         map[string]int     `json:"kinds"`
	Splits        map[string]int     `json:"splits"`
	Duplicates    int                `json:"duplicates"`
	SplitOverlaps int                `json:"split_overlaps"`
	Template      string             `json:"template"`
	Previews      []posttrainPreview `json:"previews"`
}

type posttrainComparison struct {
	N                    int `json:"n"`
	Delta, CILow, CIHigh float64
}

func (p *posttrainComparison) UnmarshalJSON(data []byte) error {
	type raw struct {
		N      int     `json:"n"`
		Delta  float64 `json:"delta"`
		CILow  float64 `json:"ci_low"`
		CIHigh float64 `json:"ci_high"`
	}
	var value raw
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	p.N, p.Delta, p.CILow, p.CIHigh = value.N, value.Delta, value.CILow, value.CIHigh
	return nil
}

type posttrainReceipt struct {
	Objective       string `json:"objective"`
	Eligible        bool   `json:"eligible"`
	Reason          string `json:"reason"`
	SelectedAdapter string `json:"selected_adapter"`
}

type posttrainCampaign struct {
	Path              string                                    `json:"-"`
	Status            string                                    `json:"status"`
	Created           float64                                   `json:"created_ts"`
	Elapsed           float64                                   `json:"elapsed_seconds"`
	Objectives        []string                                  `json:"objectives"`
	Seeds             []int                                     `json:"seeds"`
	ConfirmationSeeds []int                                     `json:"confirmation_seeds"`
	Comparisons       map[string]map[string]posttrainComparison `json:"comparisons"`
	Receipts          []posttrainReceipt                        `json:"promotion_receipts"`
}

type adapterLoopRow struct {
	Path       string `json:"-"`
	Status     string `json:"status"`
	Current    string `json:"current_checkpoint"`
	Iterations []struct {
		Accepted  bool   `json:"accepted"`
		Preserved string `json:"preserved_adapter"`
	} `json:"iterations"`
}

type posttrainDiscovery struct {
	campaigns []posttrainCampaign
	loops     []adapterLoopRow
}

func (s *Server) readPosttrainCampaigns() ([]posttrainCampaign, []adapterLoopRow) {
	value := s.cachedDiscovery("posttrain-campaigns", 2*time.Second, func() any {
		campaigns := []posttrainCampaign{}
		loops := []adapterLoopRow{}
		_ = filepath.WalkDir(s.cfg.RunsDir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return nil
			}
			if entry.Name() != "posttrain-campaign.json" && entry.Name() != "adapter-loop.json" {
				return nil
			}
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				return nil
			}
			rel, _ := filepath.Rel(s.cfg.RunsDir, filepath.Dir(path))
			if entry.Name() == "posttrain-campaign.json" {
				var row posttrainCampaign
				if json.Unmarshal(data, &row) == nil && row.Status != "" {
					row.Path = filepath.ToSlash(rel)
					campaigns = append(campaigns, row)
				}
			}
			if entry.Name() == "adapter-loop.json" {
				var row adapterLoopRow
				if json.Unmarshal(data, &row) == nil && row.Status != "" {
					row.Path = filepath.ToSlash(rel)
					loops = append(loops, row)
				}
			}
			return nil
		})
		sort.Slice(campaigns, func(i, j int) bool { return campaigns[i].Created > campaigns[j].Created })
		if len(campaigns) > 20 {
			campaigns = campaigns[:20]
		}
		return posttrainDiscovery{campaigns: campaigns, loops: loops}
	}).(posttrainDiscovery)
	return value.campaigns, value.loops
}

func (s *Server) handlePosttraining(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	var b strings.Builder
	b.WriteString(`<div id="posttraining-body"><div class="panel-title">dataset contracts <span class="sub">validate · preview rendering + train mask · immutable hash</span></div>`)
	b.WriteString(`<div class="ctl-row"><input class="ctl-input" list="posttrain-datasets" placeholder="datasets/example.jsonl" data-bind="ptDataset"><datalist id="posttrain-datasets">`)
	for _, path := range s.posttrainDatasets() {
		fmt.Fprintf(&b, `<option value="%s"></option>`, esc(path))
	}
	b.WriteString(`</datalist><button class="btn" data-on:click="@post('/api/posttraining/inspect')">inspect</button></div>`)
	b.WriteString(`<div class="ctl-row"><input class="ctl-input" placeholder="optional additional JSONL paths, comma-separated" data-bind="ptMerge"><button class="btn" data-on:click="confirm('Validate and create an immutable merged dataset version?') && @post('/api/posttraining/version')">version / merge</button></div>`)
	b.WriteString(`<div id="posttraining-inspect"><div class="empty">select a repository JSONL dataset</div></div>`)
	b.WriteString(`<div class="panel-title">post-training campaigns <span class="sub">equal token budget · paired seeds · fresh confirmation · immutable promotion receipt</span></div>`)
	b.WriteString(`<div class="pt-campaign-grid"><div><table class="field-tbl">`)
	b.WriteString(`<tr><td class="f-l">parent</td><td><input data-bind="ptCampaignCkpt" value="runs/gen_smoke/ckpt.pt"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">train / held-out</td><td><input data-bind="ptCampaignData" placeholder="datasets/train.jsonl"><input data-bind="ptCampaignEval" placeholder="datasets/eval.jsonl"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">output</td><td><input data-bind="ptCampaignOut" value="runs/posttrain-campaign"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">objectives</td><td><input data-bind="ptCampaignObjectives" value="sft,dpo,kto,orpo,simpo"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">explore / confirm</td><td><input data-bind="ptCampaignSeeds" value="0,1,2"><input data-bind="ptCampaignConfirm" value="100,101,102"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">token budget</td><td><input type="number" data-bind="ptCampaignBudget" value="100000"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">base</td><td><select data-bind="ptCampaignQuant"><option value="none">dense LoRA</option><option value="nf4">native NF4 QLoRA</option></select></td></tr>`)
	b.WriteString(`<tr><td class="f-l">NF4 backend</td><td><select data-bind="ptCampaignBackend"><option value="auto">auto · parity + speed gate</option><option value="portable">portable reference</option><option value="torchao">require TorchAO</option></select></td></tr>`)
	b.WriteString(`<tr><td class="f-l">packing</td><td><select data-bind="ptCampaignPacking"><option value="reset">reset-mask multipack</option><option value="audit">audit only</option><option value="off">off</option></select></td></tr>`)
	b.WriteString(`<tr><td class="f-l">device slots</td><td><input data-bind="ptCampaignDevices" value="cuda:0" placeholder="cuda:0,cuda:1"></td></tr>`)
	b.WriteString(`<tr><td class="f-l">timeout / retries</td><td><input data-bind="ptCampaignTimeout" value="0" title="seconds; 0 disables"><input type="number" data-bind="ptCampaignRetries" value="1" min="0" max="10"></td></tr>`)
	b.WriteString(`</table><button class="btn" data-on:click="confirm('Launch paired post-training and confirmation campaign?') && @post('/api/posttraining/campaign')">▶ run campaign</button></div><div class="pt-campaign-results">`)
	campaigns, loops := s.readPosttrainCampaigns()
	if len(campaigns) == 0 {
		b.WriteString(`<div class="empty">no post-training campaigns yet</div>`)
	}
	for _, campaign := range campaigns {
		fmt.Fprintf(&b, `<div class="rlvr-campaign"><div class="exp-tname"><code>%s</code> <span class="rlvr-status %s">%s</span> <span class="dim">%d explore · %d confirm · %.1fm</span></div>`, esc(campaign.Path), esc(campaign.Status), esc(campaign.Status), len(campaign.Seeds), len(campaign.ConfirmationSeeds), campaign.Elapsed/60)
		b.WriteString(`<table class="exp-tbl"><tr class="exp-hd"><td>objective</td><td>explore Δ [CI]</td><td>confirm Δ [CI]</td><td>promotion</td></tr>`)
		for _, objective := range campaign.Objectives {
			explore := campaign.Comparisons[objective]["explore"]
			confirm := campaign.Comparisons[objective]["confirm"]
			decision := "pending"
			for _, receipt := range campaign.Receipts {
				if receipt.Objective == objective {
					if receipt.Eligible {
						decision = "eligible"
					} else {
						decision = "rejected"
					}
				}
			}
			fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td>%+.4f [%+.4f,%+.4f]</td><td>%+.4f [%+.4f,%+.4f]</td><td class="%s">%s</td></tr>`, esc(objective), explore.Delta, explore.CILow, explore.CIHigh, confirm.Delta, confirm.CILow, confirm.CIHigh, map[bool]string{true: "sig", false: "ns"}[decision == "eligible"], decision)
		}
		b.WriteString(`</table></div>`)
	}
	if len(loops) > 0 {
		b.WriteString(`<div class="exp-h">adapter-recursive lineage</div><table class="exp-tbl"><tr class="exp-hd"><td>loop</td><td>status</td><td>rounds</td><td>accepted</td><td>current parent</td></tr>`)
		for _, loop := range loops {
			accepted := 0
			for _, iteration := range loop.Iterations {
				if iteration.Accepted {
					accepted++
				}
			}
			fmt.Fprintf(&b, `<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td><code>%s</code></td></tr>`, esc(loop.Path), esc(loop.Status), len(loop.Iterations), accepted, esc(loop.Current))
		}
		b.WriteString(`</table>`)
	}
	b.WriteString(`</div></div>`)
	b.WriteString(`<div class="panel-title">paired behavior <span class="sub">same prompt · seed · sampling settings · explicit preference capture</span></div>`)
	b.WriteString(`<div class="ctl-row"><input class="ctl-input" placeholder="run A" data-bind="ptRunA"><input class="ctl-input" placeholder="run B" data-bind="ptRunB"></div>`)
	b.WriteString(`<textarea class="ctl-input" rows="2" placeholder="comparison prompt" data-bind="ptPrompt"></textarea>`)
	b.WriteString(`<div class="ctl-row"><input class="ctl-input" style="max-width:90px" title="seed" data-bind="ptSeed"><input class="ctl-input" style="max-width:100px" title="max new" data-bind="ptMaxNew"><input class="ctl-input" style="max-width:90px" title="temperature" data-bind="ptTemp"><button class="btn" data-attr-disabled="$ptBusy" data-on:click="@post('/api/posttraining/compare')">compare</button></div>`)
	b.WriteString(`<div id="posttraining-compare"><div class="empty">choose two completed LM runs with ckpt.pt</div></div>`)
	b.WriteString(`<div class="ctl-row"><select class="ctl-input" data-bind="ptChosen"><option value="">choose preferred output</option><option value="a">A preferred</option><option value="b">B preferred</option></select><button class="btn" data-on:click="confirm('Append this pair to datasets/trainboard_preferences.jsonl?') && @post('/api/posttraining/feedback')">save training preference</button></div></div>`)
	_ = sse.PatchElements(b.String())
}

func (s *Server) posttrainDatasets() []string {
	return s.cachedDiscovery("posttrain-datasets", 2*time.Second, func() any {
		var paths []string
		for _, base := range []string{"datasets", "data"} {
			root := filepath.Join(s.cfg.RepoRoot, base)
			_ = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
				if err != nil || entry == nil {
					return nil
				}
				if entry.IsDir() && strings.Count(strings.TrimPrefix(path, root), string(filepath.Separator)) > 3 {
					return filepath.SkipDir
				}
				if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".jsonl") {
					rel, relErr := filepath.Rel(s.cfg.RepoRoot, path)
					if relErr == nil {
						paths = append(paths, filepath.ToSlash(rel))
					}
				}
				return nil
			})
		}
		sort.Strings(paths)
		return paths
	}).([]string)
}

func (s *Server) handleInspectPosttraining(w http.ResponseWriter, r *http.Request) {
	var sig struct {
		Dataset string `json:"ptDataset"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	path, err := s.pathUnderRepo(sig.Dataset, true)
	if err != nil || path == "" || !strings.HasSuffix(strings.ToLower(path), ".jsonl") {
		toastErr(sse, "dataset must be an existing repository .jsonl file")
		return
	}
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"),
		"-m", "rwkv_lab.posttrain_data", path, "--limit", "3", "--json")
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src")
	out, err := cmd.Output()
	if err != nil {
		toastErr(sse, "dataset validation failed: "+commandError(err))
		return
	}
	var result posttrainInspection
	if json.Unmarshal(out, &result) != nil {
		toastErr(sse, "dataset inspector returned invalid output")
		return
	}
	var b strings.Builder
	fmt.Fprintf(&b, `<div id="posttraining-inspect"><div class="kpi-row"><span><b>%d</b> examples</span><span><b>%d</b> bytes</span><span><b>%d</b> duplicates</span><span><b>%d</b> split overlaps</span><span><b>%s</b> template</span></div><div class="sub">sha256 %s · kinds %s · splits %s</div>`,
		result.Examples, result.Bytes, result.Duplicates, result.SplitOverlaps, esc(result.Template), esc(shortHash(result.SHA256)), esc(fmt.Sprint(result.Kinds)), esc(fmt.Sprint(result.Splits)))
	for _, preview := range result.Previews {
		fmt.Fprintf(&b, `<details><summary>%s · %s/%s</summary>`, esc(preview.ID), esc(preview.Split), esc(preview.Kind))
		keys := make([]string, 0, len(preview.Variants))
		for key := range preview.Variants {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			value := preview.Variants[key]
			fmt.Fprintf(&b, `<div class="sub">%s · %d train chars</div><pre class="sample-out">%s</pre>`, esc(key), value.TrainChars, esc(value.Text))
		}
		b.WriteString(`</details>`)
	}
	b.WriteString(`</div>`)
	_ = sse.PatchElements(b.String())
	toastOK(sse, fmt.Sprintf("validated %d post-training examples", result.Examples))
}

func (s *Server) handleVersionPosttraining(w http.ResponseWriter, r *http.Request) {
	var sig struct {
		Dataset string `json:"ptDataset"`
		Merge   string `json:"ptMerge"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	values := []string{sig.Dataset}
	values = append(values, strings.Split(sig.Merge, ",")...)
	paths := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		path, err := s.pathUnderRepo(value, true)
		if err != nil || !strings.HasSuffix(strings.ToLower(path), ".jsonl") {
			toastErr(sse, "every version source must be an existing repository .jsonl file")
			return
		}
		if !seen[path] {
			seen[path] = true
			paths = append(paths, path)
		}
	}
	if len(paths) == 0 {
		toastErr(sse, "select at least one dataset to version")
		return
	}
	root := filepath.Join(s.cfg.RepoRoot, "datasets", "versions")
	args := []string{"-m", "rwkv_lab.posttrain_data"}
	args = append(args, paths...)
	args = append(args, "--version-root", root, "--json")
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"), args...)
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src")
	out, err := cmd.Output()
	if err != nil {
		toastErr(sse, "dataset version failed: "+commandError(err))
		return
	}
	var result struct {
		Dataset string `json:"dataset"`
		Version string `json:"version"`
	}
	if json.Unmarshal(out, &result) != nil || result.Dataset == "" {
		toastErr(sse, "dataset versioner returned invalid output")
		return
	}
	relative, _ := filepath.Rel(s.cfg.RepoRoot, result.Dataset)
	_ = sse.MarshalAndPatchSignals(map[string]any{"ptDataset": filepath.ToSlash(relative), "ptMerge": ""})
	toastOK(sse, "created immutable dataset version "+result.Version+"; inspect to preview it")
}

func (s *Server) handleLaunchPosttrainingCampaign(w http.ResponseWriter, r *http.Request) {
	var sig struct {
		Checkpoint string `json:"ptCampaignCkpt"`
		Data       string `json:"ptCampaignData"`
		Eval       string `json:"ptCampaignEval"`
		Output     string `json:"ptCampaignOut"`
		Objectives string `json:"ptCampaignObjectives"`
		Seeds      string `json:"ptCampaignSeeds"`
		Confirm    string `json:"ptCampaignConfirm"`
		Budget     string `json:"ptCampaignBudget"`
		Quant      string `json:"ptCampaignQuant"`
		Backend    string `json:"ptCampaignBackend"`
		Packing    string `json:"ptCampaignPacking"`
		Devices    string `json:"ptCampaignDevices"`
		Timeout    string `json:"ptCampaignTimeout"`
		Retries    string `json:"ptCampaignRetries"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	checkpoint, err := s.pathUnderRepo(sig.Checkpoint, true)
	if err != nil || checkpoint == "" {
		toastErr(sse, "campaign parent must be an existing repository checkpoint")
		return
	}
	data, err := s.pathUnderRepo(sig.Data, true)
	if err != nil || data == "" || !strings.HasSuffix(strings.ToLower(data), ".jsonl") {
		toastErr(sse, "campaign train data must be an existing JSONL")
		return
	}
	eval, err := s.pathUnderRepo(sig.Eval, true)
	if err != nil || eval == "" || !strings.HasSuffix(strings.ToLower(eval), ".jsonl") {
		toastErr(sse, "campaign requires separate held-out JSONL")
		return
	}
	out, err := s.pathUnderRepo(sig.Output, false)
	if err != nil || out == "" {
		toastErr(sse, "campaign output is invalid")
		return
	}
	rel, _ := filepath.Rel(s.cfg.RunsDir, out)
	if rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		toastErr(sse, "campaign output must be under runs/")
		return
	}
	allowed := map[string]bool{"sft": true, "dpo": true, "kto": true, "orpo": true, "simpo": true, "reward": true, "prm": true}
	objectives := strings.Split(sig.Objectives, ",")
	for _, value := range objectives {
		if !allowed[strings.TrimSpace(value)] {
			toastErr(sse, "campaign objective is not allowlisted")
			return
		}
	}
	validateSeeds := func(raw string) bool {
		values := strings.Split(raw, ",")
		if len(values) < 1 || len(values) > 8 {
			return false
		}
		seen := map[int]bool{}
		for _, item := range values {
			value, parseErr := strconv.Atoi(strings.TrimSpace(item))
			if parseErr != nil || seen[value] {
				return false
			}
			seen[value] = true
		}
		return true
	}
	if !validateSeeds(sig.Seeds) || !validateSeeds(sig.Confirm) {
		toastErr(sse, "exploration and confirmation each need 1–8 unique integer seeds")
		return
	}
	budget, err := boundedInt(sig.Budget, 100000, 1, 2_000_000_000)
	if err != nil {
		toastErr(sse, "campaign token budget is invalid")
		return
	}
	if sig.Quant != "none" && sig.Quant != "nf4" {
		toastErr(sse, "campaign base must be dense or NF4")
		return
	}
	if sig.Backend != "auto" && sig.Backend != "portable" && sig.Backend != "torchao" {
		toastErr(sse, "NF4 backend must be auto, portable, or torchao")
		return
	}
	if sig.Packing != "off" && sig.Packing != "audit" && sig.Packing != "reset" {
		toastErr(sse, "packing must be off, audit, or reset")
		return
	}
	deviceValues := strings.Split(sig.Devices, ",")
	seenDevices := map[string]bool{}
	for _, raw := range deviceValues {
		value := strings.TrimSpace(raw)
		valid := value == "auto" || value == "cpu" || value == "cuda"
		if strings.HasPrefix(value, "cuda:") {
			_, parseErr := strconv.Atoi(strings.TrimPrefix(value, "cuda:"))
			valid = parseErr == nil
		}
		if !valid || seenDevices[value] {
			toastErr(sse, "device slots must be unique auto, cpu, cuda, or cuda:N values")
			return
		}
		seenDevices[value] = true
	}
	timeout, parseErr := strconv.ParseFloat(strings.TrimSpace(sig.Timeout), 64)
	if parseErr != nil || timeout < 0 {
		toastErr(sse, "arm timeout must be non-negative seconds")
		return
	}
	retries, err := boundedInt(sig.Retries, 1, 0, 10)
	if err != nil {
		toastErr(sse, "campaign retries are invalid")
		return
	}
	args := []string{"-m", "rwkv_lab.posttrain_campaign", "--checkpoint", checkpoint,
		"--data", data, "--eval-data", eval, "--output", out, "--objectives", sig.Objectives,
		"--seeds", sig.Seeds, "--confirm-seeds", sig.Confirm, "--token-budget", budget,
		"--base-quantization", sig.Quant, "--quant-backend", sig.Backend,
		"--packing", sig.Packing, "--devices", sig.Devices,
		"--arm-timeout", strconv.FormatFloat(timeout, 'f', -1, 64), "--retries", retries}
	pid, err := s.spawnPy(args, fmt.Sprintf("posttrain_campaign_%d.log", time.Now().Unix()))
	if err != nil {
		toastErr(sse, "post-training campaign launch failed: "+err.Error())
		return
	}
	toastOK(sse, fmt.Sprintf("launched %d-objective post-training campaign (pid %d)", len(objectives), pid))
}

type pairedSignals struct {
	RunA, RunB, Prompt, Seed, MaxNew, Temp string
	OutA, OutB                             string
	Chosen                                 string
}

func readPairedSignals(r *http.Request) pairedSignals {
	var raw struct {
		RunA   string `json:"ptRunA"`
		RunB   string `json:"ptRunB"`
		Prompt string `json:"ptPrompt"`
		Seed   string `json:"ptSeed"`
		MaxNew string `json:"ptMaxNew"`
		Temp   string `json:"ptTemp"`
		OutA   string `json:"ptOutA"`
		OutB   string `json:"ptOutB"`
		Chosen string `json:"ptChosen"`
	}
	_ = datastar.ReadSignals(r, &raw)
	return pairedSignals{raw.RunA, raw.RunB, raw.Prompt, raw.Seed, raw.MaxNew, raw.Temp,
		raw.OutA, raw.OutB, raw.Chosen}
}

func (s *Server) handleComparePosttraining(w http.ResponseWriter, r *http.Request) {
	sig := readPairedSignals(r)
	sse := datastar.NewSSE(w, r)
	ckptA, errA := s.posttrainRunCheckpoint(sig.RunA)
	ckptB, errB := s.posttrainRunCheckpoint(sig.RunB)
	if errA != nil || errB != nil || strings.TrimSpace(sig.Prompt) == "" {
		toastErr(sse, "comparison needs two valid run names and a non-empty prompt")
		return
	}
	seed, err := boundedInt(sig.Seed, 0, 0, 2147483647)
	if err != nil {
		toastErr(sse, "invalid comparison seed")
		return
	}
	maxNew, err := boundedInt(sig.MaxNew, 200, 1, 2048)
	if err != nil {
		toastErr(sse, "invalid max-new value")
		return
	}
	temp := "0.8"
	if value, parseErr := strconv.ParseFloat(strings.TrimSpace(sig.Temp), 64); parseErr == nil && value >= 0 && value <= 4 {
		temp = strconv.FormatFloat(value, 'f', -1, 64)
	}
	_ = sse.MarshalAndPatchSignals(map[string]any{"ptBusy": true})
	a, err := s.generateCheckpoint(ckptA, sig.Prompt, maxNew, temp, seed)
	if err != nil {
		_ = sse.MarshalAndPatchSignals(map[string]any{"ptBusy": false})
		toastErr(sse, "run A: "+err.Error())
		return
	}
	b, err := s.generateCheckpoint(ckptB, sig.Prompt, maxNew, temp, seed)
	_ = sse.MarshalAndPatchSignals(map[string]any{"ptBusy": false})
	if err != nil {
		toastErr(sse, "run B: "+err.Error())
		return
	}
	_ = sse.MarshalAndPatchSignals(map[string]any{"ptOutA": a.Completion, "ptOutB": b.Completion,
		"ptChosen": "", "ptSeed": seed, "ptMaxNew": maxNew, "ptTemp": temp})
	html := `<div id="posttraining-compare" class="compare-grid"><div><b>A · ` + esc(sig.RunA) +
		`</b><pre class="sample-out">` + esc(a.Completion) + `</pre></div><div><b>B · ` + esc(sig.RunB) +
		`</b><pre class="sample-out">` + esc(b.Completion) + `</pre></div></div>`
	_ = sse.PatchElements(html)
	toastOK(sse, "paired generation complete")
}

type generationResult struct {
	Completion string `json:"completion"`
}

func (s *Server) generateCheckpoint(ckpt, prompt, maxNew, temp, seed string) (generationResult, error) {
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"), "-m", "rwkv_lab.generate",
		"--ckpt", ckpt, "--prompt", prompt, "--max-new", maxNew, "--temperature", temp,
		"--seed", seed, "--json")
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src")
	out, err := cmd.Output()
	if err != nil {
		return generationResult{}, fmt.Errorf("generation failed: %s", commandError(err))
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var result generationResult
	if len(lines) == 0 || json.Unmarshal([]byte(lines[len(lines)-1]), &result) != nil {
		return generationResult{}, fmt.Errorf("generator returned invalid output")
	}
	return result, nil
}

func (s *Server) posttrainRunCheckpoint(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || filepath.Base(name) != name || strings.ContainsAny(name, `/\\`) {
		return "", fmt.Errorf("invalid run name")
	}
	path := filepath.Join(s.cfg.RunsDir, name, "ckpt.pt")
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return "", fmt.Errorf("checkpoint missing")
	}
	return path, nil
}

func (s *Server) handlePosttrainingFeedback(w http.ResponseWriter, r *http.Request) {
	sig := readPairedSignals(r)
	sse := datastar.NewSSE(w, r)
	if sig.Chosen != "a" && sig.Chosen != "b" {
		toastErr(sse, "choose A or B first")
		return
	}
	if sig.OutA == "" || sig.OutB == "" || sig.OutA == sig.OutB || strings.TrimSpace(sig.Prompt) == "" {
		toastErr(sse, "run a non-identical paired comparison first")
		return
	}
	chosen, rejected := sig.OutA, sig.OutB
	chosenRun, rejectedRun := sig.RunA, sig.RunB
	if sig.Chosen == "b" {
		chosen, rejected = rejected, chosen
		chosenRun, rejectedRun = rejectedRun, chosenRun
	}
	record := map[string]any{"schema": "rwkv-lab.posttrain.v1", "kind": "preference", "split": "train",
		"id":       fmt.Sprintf("trainboard-%d", time.Now().UnixNano()),
		"messages": []map[string]string{{"role": "user", "content": sig.Prompt}},
		"chosen":   chosen, "rejected": rejected,
		"metadata": map[string]any{"source": "trainboard", "chosen_run": chosenRun,
			"rejected_run": rejectedRun, "seed": sig.Seed, "temperature": sig.Temp}}
	data, _ := json.Marshal(record)
	directory := filepath.Join(s.cfg.RepoRoot, "datasets")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		toastErr(sse, err.Error())
		return
	}
	path := filepath.Join(directory, "trainboard_preferences.jsonl")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err == nil {
		_, err = file.Write(append(data, '\n'))
		_ = file.Close()
	}
	if err != nil {
		toastErr(sse, "preference save failed: "+err.Error())
		return
	}
	toastOK(sse, "saved explicit training preference")
}

func commandError(err error) string {
	if exit, ok := err.(*exec.ExitError); ok && len(exit.Stderr) > 0 {
		return strings.TrimSpace(string(exit.Stderr))
	}
	return err.Error()
}

func shortHash(value string) string {
	if len(value) > 16 {
		return value[:16]
	}
	return value
}
