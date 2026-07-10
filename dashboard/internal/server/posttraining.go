package server

// Post-training data + behavior panel. Dataset paths are repository-confined and validation is
// delegated to rwkv_lab.posttrain_data. Paired generation uses the same prompt, seed, temperature,
// and token budget for both checkpoints. Explicit operator choices append training preferences;
// they never enter held-out evaluation data and never trigger training or publication.

import (
	"encoding/json"
	"fmt"
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
	b.WriteString(`<div class="panel-title">paired behavior <span class="sub">same prompt · seed · sampling settings · explicit preference capture</span></div>`)
	b.WriteString(`<div class="ctl-row"><input class="ctl-input" placeholder="run A" data-bind="ptRunA"><input class="ctl-input" placeholder="run B" data-bind="ptRunB"></div>`)
	b.WriteString(`<textarea class="ctl-input" rows="2" placeholder="comparison prompt" data-bind="ptPrompt"></textarea>`)
	b.WriteString(`<div class="ctl-row"><input class="ctl-input" style="max-width:90px" title="seed" data-bind="ptSeed"><input class="ctl-input" style="max-width:100px" title="max new" data-bind="ptMaxNew"><input class="ctl-input" style="max-width:90px" title="temperature" data-bind="ptTemp"><button class="btn" data-attr-disabled="$ptBusy" data-on:click="@post('/api/posttraining/compare')">compare</button></div>`)
	b.WriteString(`<div id="posttraining-compare"><div class="empty">choose two completed LM runs with ckpt.pt</div></div>`)
	b.WriteString(`<div class="ctl-row"><select class="ctl-input" data-bind="ptChosen"><option value="">choose preferred output</option><option value="a">A preferred</option><option value="b">B preferred</option></select><button class="btn" data-on:click="confirm('Append this pair to datasets/trainboard_preferences.jsonl?') && @post('/api/posttraining/feedback')">save training preference</button></div></div>`)
	_ = sse.PatchElements(b.String())
}

func (s *Server) posttrainDatasets() []string {
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
