package server

// Post-training evidence panel. Dataset inspection and persisted campaign evidence
// remain available, while dataset mutation, campaign launch, generation, and
// preference writes have moved out of dashboard authority.

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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
	b.WriteString(`<div class="empty">Dataset versioning is read-only here. Create immutable dataset versions through an external or future descriptor-backed operation.</div>`)
	b.WriteString(`<div id="posttraining-inspect"><div class="empty">select a repository JSONL dataset</div></div>`)
	b.WriteString(`<div class="panel-title">post-training execution <span class="sub">declarative single-run adapter training</span></div>`)
	b.WriteString(`<div class="empty">The descriptor-backed <code>rwkv-lab.rwkv-posttraining@1.0.0</code> operation is available in the <a href="#trainvm-authoring">TrainVM composer</a> for SFT, DPO, KTO, ORPO, SimPO, reward-model, and PRM runs. Legacy multi-arm campaigns and recursive promotion remain read-only below.</div>`)
	b.WriteString(`<div class="panel-title">legacy post-training campaigns · read-only <span class="sub">persisted evidence and promotion receipts</span></div>`)
	b.WriteString(`<div class="pt-campaign-results">`)
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
	b.WriteString(`</div>`)
	b.WriteString(`<div class="panel-title">paired behavior · read-only</div>`)
	b.WriteString(`<div class="empty">Dashboard generation and preference writes are retired. No descriptor-backed paired-generation operation is available yet.</div></div>`)
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

func (s *Server) handleVersionPosttraining(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "dashboard dataset versioning is retired; no descriptor-backed operation is available", http.StatusGone)
}

func (s *Server) handleLaunchPosttrainingCampaign(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "dashboard post-training campaign launch is retired; no descriptor-backed operation is available", http.StatusGone)
}

func (s *Server) handleComparePosttraining(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "dashboard paired generation is retired; no descriptor-backed operation is available", http.StatusGone)
}

func (s *Server) handlePosttrainingFeedback(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "dashboard preference mutation is retired; no descriptor-backed operation is available", http.StatusGone)
}

// posttrainRunCheckpoint remains a read-only inspector used by legacy evidence tests.
// It deliberately does not map the sampled run name to a TrainVM run identity.
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
