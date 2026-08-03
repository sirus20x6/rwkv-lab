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
	b.WriteString(`<div class="rlvr-build"><div class="exp-h">RLVR execution retired</div>` +
		`<div class="empty">This board is read-only. No descriptor-backed RLVR campaign operation is available yet; persisted campaign and recursive-lineage evidence remains visible below.</div></div>`)
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

func (s *Server) handleLaunchRLVR(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "legacy RLVR campaign launch is retired; no descriptor-backed operation is available", http.StatusGone)
}
