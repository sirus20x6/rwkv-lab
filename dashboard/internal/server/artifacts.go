package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// BestInfo is the authoritative best checkpoint a run restarts from
// (runs/<run>/best/best.json, atomically published by the trainers on improve).
type BestInfo struct {
	Step          int64
	PPL           float64
	Exists        bool
	ContractReset bool
}

func readBest(runDir string) BestInfo {
	// These directories are durable outcome markers created by the trainer when
	// it removes a winner from the active branch.  The old eval records remain in
	// train.jsonl, so their SQLite minimum is no longer an authoritative winner.
	// Keep the marker active after a new winner appears: otherwise an even lower
	// abandoned-branch metric would displace that new active checkpoint again.
	result := BestInfo{ContractReset: hasEvalContractReset(runDir)}
	bestDir := filepath.Join(runDir, "best")
	if info, err := os.Lstat(bestDir); err != nil || !info.Mode().IsDir() {
		return result
	}
	data, err := os.ReadFile(filepath.Join(bestDir, "best.json"))
	if err != nil {
		return result
	}
	var manifest map[string]json.RawMessage
	if json.Unmarshal(data, &manifest) != nil || manifest == nil {
		return result
	}
	stepRaw, ok := manifest["step"]
	if !ok {
		return result
	}
	step, err := strconv.ParseInt(strings.TrimSpace(string(stepRaw)), 10, 64)
	if err != nil || step < 0 {
		return result
	}

	// Match the trainer's manifest contract: every metric that is present must
	// be finite and in range, and at least loss or PPL must identify the eval.
	// PPL-only legacy manifests remain valid; a loss-only manifest can be
	// rendered without silently treating a missing JSON field as zero.
	lossRaw, hasLoss := manifest["loss"]
	pplRaw, hasPPL := manifest["ppl"]
	if !hasLoss && !hasPPL {
		return result
	}
	ppl := 0.0
	if hasLoss {
		loss, err := strconv.ParseFloat(strings.TrimSpace(string(lossRaw)), 64)
		if err != nil || math.IsNaN(loss) || math.IsInf(loss, 0) || loss < 0 {
			return result
		}
		ppl = math.Exp(math.Min(loss, 20))
	}
	if hasPPL {
		ppl, err = strconv.ParseFloat(strings.TrimSpace(string(pplRaw)), 64)
		if err != nil || math.IsNaN(ppl) || math.IsInf(ppl, 0) || ppl <= 0 {
			return result
		}
	}
	// New vision runs atomically publish a manifest pointing at an immutable
	// checkpoint. Validate that exact target; falling back to ckpt.pt when the
	// field exists but is missing/invalid could label old bytes with new metrics.
	checkpoint := filepath.Join(bestDir, "ckpt.pt")
	if checkpointRaw, present := manifest["checkpoint"]; present {
		var name string
		if json.Unmarshal(checkpointRaw, &name) != nil || name == "" ||
			filepath.Base(name) != name || filepath.Ext(name) != ".pt" {
			return result
		}
		checkpoint = filepath.Join(bestDir, name)
	}
	// Trainer publications are regular files (hardlinks for vision winners).
	// Lstat deliberately rejects symlinks, including a same-basename link that
	// escapes best/ after the containment check above.
	if info, err := os.Lstat(checkpoint); err != nil || !info.Mode().IsRegular() {
		return result
	}
	result.Step, result.PPL, result.Exists = step, ppl, true
	return result
}

func hasEvalContractReset(runDir string) bool {
	if reset, authoritative := readEvalContractResetReceipt(runDir); authoritative {
		return reset
	}
	// Pre-receipt runs still have the trainer's atomic quarantine rename as a
	// durable signal. This fallback also covers a crash after the rename but
	// before the separate receipt publication.
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return false
	}
	prefixes := [...]string{
		"best.before-explicit-resume-step-",
		"best.before-loop-reset-step-",
		"best.before-text-limit-",
		// Recognize the descriptive spelling as well as the current trainer's
		// text-limit label so older/alternate migration runs remain fail-closed.
		"best.before-text-migration-",
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		for _, prefix := range prefixes {
			if strings.HasPrefix(entry.Name(), prefix) {
				return true
			}
		}
	}
	return false
}

func readEvalContractResetReceipt(runDir string) (reset, authoritative bool) {
	path := filepath.Join(runDir, "eval_contract_reset.json")
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, false
		}
		return true, true
	}
	// A present but untrustworthy receipt fails closed: it must not restore a
	// potentially abandoned DB minimum. Atomic trainer writes make malformed
	// regular files impossible during ordinary publication.
	if !info.Mode().IsRegular() {
		return true, true
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return true, true
	}
	var receipt struct {
		Schema  int      `json:"schema"`
		Reset   *bool    `json:"reset"`
		Step    *int64   `json:"step"`
		Reasons []string `json:"reasons"`
	}
	if json.Unmarshal(data, &receipt) != nil || receipt.Schema != 1 ||
		receipt.Reset == nil || receipt.Step == nil || *receipt.Step < 0 ||
		len(receipt.Reasons) == 0 || !*receipt.Reset {
		return true, true
	}
	for _, reason := range receipt.Reasons {
		if strings.TrimSpace(reason) == "" {
			return true, true
		}
	}
	return true, true
}

// LoopRW mirrors runs/<run>/loop_rw.json (LoopedRWKV residual weights).
type LoopRWSplit struct {
	Heads          int         `json:"heads"`
	Channels       int         `json:"channels"`
	ChPerHead      int         `json:"ch_per_head"`
	ChannelBuckets int         `json:"channel_buckets"`
	HeadAbs        [][]float64 `json:"head_abs"`
	ChannelAbs     [][]float64 `json:"channel_abs"`
}

type LoopRWLayer struct {
	Layer         int          `json:"layer"`
	Label         string       `json:"label,omitempty"`
	MaxRW         float64      `json:"max_rw"`
	RW            []float64    `json:"rw"`
	ExpectedLoops float64      `json:"expected_loops,omitempty"`
	Split         *LoopRWSplit `json:"split,omitempty"`
}

type LoopRW struct {
	Step                    int64         `json:"step"`
	LoopCount               int           `json:"loop_count"`
	NLayers                 int           `json:"n_layers"`
	NPinned                 int           `json:"n_pinned"`
	MeanMaxRW               float64       `json:"mean_max_rw"`
	GateMode                string        `json:"gate_mode"`
	ModelFamily             string        `json:"model_family,omitempty"`
	AggregateOnly           bool          `json:"aggregate_only,omitempty"`
	MeanGateRMS             float64       `json:"mean_gate_rms,omitempty"`
	MeanExpectedLoops       float64       `json:"mean_expected_loops,omitempty"`
	ExecutedRefinementCalls int           `json:"executed_refinement_calls,omitempty"`
	MaximumRefinementCalls  int           `json:"maximum_refinement_calls,omitempty"`
	FactorCount             int           `json:"factor_count,omitempty"`
	ExecutedFactorCalls     int           `json:"executed_refinement_factor_calls,omitempty"`
	MaximumFactorCalls      int           `json:"maximum_refinement_factor_calls,omitempty"`
	TreadActiveFraction     float64       `json:"tread_active_fraction,omitempty"`
	ResidentDomain          string        `json:"resident_domain,omitempty"`
	Layers                  []LoopRWLayer `json:"layers"`
}

func readLoopRW(runDir string) (LoopRW, bool) {
	data, err := os.ReadFile(filepath.Join(runDir, "loop_rw.json"))
	if err != nil {
		return readMageFlowLoopFromTrainLog(runDir)
	}
	var lr LoopRW
	if json.Unmarshal(data, &lr) != nil ||
		(len(lr.Layers) == 0 && !lr.AggregateOnly) {
		return readMageFlowLoopFromTrainLog(runDir)
	}
	return lr, true
}

func readMageFlowLoopFromTrainLog(runDir string) (LoopRW, bool) {
	file, err := os.Open(filepath.Join(runDir, "train.jsonl"))
	if err != nil {
		return LoopRW{}, false
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() == 0 {
		return LoopRW{}, false
	}
	const tailBytes int64 = 2 << 20
	start := max(int64(0), info.Size()-tailBytes)
	if _, err := file.Seek(start, 0); err != nil {
		return LoopRW{}, false
	}
	data, err := io.ReadAll(file)
	if err != nil {
		return LoopRW{}, false
	}
	lines := bytes.Split(data, []byte{'\n'})
	for index := len(lines) - 1; index >= 0; index-- {
		var row struct {
			Kind                    string  `json:"kind"`
			Step                    int64   `json:"step"`
			Domain                  string  `json:"domain"`
			MeanGateRMS             float64 `json:"loop/mean_gate_rms"`
			MeanExpectedLoops       float64 `json:"loop/mean_expected_loops"`
			MaximumLoops            int     `json:"loop/maximum_loops_per_block"`
			ExecutedRefinementCalls int     `json:"loop/executed_refinement_block_calls"`
			MaximumRefinementCalls  int     `json:"loop/maximum_refinement_block_calls"`
			FactorCount             int     `json:"loop/factor_count"`
			ExecutedFactorCalls     int     `json:"loop/executed_refinement_factor_calls"`
			MaximumFactorCalls      int     `json:"loop/maximum_refinement_factor_calls"`
			TotalImageTokens        int     `json:"loop/total_image_tokens"`
			ActiveImageTokens       int     `json:"loop/active_tread_image_tokens"`
			BackboneBlocks          int     `json:"loop/original_backbone_blocks_executed"`
			ExpertBlocks            int     `json:"loop/expert_blocks_executed"`
		}
		if json.Unmarshal(lines[index], &row) != nil || row.Kind != "train" ||
			row.MaximumLoops <= 1 || row.MaximumRefinementCalls <= 0 {
			continue
		}
		activeFraction := 1.0
		if row.TotalImageTokens > 0 {
			activeFraction = float64(row.ActiveImageTokens) /
				float64(row.TotalImageTokens)
		}
		gateMode := "factored-head-channel"
		if row.FactorCount > 0 {
			gateMode = "executable-factored-head-channel"
		}
		return LoopRW{
			Step: row.Step, LoopCount: row.MaximumLoops,
			NLayers:  row.BackboneBlocks + row.ExpertBlocks,
			GateMode: gateMode, ModelFamily: "mageflow",
			AggregateOnly: true, MeanGateRMS: row.MeanGateRMS,
			MeanExpectedLoops:       row.MeanExpectedLoops,
			ExecutedRefinementCalls: row.ExecutedRefinementCalls,
			MaximumRefinementCalls:  row.MaximumRefinementCalls,
			FactorCount:             row.FactorCount,
			ExecutedFactorCalls:     row.ExecutedFactorCalls,
			MaximumFactorCalls:      row.MaximumFactorCalls,
			TreadActiveFraction:     activeFraction,
			ResidentDomain:          row.Domain,
		}, true
	}
	return LoopRW{}, false
}

// renderLoopRW paints the LoopedRWKV residual-weight panel: one bar per converted
// layer (max|residual_weight|; 0 = the loop collapsed to a single pass), with
// optional per-head/per-channel heat strips for split gate modes.
func renderLoopRW(lr LoopRW) string {
	var b strings.Builder
	mode := lr.GateMode
	if mode == "" {
		mode = "scalar"
	}
	if lr.ModelFamily == "mageflow" {
		factorSummary := ""
		if lr.MaximumFactorCalls > 0 {
			factorSummary = fmt.Sprintf(" · factors %d/%d",
				lr.ExecutedFactorCalls, lr.MaximumFactorCalls)
		}
		b.WriteString(`<div id="looprw-panel" class="panel"><div class="panel-title">loop utilization · MageFlow learned depth ` +
			fmt.Sprintf(`<span class="sub">step %d · resident %s · expected %.2f/%d passes · refinements %d/%d%s · TREAD active %.0f%% · %s</span></div>`,
				lr.Step, html.EscapeString(lr.ResidentDomain),
				lr.MeanExpectedLoops, lr.LoopCount,
				lr.ExecutedRefinementCalls, lr.MaximumRefinementCalls,
				factorSummary,
				100*lr.TreadActiveFraction, html.EscapeString(mode)))
		if lr.AggregateOnly {
			b.WriteString(`<div class="looprw-bars">`)
			renderLoopAggregateRow(&b, "gate RMS", lr.MeanGateRMS, 0.25)
			renderLoopAggregateRow(
				&b, "depth", lr.MeanExpectedLoops, float64(lr.LoopCount))
			renderLoopAggregateRow(
				&b, "refine", float64(lr.ExecutedRefinementCalls),
				float64(lr.MaximumRefinementCalls))
			if lr.MaximumFactorCalls > 0 {
				renderLoopAggregateRow(
					&b, "factors", float64(lr.ExecutedFactorCalls),
					float64(lr.MaximumFactorCalls))
			}
			renderLoopAggregateRow(&b, "TREAD", lr.TreadActiveFraction, 1)
			b.WriteString(`</div></div>`)
			return b.String()
		}
	} else {
		b.WriteString(`<div id="looprw-panel" class="panel"><div class="panel-title">loop usage · effective gate per layer ` +
			fmt.Sprintf(`<span class="sub">step %d · mean layer max %.3f · %d/%d pinned · %d passes · %s</span></div>`,
				lr.Step, lr.MeanMaxRW, lr.NPinned, lr.NLayers, lr.LoopCount, html.EscapeString(mode)))
	}
	rows := append([]LoopRWLayer{}, lr.Layers...)
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
		label := r.Label
		if label == "" {
			label = fmt.Sprintf("L%d", r.Layer)
		}
		value := fmt.Sprintf("%.4f", r.MaxRW)
		if r.ExpectedLoops > 0 {
			value += fmt.Sprintf(" · %.2fx", r.ExpectedLoops)
		}
		fmt.Fprintf(&b,
			`<div class="looprw-row" title="rw=[%s]"><span class="lrw-l">%s</span>`+
				`<span class="lrw-track"><i class="%s" style="width:%.1f%%"></i></span>`+
				`<span class="lrw-v">%s</span></div>`,
			html.EscapeString(strings.Join(passes, ", ")),
			html.EscapeString(label), cls, pct, html.EscapeString(value))
		if r.Split != nil {
			renderLoopRWSplit(&b, r.Split)
		}
	}
	b.WriteString(`</div></div>`)
	return b.String()
}

func renderLoopAggregateRow(
	b *strings.Builder, label string, value, maximum float64,
) {
	percent := 0.0
	if maximum > 0 {
		percent = 100 * value / maximum
	}
	percent = min(100, max(0, percent))
	fmt.Fprintf(
		b,
		`<div class="looprw-row"><span class="lrw-l wide">%s</span>`+
			`<span class="lrw-track"><i class="lo" style="width:%.1f%%"></i></span>`+
			`<span class="lrw-v">%.4f</span></div>`,
		html.EscapeString(label), percent, value,
	)
}

func renderLoopRWSplit(b *strings.Builder, sp *LoopRWSplit) {
	if sp.Heads <= 0 || sp.Channels <= 0 {
		return
	}
	fmt.Fprintf(b, `<div class="looprw-split"><div class="lrw-split-meta">heads %d x %d ch · channel buckets %d</div>`,
		sp.Heads, sp.ChPerHead, sp.ChannelBuckets)
	for i := range sp.HeadAbs {
		renderLoopRWSummary(b, fmt.Sprintf("p%d heads", i+1), sp.HeadAbs[i])
		if i < len(sp.ChannelAbs) {
			renderLoopRWSummary(b, fmt.Sprintf("p%d channels", i+1), sp.ChannelAbs[i])
		}
	}
	b.WriteString(`</div>`)
}

func renderLoopRWSummary(b *strings.Builder, label string, vals []float64) {
	if len(vals) == 0 {
		return
	}
	mean, maximum, active := 0.0, 0.0, 0
	for _, v := range vals {
		if v < 0 {
			v = -v
		}
		mean += v
		if v > maximum {
			maximum = v
		}
		if v >= 1e-4 {
			active++
		}
	}
	mean /= float64(len(vals))
	fmt.Fprintf(b,
		`<div class="lrw-split-meta">%s · mean %.4f · max %.4f · active %.0f%%</div>`,
		html.EscapeString(label), mean, maximum,
		100*float64(active)/float64(len(vals)))
}

// emptyLoopRW hides the panel when a run has no loop_rw.json.
func emptyLoopRW() string {
	return `<div id="looprw-panel" class="panel hidden"></div>`
}
