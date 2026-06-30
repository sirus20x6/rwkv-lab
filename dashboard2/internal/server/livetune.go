package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/starfederation/datastar-go/datastar"
)

// controlWhitelist is the set of hyperparameters the dashboard may live-tune.
// The instrumented trainer applies only these keys. Loss weights and cadence
// are trivially hot-swappable; lr is direct-set (schedulefree) and lr_scale is
// a multiplier for scheduled optimizers.
var controlWhitelist = map[string]bool{
	"w_lmce": true, "w_block": true, "w_smt": true, "w_dmt": true,
	"lr": true, "lr_scale": true,
	"eval_every": true, "save_every": true, "log_every": true, "grad_clip": true,
	// grokking levers (convert_train.py): decoupled weight decay + tail ramp +
	// spectral (nuclear-norm) penalty. fixed_trainset is intentionally NOT here —
	// it sets the data regime at start and can't change mid-run.
	"weight_decay": true, "tail_weight_decay": true, "wd_tail_frac": true,
	"nuc_weight": true, "nuc_every": true,
	// best-model-fastest levers: GrokFast slow-gradient amplification + readout-group
	// LR boost. (--grokfast must be on for grokfast_lamb to apply; the autopilot can
	// also drive these.)
	"grokfast_lamb": true, "grokfast_alpha": true, "readout_lr_mult": true,
	// spectral_muon live knobs (convert_train_spectral.py): Muon^p exponent + amplifier.
	"sm_spectral_power": true, "sm_scale": true,
	// DDC strength + PC-Layer blend + LLR spread (convert_train_spectral.py).
	"ddc_strength": true, "pc_strength": true, "llr_smax": true,
	// relational / cross-arch distillation objective weights (convert_train_spectral.py).
	"w_cos": true, "w_cka": true, "w_flow": true, "w_bridge": true, "agreement_gate": true,
}

// handleSetControl atomically writes the edited overrides (the $ctl signal
// object) for a run. Only whitelisted, numeric, non-empty fields are written, so
// the user edits just the knobs they care about.
func (s *Server) handleSetControl(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var body struct {
		Ctl map[string]any `json:"ctl"`
	}
	_ = datastar.ReadSignals(r, &body)
	sse := datastar.NewSSE(w, r)

	kv := map[string]float64{}
	for k, v := range body.Ctl {
		if !controlWhitelist[k] {
			continue
		}
		switch t := v.(type) {
		case float64:
			kv[k] = t
		case string:
			if t == "" {
				continue
			}
			if f, err := strconv.ParseFloat(t, 64); err == nil {
				kv[k] = f
			}
		}
	}
	if len(kv) == 0 {
		toast(sse, "live-tune: no valid overrides entered")
		return
	}
	if err := s.db.SetControls(name, kv, nowTs()); err != nil {
		toast(sse, "live-tune failed: "+err.Error())
		return
	}
	argsJSON, _ := json.Marshal(kv)
	s.db.LogAction(nowTs(), "control", name, string(argsJSON), "queued (pending trainer apply)", 0)

	// stable, human-readable summary for the toast
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%s=%g", k, kv[k])
	}
	toast(sse, fmt.Sprintf("queued for %s: %s (pending next trainer poll)", name, strings.Join(parts, " ")))
}
