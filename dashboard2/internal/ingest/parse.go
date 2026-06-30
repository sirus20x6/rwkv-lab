// Package ingest tails runs/*/train.jsonl into SQLite by byte offset, parsing
// each JSONL line into a typed train/eval/checkpoint row. Known fields become
// columns; everything else is preserved verbatim in extra_json so new
// instrumentation fields need no schema change.
package ingest

import (
	"bytes"
	"encoding/json"
	"regexp"

	"trainboard/internal/db"
)

type kind int

const (
	kindUnknown kind = iota
	kindTrain
	kindEval
	kindCheckpoint
)

// event is exactly one of Train/Eval/Ckpt (per Kind).
type event struct {
	Kind  kind
	Train db.TrainRow
	Eval  db.EvalRow
	Ckpt  db.CkptRow
}

// Fields promoted to columns (excluded from extra_json).
var trainKnown = map[string]bool{
	"kind": true, "step": true, "loss": true, "lr": true,
	"gnorm": true, "tok_per_sec": true, "skipped": true,
}
var evalKnown = map[string]bool{
	"kind": true, "step": true, "loss": true, "ppl": true,
	"top1_acc": true, "top5_acc": true,
}

// Python's json.dumps can emit bare NaN/Infinity/-Infinity tokens (invalid
// JSON). Go's encoding/json rejects them and would drop the whole line. Replace
// such value-position tokens with null before parsing. Anchored on the chars
// that legally precede a JSON value (: , [) so we never touch string contents.
var nonFiniteRe = regexp.MustCompile(`([:\[,]\s*)(-?Infinity|NaN)\b`)

func sanitize(raw []byte) []byte {
	if !bytes.Contains(raw, []byte("NaN")) && !bytes.Contains(raw, []byte("Infinity")) {
		return raw // fast path: the overwhelming majority of lines
	}
	return nonFiniteRe.ReplaceAll(raw, []byte("${1}null"))
}

// fptr returns a *float64 for a numeric JSON value, or nil if absent/null/non-numeric.
func fptr(m map[string]any, key string) *float64 {
	v, ok := m[key]
	if !ok || v == nil {
		return nil
	}
	if f, ok := v.(float64); ok {
		return &f
	}
	return nil
}

func istep(m map[string]any) (int64, bool) {
	v, ok := m["step"]
	if !ok {
		return 0, false
	}
	f, ok := v.(float64)
	if !ok {
		return 0, false
	}
	return int64(f), true
}

func truthy(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case float64:
		return t != 0
	default:
		return false
	}
}

// extraJSON marshals all keys not in `known` into a compact JSON object.
// Returns "" when nothing is left over.
func extraJSON(m map[string]any, known map[string]bool) string {
	leftover := make(map[string]any, len(m))
	for k, v := range m {
		if known[k] {
			continue
		}
		leftover[k] = v
	}
	if len(leftover) == 0 {
		return ""
	}
	b, err := json.Marshal(leftover)
	if err != nil {
		return ""
	}
	return string(b)
}

// parseLine parses one JSONL line. ok=false means skip (blank, parse error,
// unknown kind, or missing step). ts is attached to train/eval rows.
func parseLine(raw []byte, ts float64) (event, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return event{}, false
	}
	var m map[string]any
	if err := json.Unmarshal(sanitize(raw), &m); err != nil {
		return event{}, false
	}
	step, hasStep := istep(m)
	if !hasStep {
		return event{}, false
	}
	switch m["kind"] {
	case "train":
		return event{Kind: kindTrain, Train: db.TrainRow{
			Step: step, Loss: fptr(m, "loss"), LR: fptr(m, "lr"),
			Gnorm: fptr(m, "gnorm"), TokPerSec: fptr(m, "tok_per_sec"),
			Skipped: truthy(m["skipped"]), Extra: extraJSON(m, trainKnown), TS: ts,
		}}, true
	case "eval":
		return event{Kind: kindEval, Eval: db.EvalRow{
			Step: step, Loss: fptr(m, "loss"), PPL: fptr(m, "ppl"),
			Top1: fptr(m, "top1_acc"), Top5: fptr(m, "top5_acc"),
			Extra: extraJSON(m, evalKnown), TS: ts,
		}}, true
	case "checkpoint":
		reason, _ := m["reason"].(string)
		return event{Kind: kindCheckpoint, Ckpt: db.CkptRow{Step: step, Reason: reason}}, true
	default:
		return event{}, false
	}
}
