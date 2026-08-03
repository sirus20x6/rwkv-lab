package server

import (
	"net/http"
	"strings"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

func nowTs() float64 { return float64(time.Now().UnixNano()) / 1e9 }

// toast pushes a transient status line to the UI via the $toast signal.
// $toastKind colors it: "" neutral info, "ok" green success, "err" red failure.
func toast(sse *datastar.ServerSentEventGenerator, msg string) { toastKind(sse, "", msg) }

func toastOK(sse *datastar.ServerSentEventGenerator, msg string) { toastKind(sse, "ok", msg) }

func toastErr(sse *datastar.ServerSentEventGenerator, msg string) { toastKind(sse, "err", msg) }

func toastKind(sse *datastar.ServerSentEventGenerator, kind, msg string) {
	_ = sse.MarshalAndPatchSignals(map[string]any{"toast": msg, "toastKind": kind})
}

// handleNotes / handleTags persist user annotations on a run.
func (s *Server) handleNotes(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var sig struct {
		Notes string `json:"notes"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	if err := s.db.SetNotes(name, sig.Notes); err != nil {
		toastErr(sse, "notes save failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "notes", name, "{}", "saved", 0)
	toastOK(sse, "notes saved for "+name)
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var sig struct {
		Tags string `json:"tags"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	// store the raw comma/space list as a JSON array
	tags := splitTags(sig.Tags)
	tagsJSON := "[" + strings.Join(quoteAll(tags), ",") + "]"
	if err := s.db.SetTags(name, tagsJSON); err != nil {
		toastErr(sse, "tags save failed: "+err.Error())
		return
	}
	s.db.LogAction(nowTs(), "tags", name, tagsJSON, "saved", 0)
	toastOK(sse, "tags saved for "+name)
}

// ---- helpers ----

func splitTags(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' })
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

func quoteAll(items []string) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = `"` + strings.ReplaceAll(it, `"`, `\"`) + `"`
	}
	return out
}
