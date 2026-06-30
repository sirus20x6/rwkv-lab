package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/starfederation/datastar-go/datastar"

	"trainboard/internal/convboard"
)

// lbRow is one leaderboard entry (run + its headline metrics).
type lbRow struct {
	Name     string
	Layer    int
	HasLayer bool
	LastPPL  *float64
	BestPPL  *float64
	BestTop1 *float64
	NTrain   int
	NEval    int
	Status   string
	AgeS     float64
}

// handleLeaderboard renders a sortable table of all runs. ?sort= one of
// ppl|top1|layer|updated|name|events (default best ppl ascending).
func (s *Server) handleLeaderboard(w http.ResponseWriter, r *http.Request) {
	now := nowTs()
	summaries, err := s.db.RunSummaries(now)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	rows := make([]lbRow, 0, len(summaries))
	for _, su := range summaries {
		row := lbRow{Name: su.Name, LastPPL: su.LatestPPL, NTrain: su.NTrain, NEval: su.NEval,
			Status: su.Status, AgeS: now - su.LastUpdateTs}
		if L, ok := convboard.RunLayer(filepath.Join(s.cfg.RunsDir, su.Name)); ok {
			row.Layer, row.HasLayer = L, true
		}
		if k, ok, _ := s.db.RunKPIsByName(su.Name); ok {
			row.BestPPL, row.BestTop1 = k.BestPPL, k.BestTop1
		}
		rows = append(rows, row)
	}
	sortLeaderboard(rows, r.URL.Query().Get("sort"))

	sse := datastar.NewSSE(w, r)
	_ = sse.PatchElements(renderLeaderboard(rows))
}

func sortLeaderboard(rows []lbRow, key string) {
	less := map[string]func(a, b lbRow) bool{
		"ppl":     func(a, b lbRow) bool { return fOr(a.BestPPL, 1e18) < fOr(b.BestPPL, 1e18) },
		"top1":    func(a, b lbRow) bool { return fOr(a.BestTop1, -1) > fOr(b.BestTop1, -1) },
		"layer":   func(a, b lbRow) bool { return a.Layer < b.Layer },
		"updated": func(a, b lbRow) bool { return a.AgeS < b.AgeS },
		"name":    func(a, b lbRow) bool { return a.Name < b.Name },
		"events":  func(a, b lbRow) bool { return a.NTrain > b.NTrain },
	}[key]
	if less == nil {
		less = func(a, b lbRow) bool { return fOr(a.BestPPL, 1e18) < fOr(b.BestPPL, 1e18) }
	}
	sort.SliceStable(rows, func(i, j int) bool { return less(rows[i], rows[j]) })
}

func renderLeaderboard(rows []lbRow) string {
	hdr := func(key, label string) string {
		return fmt.Sprintf(`<th data-on:click="@get('/api/leaderboard?sort=%s')">%s</th>`, key, label)
	}
	var b strings.Builder
	b.WriteString(`<div id="leaderboard"><table class="lb"><thead><tr>`)
	b.WriteString(hdr("name", "run") + hdr("layer", "L") + hdr("ppl", "best ppl") +
		`<th>last ppl</th>` + hdr("top1", "best top1") + hdr("events", "events") +
		hdr("updated", "updated") + `<th>status</th></tr></thead><tbody>`)
	for _, r := range rows {
		layer := "—"
		if r.HasLayer {
			layer = fmt.Sprintf("%d", r.Layer)
		}
		fmt.Fprintf(&b,
			`<tr data-on:click="$selectedRun='%s'; @get('/api/run/%s')">`+
				`<td class="lb-name">%s</td><td>%s</td><td class="ok">%s</td><td>%s</td>`+
				`<td>%s</td><td>%d/%d</td><td>%s ago</td><td><span class="dot %s"></span> %s</td></tr>`,
			jsName(r.Name), urlName(r.Name), esc(r.Name), layer,
			fmtPtr(r.BestPPL, 3), fmtPtr(r.LastPPL, 3), fmtPctPtr(r.BestTop1),
			r.NTrain, r.NEval, fmtAge(&r.AgeS), r.Status, r.Status)
	}
	b.WriteString(`</tbody></table></div>`)
	return b.String()
}

// handleDiff renders the differing config keys between two runs' sidecars.
func (s *Server) handleDiff(w http.ResponseWriter, r *http.Request) {
	a := r.URL.Query().Get("a")
	bRun := r.URL.Query().Get("b")
	ca := s.readSidecarConfig(a)
	cb := s.readSidecarConfig(bRun)
	sse := datastar.NewSSE(w, r)
	_ = sse.PatchElements(renderDiff(a, bRun, ca, cb))
}

func (s *Server) readSidecarConfig(run string) map[string]any {
	if run == "" {
		return nil
	}
	cfgPath, _ := latestSidecarPath(filepath.Join(s.cfg.RunsDir, run))
	if cfgPath == "" {
		return nil
	}
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil
	}
	var sc struct {
		Config map[string]any `json:"config"`
	}
	if json.Unmarshal(data, &sc) != nil || sc.Config == nil {
		// some sidecars are the flat config itself
		var flat map[string]any
		if json.Unmarshal(data, &flat) == nil {
			return flat
		}
		return nil
	}
	return sc.Config
}

func renderDiff(a, b string, ca, cb map[string]any) string {
	var bld strings.Builder
	bld.WriteString(fmt.Sprintf(`<div id="diff-out"><div class="diff-head">config diff · <b>%s</b> vs <b>%s</b></div>`, esc(a), esc(b)))
	if ca == nil || cb == nil {
		bld.WriteString(`<div class="muted">need two runs with sidecar config.json</div></div>`)
		return bld.String()
	}
	keys := map[string]bool{}
	for k := range ca {
		keys[k] = true
	}
	for k := range cb {
		keys[k] = true
	}
	var ks []string
	for k := range keys {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	bld.WriteString(`<table class="lb diff"><thead><tr><th>key</th><th>` + esc(a) + `</th><th>` + esc(b) + `</th></tr></thead><tbody>`)
	n := 0
	for _, k := range ks {
		va, vb := fmt.Sprint(ca[k]), fmt.Sprint(cb[k])
		if va == vb {
			continue
		}
		fmt.Fprintf(&bld, `<tr><td class="lb-name">%s</td><td>%s</td><td class="warn">%s</td></tr>`, esc(k), esc(va), esc(vb))
		n++
	}
	if n == 0 {
		bld.WriteString(`<tr><td colspan="3" class="muted">configs identical</td></tr>`)
	}
	bld.WriteString(`</tbody></table></div>`)
	return bld.String()
}

// ---- helpers ----

func fOr(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}
func fmtPtr(p *float64, d int) string {
	if p == nil {
		return "—"
	}
	return fmt.Sprintf("%.*f", d, *p)
}
func fmtPctPtr(p *float64) string {
	if p == nil {
		return "—"
	}
	return fmt.Sprintf("%.1f%%", 100**p)
}

// latestSidecarPath returns the newest step_*/config.json path for a run dir.
func latestSidecarPath(runDir string) (string, float64) {
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return "", 0
	}
	var best string
	var bestMtime float64
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "step_") {
			continue
		}
		p := filepath.Join(runDir, e.Name(), "config.json")
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		if mt := float64(info.ModTime().UnixNano()) / 1e9; mt >= bestMtime {
			bestMtime, best = mt, p
		}
	}
	return best, bestMtime
}
