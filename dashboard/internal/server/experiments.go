package server

// Experiments panel: reads the rwkv-lab results registry (experiments.db, written by
// experiment.py / config.py) and the experiments/*.yaml config files, renders a ranked
// mean±std table with significance vs baseline, and offers a one-click launch of a config
// run (python -m rwkv_lab.config run <file>). Turns the CLI-first lab into a managed one.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/starfederation/datastar-go/datastar"
)

type expRow struct {
	Config    string
	Mean, Std float64
	Seeds     int
	Sha       string
}

// readRegistry returns task -> its config rows (latest per config), acc metric.
func (s *Server) readRegistry() (map[string][]expRow, error) {
	path := filepath.Join(s.cfg.RepoRoot, "experiments.db")
	if _, err := os.Stat(path); err != nil {
		return nil, nil // no registry yet
	}
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro&_pragma=busy_timeout(2000)")
	if err != nil {
		return nil, err
	}
	defer db.Close()
	// latest row per (task, config)
	rows, err := db.Query(`SELECT task, config, seeds, git_sha, metrics_json, MAX(ts)
		FROM results GROUP BY task, config ORDER BY task`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string][]expRow{}
	for rows.Next() {
		var task, config, sha, mj string
		var seeds int
		var ts float64
		if rows.Scan(&task, &config, &seeds, &sha, &mj, &ts) != nil {
			continue
		}
		var m map[string][]float64
		if json.Unmarshal([]byte(mj), &m) != nil {
			continue
		}
		acc, ok := m["acc"]
		if !ok || len(acc) < 2 {
			continue
		}
		out[task] = append(out[task], expRow{Config: config, Mean: acc[0], Std: acc[1], Seeds: seeds, Sha: sha})
	}
	return out, nil
}

func (s *Server) listConfigs() []string {
	dir := filepath.Join(s.cfg.RepoRoot, "experiments")
	var files []string
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if n := e.Name(); strings.HasSuffix(n, ".yaml") || strings.HasSuffix(n, ".yml") || strings.HasSuffix(n, ".json") {
			files = append(files, "experiments/"+n)
		}
	}
	sort.Strings(files)
	return files
}

func (s *Server) handleExperiments(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	reg, err := s.readRegistry()
	if err != nil {
		_ = sse.PatchElements(`<div id="experiments-body"><div class="empty">registry error: ` + esc(err.Error()) + `</div></div>`)
		return
	}
	var b strings.Builder
	b.WriteString(`<div id="experiments-body" class="exp-body">`)

	// launchable configs
	b.WriteString(`<div class="exp-configs"><div class="exp-h">config files</div>`)
	cfgs := s.listConfigs()
	if len(cfgs) == 0 {
		b.WriteString(`<div class="empty">no experiments/*.yaml</div>`)
	}
	for _, c := range cfgs {
		fmt.Fprintf(&b, `<div class="exp-cfg"><code>%s</code>`+
			`<button class="btn sm" data-on:click="@post('/api/experiments/run?cfg=%s')">run</button></div>`,
			esc(c), esc(c))
	}
	b.WriteString(`</div>`)

	// results by task, ranked, with significance vs baseline
	b.WriteString(`<div class="exp-results"><div class="exp-h">results (latest per config)</div>`)
	tasks := make([]string, 0, len(reg))
	for t := range reg {
		tasks = append(tasks, t)
	}
	sort.Strings(tasks)
	if len(tasks) == 0 {
		b.WriteString(`<div class="empty">no results yet — run a config</div>`)
	}
	for _, task := range tasks {
		rowsT := reg[task]
		sort.Slice(rowsT, func(i, j int) bool { return rowsT[i].Mean > rowsT[j].Mean })
		var base *expRow
		for i := range rowsT {
			if rowsT[i].Config == "baseline" {
				base = &rowsT[i]
			}
		}
		fmt.Fprintf(&b, `<div class="exp-task"><div class="exp-tname">%s</div><table class="exp-tbl">`, esc(task))
		for _, rw := range rowsT {
			delta := ""
			if base != nil && rw.Config != "baseline" {
				d := rw.Mean - base.Mean
				sig := math.Abs(d) > (rw.Std + base.Std)
				cls := "ns"
				if sig {
					cls = "sig"
				}
				delta = fmt.Sprintf(`<td class="%s">Δ%+.3f %s</td>`, cls, d, map[bool]string{true: "✓", false: "·"}[sig])
			} else {
				delta = `<td></td>`
			}
			fmt.Fprintf(&b, `<tr><td class="exp-cfgn">%s</td><td>%.3f±%.3f</td><td class="dim">n=%d @%s</td>%s</tr>`,
				esc(rw.Config), rw.Mean, rw.Std, rw.Seeds, esc(rw.Sha), delta)
		}
		b.WriteString(`</table></div>`)
	}
	b.WriteString(`</div></div>`)
	_ = sse.PatchElements(b.String())
}

// handleRunConfig launches `python -m rwkv_lab.config run <cfg>` detached. The cfg must be a
// file under experiments/ (no traversal) — same safety posture as the training-script allowlist.
func (s *Server) handleRunConfig(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	cfg := r.URL.Query().Get("cfg")
	if cfg == "" || strings.Contains(cfg, "..") || !strings.HasPrefix(cfg, "experiments/") ||
		!(strings.HasSuffix(cfg, ".yaml") || strings.HasSuffix(cfg, ".yml") || strings.HasSuffix(cfg, ".json")) {
		toastErr(sse, "run refused: cfg must be experiments/*.yaml")
		return
	}
	full := filepath.Join(s.cfg.RepoRoot, cfg)
	if _, err := os.Stat(full); err != nil {
		toastErr(sse, "run refused: "+cfg+" not found")
		return
	}
	base := strings.TrimSuffix(filepath.Base(cfg), filepath.Ext(cfg))
	logPath := filepath.Join(s.cfg.RunsDir, "config_"+base+".log")
	lf, err := os.Create(logPath)
	if err != nil {
		toastErr(sse, "run refused: "+err.Error())
		return
	}
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"),
		"-m", "rwkv_lab.config", "run", cfg)
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
	cmd.Stdout, cmd.Stderr = lf, lf
	if err := cmd.Start(); err != nil {
		lf.Close()
		toastErr(sse, "run failed: "+err.Error())
		return
	}
	go func() { _ = cmd.Wait(); lf.Close() }()
	toast(sse, fmt.Sprintf("launched %s (pid %d) — log %s", base, cmd.Process.Pid, filepath.Base(logPath)))
}
