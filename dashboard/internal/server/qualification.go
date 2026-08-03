package server

// Production qualification receipt panel. Qualification execution is no longer
// owned by the dashboard; this file only discovers and renders persisted evidence.

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

type qualificationReport struct {
	Environment map[string]any            `json:"environment"`
	Reports     map[string]map[string]any `json:"reports"`
	Metrics     map[string]any            `json:"metrics"`
	Adopted     []string                  `json:"adopted"`
	Gate        map[string]any            `json:"regression_gate"`
}

type qualificationReceipt struct {
	Path    string
	ModTime time.Time
	Report  qualificationReport
}

func (s *Server) qualificationReceipts() []qualificationReceipt {
	return s.cachedDiscovery("qualification-receipts", 2*time.Second, func() any {
		var receipts []qualificationReceipt
		_ = filepath.WalkDir(s.cfg.RunsDir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry == nil {
				return nil
			}
			if entry.IsDir() {
				rel, _ := filepath.Rel(s.cfg.RunsDir, path)
				if rel != "." && strings.Count(rel, string(filepath.Separator)) > 4 {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(strings.ToLower(entry.Name()), ".json") {
				return nil
			}
			data, readErr := os.ReadFile(path)
			if readErr != nil || !strings.Contains(string(data), "rwkv-lab.production-kernel-qualification.v1") {
				return nil
			}
			var report qualificationReport
			if json.Unmarshal(data, &report) != nil || report.Reports == nil {
				return nil
			}
			info, infoErr := entry.Info()
			if infoErr != nil {
				return nil
			}
			rel, _ := filepath.Rel(s.cfg.RepoRoot, path)
			receipts = append(receipts, qualificationReceipt{
				Path: filepath.ToSlash(rel), ModTime: info.ModTime(), Report: report})
			return nil
		})
		sort.Slice(receipts, func(i, j int) bool { return receipts[i].ModTime.After(receipts[j].ModTime) })
		if len(receipts) > 20 {
			receipts = receipts[:20]
		}
		return receipts
	}).([]qualificationReceipt)
}

func metricFloat(values map[string]any, key string) (float64, bool) {
	value, ok := values[key]
	if !ok {
		return 0, false
	}
	number, ok := value.(float64)
	return number, ok
}

func (s *Server) handleQualification(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)
	var b strings.Builder
	b.WriteString(`<div id="qualification-body" class="qualification-body"><div class="qual-build"><div class="exp-h">qualification execution retired</div>`)
	b.WriteString(`<div class="empty">This board is read-only. No descriptor-backed qualification operation is available yet; run qualification outside the dashboard and inspect its persisted receipt here.</div></div>`)
	b.WriteString(`<div class="qual-results"><div class="exp-h">qualification receipts <button class="btn sm" data-on:click="@get('/api/qualification')">refresh</button></div>`)
	receipts := s.qualificationReceipts()
	if len(receipts) == 0 {
		b.WriteString(`<div class="empty">no persisted qualification receipts yet</div>`)
	}
	for _, receipt := range receipts {
		device := fmt.Sprint(receipt.Report.Environment["device_name"])
		if device == "<nil>" || strings.TrimSpace(device) == "" {
			device = fmt.Sprint(receipt.Report.Environment["device"])
		}
		gate := "not compared"
		gateClass := "dim"
		if passed, ok := receipt.Report.Gate["passed"].(bool); ok {
			if passed {
				gate, gateClass = "baseline passed", "sig"
			} else {
				gate, gateClass = "baseline failed", "ns"
			}
		}
		fmt.Fprintf(&b, `<div class="rlvr-campaign"><div class="exp-tname"><code>%s</code> <span class="%s">%s</span><span class="dim"> · %s · adopted %s</span></div>`,
			esc(receipt.Path), gateClass, gate, esc(device), esc(strings.Join(receipt.Report.Adopted, ", ")))
		b.WriteString(`<table class="exp-tbl"><tr class="exp-hd"><td>backend</td><td>available</td><td>parity/exact</td><td>speedup</td><td>launches</td><td>path µs</td><td>GPU µs/top</td><td>compile</td><td>memory</td><td>adopted</td></tr>`)
		names := make([]string, 0, len(receipt.Report.Reports))
		for name := range receipt.Report.Reports {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			report := receipt.Report.Reports[name]
			available, _ := report["available"].(bool)
			adopted, _ := report["adopted"].(bool)
			parity := report["parity_passed"]
			if parity == nil {
				parity = report["exact"]
			}
			if parity == nil {
				parity = report["exact_tokens"]
			}
			speed := "—"
			if value, ok := metricFloat(report, "speedup"); ok {
				speed = fmt.Sprintf("%.2fx", value)
			}
			memory := "—"
			if value, ok := metricFloat(report, "production_memory_fraction"); ok {
				memory = fmt.Sprintf("%.2f%%", value*100)
			}
			launches := "—"
			if before, beforeOK := metricFloat(report, "cuda_kernels_before"); beforeOK {
				if after, afterOK := metricFloat(report, "cuda_kernels_after"); afterOK {
					launches = fmt.Sprintf("%.0f→%.0f", before, after)
				}
			}
			compile := "—"
			if plan, ok := report["plan"].(map[string]any); ok {
				if seconds, secondsOK := metricFloat(plan, "compile_seconds"); secondsOK {
					compile = fmt.Sprintf("%.1fs", seconds)
				}
			}
			pathLatency := "—"
			kernelEvidence := "—"
			if ablation, ok := report["ablation"].(map[string]any); ok {
				if paths, pathsOK := ablation["paths"].(map[string]any); pathsOK {
					values := make([]string, 0, 4)
					for _, pathName := range []string{"eager_reference", "fused_state_epilogue", "compiled_fullgraph", "cuda_graph"} {
						path, pathOK := paths[pathName].(map[string]any)
						if micros, microsOK := metricFloat(path, "median_us"); pathOK && microsOK {
							values = append(values, fmt.Sprintf("%.0f", micros))
						}
					}
					if len(values) == 4 {
						pathLatency = strings.Join(values, "→")
					}
					if graph, graphOK := paths["cuda_graph"].(map[string]any); graphOK {
						if micros, microsOK := metricFloat(graph, "cuda_time_us"); microsOK {
							kernelEvidence = fmt.Sprintf("%.0f", micros)
							if top, topOK := graph["top_cuda_kernels"].([]any); topOK && len(top) > 0 {
								if first, firstOK := top[0].(map[string]any); firstOK {
									name := fmt.Sprint(first["name"])
									if len(name) > 28 {
										name = name[:28] + "…"
									}
									kernelEvidence += " · " + name
								}
							}
						}
					}
				}
			}
			fmt.Fprintf(&b, `<tr><td><code>%s</code></td><td>%t</td><td>%v</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class="%s">%t</td></tr>`,
				esc(name), available, parity, speed, launches, pathLatency, esc(kernelEvidence), compile, memory, map[bool]string{true: "sig", false: "ns"}[adopted], adopted)
		}
		b.WriteString(`</table></div>`)
	}
	b.WriteString(`</div></div>`)
	_ = sse.PatchElements(b.String())
}

func (s *Server) handleRunQualification(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "dashboard qualification execution is retired; no descriptor-backed operation is available", http.StatusGone)
}
