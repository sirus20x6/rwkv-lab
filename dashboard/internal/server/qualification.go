package server

// Production qualification panel. Hardware paths remain fail-closed in Python;
// this UI only launches the allowlisted qualification command and renders its
// persisted evidence. It never changes backend adoption or promotes a model.

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
	b.WriteString(`<div id="qualification-body" class="qualification-body"><div class="qual-build"><div class="exp-h">new qualification</div><table class="field-tbl">`)
	b.WriteString(`<tr><td class="f-l">device</td><td><input data-bind="qualDevice" value="cuda"></td><td class="f-d">auto, cpu, cuda, or cuda:N</td></tr>`)
	b.WriteString(`<tr><td class="f-l">checkpoint</td><td><input data-bind="qualCheckpoint" placeholder="optional runs/name/ckpt.pt"></td><td class="f-d">enables recurrent serving qualification</td></tr>`)
	b.WriteString(`<tr><td class="f-l">prompt ids</td><td><input data-bind="qualPrompt" value="1,2,3,4"></td><td class="f-d">integer recurrent-decoding prompt</td></tr>`)
	b.WriteString(`<tr><td class="f-l">megakernel tuning</td><td><select data-bind="qualMegakernelMode"><option value="max-autotune-no-cudagraphs">max autotune</option><option value="default">fast compile</option></select></td><td class="f-d">Inductor plan search; the outer CUDA Graph is always captured</td></tr>`)
	b.WriteString(`<tr><td class="f-l">repeats / max new</td><td><input type="number" min="1" max="20" data-bind="qualRepeats" value="5"><input type="number" min="1" max="2048" data-bind="qualMaxNew" value="32"></td><td class="f-d">median timing samples · serving token budget</td></tr>`)
	b.WriteString(`<tr><td class="f-l">baseline</td><td><input data-bind="qualBaseline" placeholder="optional prior receipt JSON"></td><td class="f-d">fails on lost adoption or performance regression</td></tr>`)
	b.WriteString(`<tr><td class="f-l">regression limits</td><td><input data-bind="qualThroughput" value="0.05" title="throughput"><input data-bind="qualMemory" value="0.10" title="memory"><input data-bind="qualKernels" value="0.10" title="kernel count"></td><td class="f-d">fractional throughput · memory · launch limits</td></tr>`)
	b.WriteString(`<tr><td class="f-l">output</td><td><input data-bind="qualOutput" value="runs/qualification/kernel-qualification.json"></td><td class="f-d">persisted receipt under runs/</td></tr>`)
	b.WriteString(`</table><button class="btn" data-on:click="confirm('Run production kernel qualification?') && @post('/api/qualification/run')">▶ qualify backends</button></div>`)
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
		b.WriteString(`<table class="exp-tbl"><tr class="exp-hd"><td>backend</td><td>available</td><td>parity/exact</td><td>speedup</td><td>launches</td><td>compile</td><td>memory</td><td>adopted</td></tr>`)
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
			fmt.Fprintf(&b, `<tr><td><code>%s</code></td><td>%t</td><td>%v</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class="%s">%t</td></tr>`,
				esc(name), available, parity, speed, launches, compile, memory, map[bool]string{true: "sig", false: "ns"}[adopted], adopted)
		}
		b.WriteString(`</table></div>`)
	}
	b.WriteString(`</div></div>`)
	_ = sse.PatchElements(b.String())
}

func (s *Server) handleRunQualification(w http.ResponseWriter, r *http.Request) {
	var signals struct {
		Device     string `json:"qualDevice"`
		Checkpoint string `json:"qualCheckpoint"`
		Prompt     string `json:"qualPrompt"`
		MegaMode   string `json:"qualMegakernelMode"`
		Repeats    string `json:"qualRepeats"`
		MaxNew     string `json:"qualMaxNew"`
		Baseline   string `json:"qualBaseline"`
		Throughput string `json:"qualThroughput"`
		Memory     string `json:"qualMemory"`
		Kernels    string `json:"qualKernels"`
		Output     string `json:"qualOutput"`
	}
	_ = datastar.ReadSignals(r, &signals)
	sse := datastar.NewSSE(w, r)
	device := strings.TrimSpace(signals.Device)
	if device != "auto" && device != "cpu" && device != "cuda" && !strings.HasPrefix(device, "cuda:") {
		toastErr(sse, "qualification device must be auto, cpu, cuda, or cuda:N")
		return
	}
	if strings.HasPrefix(device, "cuda:") {
		if index, parseErr := strconv.Atoi(strings.TrimPrefix(device, "cuda:")); parseErr != nil || index < 0 {
			toastErr(sse, "qualification CUDA device index invalid")
			return
		}
	}
	megaMode := strings.TrimSpace(signals.MegaMode)
	if megaMode == "" {
		megaMode = "max-autotune-no-cudagraphs"
	}
	if megaMode != "default" && megaMode != "max-autotune-no-cudagraphs" {
		toastErr(sse, "megakernel tuning mode invalid")
		return
	}
	for _, token := range strings.Split(strings.TrimSpace(signals.Prompt), ",") {
		if _, parseErr := strconv.Atoi(strings.TrimSpace(token)); parseErr != nil {
			toastErr(sse, "qualification prompt IDs must be comma-separated integers")
			return
		}
	}
	repeats, err := boundedInt(signals.Repeats, 5, 1, 20)
	if err != nil {
		toastErr(sse, "qualification repeats invalid")
		return
	}
	maxNew, err := boundedInt(signals.MaxNew, 32, 1, 2048)
	if err != nil {
		toastErr(sse, "qualification max-new invalid")
		return
	}
	output, err := s.pathUnderRepo(strings.TrimSpace(signals.Output), false)
	if err != nil || output == "" || !strings.HasSuffix(strings.ToLower(output), ".json") {
		toastErr(sse, "qualification output path invalid")
		return
	}
	rel, _ := filepath.Rel(s.cfg.RunsDir, output)
	if rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		toastErr(sse, "qualification output must be under runs/")
		return
	}
	if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
		toastErr(sse, "cannot create qualification output directory")
		return
	}
	args := []string{"-m", "rwkv_lab.production_kernels", "--device", device,
		"--repeats", repeats, "--prompt-ids", strings.TrimSpace(signals.Prompt),
		"--max-new", maxNew, "--megakernel-compile-mode", megaMode, "--output", output}
	if strings.TrimSpace(signals.Checkpoint) != "" {
		checkpoint, pathErr := s.pathUnderRepo(signals.Checkpoint, true)
		if pathErr != nil {
			toastErr(sse, "qualification checkpoint invalid")
			return
		}
		args = append(args, "--checkpoint", checkpoint)
	}
	if strings.TrimSpace(signals.Baseline) != "" {
		baseline, pathErr := s.pathUnderRepo(signals.Baseline, true)
		if pathErr != nil || !strings.HasSuffix(strings.ToLower(baseline), ".json") {
			toastErr(sse, "qualification baseline invalid")
			return
		}
		args = append(args, "--baseline", baseline)
	}
	for _, limit := range []struct{ flag, value string }{
		{"--max-throughput-regression", signals.Throughput},
		{"--max-memory-regression", signals.Memory},
		{"--max-kernel-regression", signals.Kernels},
	} {
		parsed, parseErr := strconv.ParseFloat(strings.TrimSpace(limit.value), 64)
		if parseErr != nil || parsed < 0 || parsed > 1 {
			toastErr(sse, "qualification regression limits must be in [0,1]")
			return
		}
		args = append(args, limit.flag, strconv.FormatFloat(parsed, 'f', -1, 64))
	}
	pid, err := s.spawnPy(args, fmt.Sprintf("qualification_%d.log", time.Now().Unix()))
	if err != nil {
		toastErr(sse, "qualification launch failed: "+err.Error())
		return
	}
	toastOK(sse, fmt.Sprintf("launched production qualification (pid %d)", pid))
}
