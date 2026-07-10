package server

// Sample panel: generate text from a selected run's saved checkpoint via
// `python -m rwkv_lab.generate --json`. Synchronous within the SSE request
// (lab models sample in seconds); the checkpoint is self-describing
// (blob["arch"]), so any lever combination — seed-chain, DeepEmbed, Engram —
// rebuilds and samples without extra flags.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

func (s *Server) handleSample(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var sig struct {
		SamplePrompt string `json:"samplePrompt"`
		SampleMaxNew string `json:"sampleMaxNew"`
		SampleTemp   string `json:"sampleTemp"`
	}
	_ = datastar.ReadSignals(r, &sig)
	sse := datastar.NewSSE(w, r)
	ckpt := filepath.Join(s.cfg.RunsDir, name, "ckpt.pt")
	if _, err := os.Stat(ckpt); err != nil {
		toastErr(sse, "no ckpt.pt in this run — LM runs save one at completion")
		return
	}
	prompt := strings.TrimSpace(sig.SamplePrompt)
	if prompt == "" {
		prompt = "Write a Python function that reverses a string."
	}
	maxNew := "200"
	if n, err := strconv.Atoi(strings.TrimSpace(sig.SampleMaxNew)); err == nil && n > 0 && n <= 2048 {
		maxNew = strconv.Itoa(n)
	}
	temp := "0.8"
	if t, err := strconv.ParseFloat(strings.TrimSpace(sig.SampleTemp), 64); err == nil && t >= 0 && t <= 4 {
		temp = strconv.FormatFloat(t, 'f', -1, 64)
	}
	_ = sse.MarshalAndPatchSignals(map[string]any{"sampleBusy": true})
	_ = sse.PatchElements(`<pre id="sample-out" class="sample-out">sampling…</pre>`)
	start := time.Now()
	cmd := exec.Command(filepath.Join(s.cfg.RepoRoot, ".venv", "bin", "python"),
		"-m", "rwkv_lab.generate", "--ckpt", ckpt, "--prompt", prompt,
		"--max-new", maxNew, "--temperature", temp, "--json")
	cmd.Dir = s.cfg.RepoRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH=src")
	cmd.WaitDelay = 3 * time.Minute
	out, err := cmd.Output()
	_ = sse.MarshalAndPatchSignals(map[string]any{"sampleBusy": false})
	if err != nil {
		msg := err.Error()
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			lines := strings.Split(strings.TrimSpace(string(ee.Stderr)), "\n")
			msg = lines[len(lines)-1]
		}
		toastErr(sse, "sample failed: "+msg)
		return
	}
	var res struct {
		Config     string `json:"config"`
		Step       int    `json:"step"`
		Prompt     string `json:"prompt"`
		Completion string `json:"completion"`
		Tokens     int    `json:"tokens"`
	}
	// generate.py may print harness noise before the JSON line — take the last line.
	linesOut := strings.Split(strings.TrimSpace(string(out)), "\n")
	if json.Unmarshal([]byte(linesOut[len(linesOut)-1]), &res) != nil {
		toastErr(sse, "sample: unparseable generator output")
		return
	}
	head := fmt.Sprintf("[%s @ step %d · %d tokens · %.1fs]\n\n",
		res.Config, res.Step, res.Tokens, time.Since(start).Seconds())
	_ = sse.PatchElements(`<pre id="sample-out" class="sample-out">` +
		esc(head+res.Prompt) + `<b>` + esc(res.Completion) + `</b></pre>`)
	s.db.LogAction(nowTs(), "sample", name, "{}", "ok", 0)
	toastOK(sse, fmt.Sprintf("sampled %d tokens from %s", res.Tokens, name))
}
