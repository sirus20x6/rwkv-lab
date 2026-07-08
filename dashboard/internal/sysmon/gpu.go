package sysmon

import (
	"bufio"
	"bytes"
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// nvidia-smi query columns, in order. csv,noheader,nounits gives bare numbers.
const nvQuery = "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,enforced.power.limit"

// readGPUs shells out to nvidia-smi. Returns nil if nvidia-smi is missing or
// errors (no GPU / driver issue) — the caller renders "no GPU" gracefully.
func readGPUs() []GPU {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu="+nvQuery, "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var gpus []GPU
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		cols := strings.Split(line, ",")
		for i := range cols {
			cols[i] = strings.TrimSpace(cols[i])
		}
		if len(cols) < 9 {
			continue
		}
		memUsed := mb2gb(pf(cols[4]))
		memTotal := mb2gb(pf(cols[5]))
		memPct := 0.0
		if memTotal > 0 {
			memPct = 100 * memUsed / memTotal
		}
		g := GPU{
			Index:      pi(cols[0]),
			Name:       cols[1],
			UtilPct:    pi(cols[2]),
			MemUtilPct: pi(cols[3]),
			MemUsedGB:  memUsed,
			MemTotalGB: memTotal,
			MemPct:     memPct,
			TempC:      pi(cols[6]),
		}
		if v, ok := pfOpt(cols[7]); ok {
			g.PowerW = &v
		}
		if v, ok := pfOpt(cols[8]); ok {
			g.PowerCapW = &v
		}
		gpus = append(gpus, g)
	}
	return gpus
}

func mb2gb(mb float64) float64 { return mb / 1024.0 }

func pf(s string) float64 { f, _ := strconv.ParseFloat(strings.TrimSpace(s), 64); return f }
func pi(s string) int     { return int(pf(s)) }

// pfOpt parses a float, returning ok=false for nvidia-smi's "[N/A]"/"[Not Supported]".
func pfOpt(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" || strings.HasPrefix(s, "[") {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}
