// Package sysmon samples host + GPU telemetry and detects training processes.
// It keeps the latest snapshot in memory (for the SSE stream) and persists a
// downsampled row per tick to system_samples (for history sparklines). GPU
// stats come from shelling out to nvidia-smi (no cgo); host stats from gopsutil.
package sysmon

import (
	"context"
	"encoding/json"
	"log"
	"sync"
	"time"

	"trainboard/internal/db"
)

// GPU is one device's telemetry (fields mirror v1's pynvml set).
type GPU struct {
	Index      int      `json:"index"`
	Name       string   `json:"name"`
	UtilPct    int      `json:"util_pct"`
	MemUtilPct int      `json:"mem_util_pct"`
	MemUsedGB  float64  `json:"mem_used_gb"`
	MemTotalGB float64  `json:"mem_total_gb"`
	MemPct     float64  `json:"mem_pct"`
	TempC      int      `json:"temp_c"`
	PowerW     *float64 `json:"power_w"`
	PowerCapW  *float64 `json:"power_cap_w"`
}

// Host is CPU/RAM/disk/load.
type Host struct {
	CPUPct      float64   `json:"cpu_pct"`
	CPUPerCore  []float64 `json:"cpu_per_core"`
	CPUCount    int       `json:"cpu_count"`
	RAMUsedGB   float64   `json:"ram_used_gb"`
	RAMTotalGB  float64   `json:"ram_total_gb"`
	RAMPct      float64   `json:"ram_pct"`
	DiskUsedGB  float64   `json:"disk_used_gb"`
	DiskFreeGB  float64   `json:"disk_free_gb"`
	DiskTotalGB float64   `json:"disk_total_gb"`
	DiskPct     float64   `json:"disk_pct"`
	Load1       float64   `json:"load1"`
	Load5       float64   `json:"load5"`
	Load15      float64   `json:"load15"`
}

// Proc is a detected training process.
type Proc struct {
	PID        int32    `json:"pid"`
	Script     string   `json:"script"`
	RunName    string   `json:"run_name"`
	RuntimeS   float64  `json:"runtime_s"`
	CPUPct     float64  `json:"cpu_pct"`
	RSSGB      float64  `json:"rss_gb"`
	NumThreads int32    `json:"num_threads"`
	LogAgeS    *float64 `json:"log_age_s"`
	State      string   `json:"state"`
	MaxSteps   *int     `json:"max_steps"`
}

// Snapshot is one full sample.
type Snapshot struct {
	GPUs  []GPU   `json:"gpus"`
	Host  Host    `json:"host"`
	Procs []Proc  `json:"procs"`
	TS    float64 `json:"ts"`
}

// Liveness windows (seconds) — shared with the run-summary status logic. Same
// thresholds as v1 (dashboard/app.py): log fresh <5min = healthy, <15min = stalling.
const (
	HealthyWindow = 300.0
	StaleWindow   = 900.0
)

// Sampler periodically refreshes the snapshot and persists history.
type Sampler struct {
	runsDir  string
	diskPath string
	db       *db.DB
	interval time.Duration

	mu     sync.RWMutex
	latest Snapshot
}

// New builds a sampler. interval<=0 defaults to 1.5s.
func New(database *db.DB, runsDir, diskPath string, interval time.Duration) *Sampler {
	if interval <= 0 {
		interval = 1500 * time.Millisecond
	}
	return &Sampler{runsDir: runsDir, diskPath: diskPath, db: database, interval: interval}
}

// Run primes the CPU counter, then samples until ctx is cancelled.
func (s *Sampler) Run(ctx context.Context) {
	primeCPU() // first cpu.Percent call must establish a baseline
	s.sampleOnce()
	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.sampleOnce()
		}
	}
}

func (s *Sampler) sampleOnce() {
	snap := Snapshot{
		GPUs:  readGPUs(),
		Host:  readHost(s.diskPath),
		Procs: readProcs(s.runsDir),
		TS:    float64(time.Now().UnixNano()) / 1e9,
	}
	s.mu.Lock()
	s.latest = snap
	s.mu.Unlock()
	s.persist(snap)
}

func (s *Sampler) persist(snap Snapshot) {
	if s.db == nil {
		return
	}
	gpuJSON, _ := json.Marshal(snap.GPUs)
	load := snap.Host.Load1
	if _, err := s.db.Exec(
		`INSERT OR REPLACE INTO system_samples(ts,gpu_json,cpu_pct,ram_pct,disk_pct,loadavg) VALUES(?,?,?,?,?,?)`,
		snap.TS, string(gpuJSON), snap.Host.CPUPct, snap.Host.RAMPct, snap.Host.DiskPct, load); err != nil {
		log.Printf("[sysmon] persist: %v", err)
	}
}

// Latest returns a copy of the most recent snapshot.
func (s *Sampler) Latest() Snapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.latest
}

// Probe builds a fresh snapshot without persisting — used by the -sysmon-once
// verification path. Call twice (a beat apart) for a meaningful CPU%.
func (s *Sampler) Probe() Snapshot {
	return Snapshot{
		GPUs:  readGPUs(),
		Host:  readHost(s.diskPath),
		Procs: readProcs(s.runsDir),
		TS:    float64(time.Now().UnixNano()) / 1e9,
	}
}
