package server

import (
	"context"
	"log"
	"path/filepath"
	"time"

	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

// tickSnap is the shared per-second payload every connected stream renders.
// It is computed ONCE per second by refreshLoop and read by all connections:
// without this, each open tab re-ran the full summary/alert/convboard query
// suite every second through the single SQLite connection, and a handful of
// tabs (or an SSE reconnect burst) outran the drain rate and wedged the pool.
type tickSnap struct {
	ts        float64
	summaries []db.RunSummary
	procByRun map[string]sysmon.Proc
	bestByRun map[string]BestInfo

	sysGPUs, sysHost, sysProc string
	runList, alerts, conv     string
	queue, launchHist         string
	versions                  map[string]int64
}

// refreshLoop recomputes the shared snapshot once per second until ctx ends.
func (s *Server) refreshLoop(ctx context.Context) {
	s.refreshTick() // synchronous first snapshot so early connections have data
	t := time.NewTicker(streamInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.refreshTick()
		}
	}
}

func (s *Server) refreshTick() {
	now := float64(time.Now().UnixNano()) / 1e9
	snap := s.sampler.Latest()

	summaries, err := s.db.RunSummaries(now)
	if err != nil {
		log.Printf("[tick] run summaries: %v", err)
		return // keep serving the previous snapshot
	}
	procByRun := procIndex(snap.Procs)
	bestByRun := make(map[string]BestInfo, len(summaries))
	for index := range summaries {
		best := readBest(filepath.Join(s.cfg.RunsDir, summaries[index].Name))
		bestByRun[summaries[index].Name] = best
		applyEvalContractSummary(&summaries[index], best)
	}

	ts := &tickSnap{
		ts:         now,
		summaries:  summaries,
		procByRun:  procByRun,
		bestByRun:  bestByRun,
		sysGPUs:    renderSysGPUs(snap.GPUs),
		sysHost:    renderSysHost(snap.Host),
		sysProc:    renderSysProc(snap.Procs),
		runList:    renderRunList(summaries, procByRun, now),
		conv:       renderConvBoard(s.scanBoard(summaries, snap.Procs)),
		launchHist: renderLaunchHistory(s.db.RecentLaunchArgs(12)),
		versions:   make(map[string]int64, len(summaries)),
	}
	if active, err := s.db.ActiveAlerts(20); err == nil {
		ts.alerts = renderAlerts(active, s.autoStopOn())
	}
	if q, err := s.db.ActiveQueue(); err == nil {
		ts.queue = renderQueue(q, s.queueAuto.Load(), len(snap.Procs) == 0)
	}
	for _, su := range summaries {
		// A step number is not a data revision: eval/checkpoint records are often
		// appended after the train row at the same step. File mtime in milliseconds
		// is monotonic across appends and remains JS-number safe, including rewrites.
		ts.versions[su.Name] = runVersion(su)
	}
	s.tick.Store(ts)
}

// latestTick returns the current shared snapshot (nil before the first compute).
func (s *Server) latestTick() *tickSnap { return s.tick.Load() }
