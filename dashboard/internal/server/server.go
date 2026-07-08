// Package server wires the HTTP surface for trainboard: the static front-end
// shell, the Datastar SSE stream, the Pixi series endpoints, and the
// confirm-gated control actions. This file holds the router + lifecycle; the
// per-area handlers live in handlers.go, stream.go, and control.go.
package server

import (
	"context"
	"io/fs"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"trainboard/internal/alerts"
	"trainboard/internal/db"
	"trainboard/internal/sysmon"
)

// Config is the immutable wiring for a Server.
type Config struct {
	Addr     string           // e.g. "127.0.0.1:9124"
	RunsDir  string           // /thearray/git/moe-mla/runs
	RepoRoot string           // /thearray/git/moe-mla
	Static   fs.FS            // front-end assets (web.Static())
	DB       *db.DB           // datastore
	Sampler  *sysmon.Sampler  // live telemetry
	Detector *alerts.Detector // divergence/health detector
	LibDir   string           // converted_layers_lib path (conversion board)
	NLayers  int              // base-model layer count (0 = autodetect/default)
}

// Server owns the HTTP handler, the datastore, the live telemetry sampler, and
// the (single-user, localhost) selected-run state.
type Server struct {
	cfg      Config
	mux      *http.ServeMux
	db       *db.DB
	sampler  *sysmon.Sampler
	detector *alerts.Detector

	mu sync.RWMutex
	// Per-viewer selection: each browser tab sends a stable tabId signal, so one
	// viewer choosing a run never flips another's chart/header/KPIs. Empty tabId
	// (signal-less client) shares the "_" bucket. seen drives idle GC.
	selected map[string]string
	seen     map[string]time.Time

	queueAuto atomic.Bool // opt-in: auto-start next queued run when GPU free (off by default)

	// tick is the shared per-second snapshot every stream connection renders.
	// Computed once by refreshLoop (tickcache.go) so N tabs cost one query suite.
	tick atomic.Pointer[tickSnap]
}

// tabTTL bounds the per-tab selection map: a tab idle (no stream tick, no select)
// this long is forgotten. Active stream connections touch their tab every second.
const tabTTL = 10 * time.Minute

// New builds the router and wires the DB + sampler + detector.
func New(cfg Config) *Server {
	s := &Server{
		cfg: cfg, mux: http.NewServeMux(), db: cfg.DB, sampler: cfg.Sampler, detector: cfg.Detector,
		selected: map[string]string{},
		seen:     map[string]time.Time{},
	}
	s.routes()
	return s
}

func tabKey(tabID string) string {
	if tabID == "" {
		return "_"
	}
	return tabID
}

// setSelected records this tab's chosen run (called from /api/run/{name}).
func (s *Server) setSelected(tabID, name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := tabKey(tabID)
	s.selected[k] = name
	s.seen[k] = time.Now()
	s.gcTabsLocked()
}

// selectedFor returns the run this tab is viewing, refreshing its liveness.
func (s *Server) selectedFor(tabID string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := tabKey(tabID)
	s.seen[k] = time.Now()
	s.gcTabsLocked()
	return s.selected[k]
}

// gcTabsLocked drops tabs idle past tabTTL (caller holds s.mu).
func (s *Server) gcTabsLocked() {
	cutoff := time.Now().Add(-tabTTL)
	for id, t := range s.seen {
		if t.Before(cutoff) {
			delete(s.seen, id)
			delete(s.selected, id)
		}
	}
}

func (s *Server) routes() {
	// Static assets under /static/. http.FileServerFS serves the embedded FS.
	fileServer := http.FileServerFS(s.cfg.Static)
	s.mux.Handle("GET /static/", http.StripPrefix("/static/", fileServer))

	// App shell.
	s.mux.HandleFunc("GET /{$}", s.handleIndex)

	// Liveness probe (handy for `curl` smoke tests and external monitors).
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("ok"))
	})

	// Datastar SSE stream (system header + run list + selected-run header/KPIs).
	s.mux.HandleFunc("GET /api/stream", s.handleStream)
	// Select a run for the detail panel (one-shot SSE).
	s.mux.HandleFunc("GET /api/run/{name}", s.handleRunSelect)
	// Columnar metric series for the Pixi charts.
	s.mux.HandleFunc("GET /api/series/{run}", s.handleSeries)
	// Timeline markers (checkpoints/alerts/controls/actions) for chart overlays.
	s.mux.HandleFunc("GET /api/timeline/{run}", s.handleTimeline)
	// Metric catalog (known cols + extra_json keys) for the dynamic metric picker.
	s.mux.HandleFunc("GET /api/metrics/{run}", s.handleMetrics)

	// Control actions (confirm-gated client-side, validated + audited here).
	s.mux.HandleFunc("POST /api/runs/{name}/stop", s.handleStop)
	s.mux.HandleFunc("POST /api/runs/{name}/checkpoint", s.handleCheckpoint)
	s.mux.HandleFunc("POST /api/runs/{name}/notes", s.handleNotes)
	s.mux.HandleFunc("POST /api/runs/{name}/tags", s.handleTags)
	s.mux.HandleFunc("POST /api/runs/{name}/control", s.handleSetControl)
	s.mux.HandleFunc("GET /api/runs/{name}/architecture", s.handleArchitecture)
	s.mux.HandleFunc("POST /api/launch", s.handleLaunch)
	s.mux.HandleFunc("POST /api/alerts/ack", s.handleAckAlert)
	s.mux.HandleFunc("POST /api/autostop", s.handleAutoStop)
	s.mux.HandleFunc("POST /api/convboard/accept", s.handleAcceptLayer)
	s.mux.HandleFunc("GET /api/leaderboard", s.handleLeaderboard)
	s.mux.HandleFunc("GET /api/diff", s.handleDiff)
	s.mux.HandleFunc("POST /api/queue/enqueue", s.handleEnqueue)
	s.mux.HandleFunc("POST /api/queue/start-next", s.handleStartNext)
	s.mux.HandleFunc("POST /api/queue/cancel", s.handleCancelQueue)
	s.mux.HandleFunc("POST /api/queue/auto", s.handleQueueAuto)

	// NOTE: /api/runs/{run}/architecture and /api/system/history are phase 4b.
}

// handleIndex serves the Datastar shell (static index.html for now). We read
// the whole file (it's small) rather than ServeContent — embed.FS and os.DirFS
// files don't both guarantee io.Seeker, and index.html doesn't need ranges.
func (s *Server) handleIndex(w http.ResponseWriter, _ *http.Request) {
	data, err := fs.ReadFile(s.cfg.Static, "index.html")
	if err != nil {
		http.Error(w, "index.html missing", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(data)
}

// Handler exposes the router (useful for tests / middleware wrapping).
func (s *Server) Handler() http.Handler { return s.mux }

// Run starts the HTTP server and blocks until ctx is cancelled, then drains.
func (s *Server) Run(ctx context.Context) error {
	srv := &http.Server{
		Addr:              announcedAddr(s.cfg.Addr),
		Handler:           s.mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go s.queueManager(ctx) // reconcile running items + opt-in auto-start
	go s.refreshLoop(ctx)  // shared once-per-second tick snapshot for all streams
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	log.Printf("trainboard listening on http://%s", s.cfg.Addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

func announcedAddr(a string) string {
	if a == "" {
		return "127.0.0.1:9124"
	}
	return a
}
