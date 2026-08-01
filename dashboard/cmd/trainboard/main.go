// Command trainboard is the moe-mla training dashboard v2.0 server.
//
// Stack: Go + SQLite + Datastar + Pixi.js. It ingests runs/*/train.jsonl and
// system telemetry into SQLite, streams live scalars over a Datastar SSE
// stream, and serves GPU-accelerated Pixi charts. GPU-light — safe to run
// alongside live training.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"trainboard/internal/alerts"
	"trainboard/internal/db"
	"trainboard/internal/ingest"
	"trainboard/internal/server"
	"trainboard/internal/sysmon"
	trainvmstore "trainboard/internal/trainvm"
	"trainboard/web"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:9124", "listen address (localhost only by default)")
	repo := flag.String("repo", "/thearray/git/moe-mla", "moe-mla repo root")
	runs := flag.String("runs", "", "runs dir (default <repo>/runs)")
	dbPath := flag.String("db", "", "sqlite path (default <repo>/dashboard/trainboard.db)")
	trainVMPath := flag.String("trainvm-db", "", "legacy read-only TrainVM journal path (explicit compatibility fallback only)")
	trainVMBinary := flag.String("trainvm-bin", "", "TrainVM compiler binary (default <repo>/trainvm/build/trainvm)")
	trainVMSchema := flag.String("trainvm-schema", "", "TrainVM experiment schema (default <repo>/docs/experiment-vm/experiment-v1.schema.json)")
	trainVMSocket := flag.String("trainvm-socket", "", "TrainVM authority Unix socket (default <repo>/trainvm.sock)")
	imageRoots := flag.String("image-roots", "/thearray",
		"comma-separated roots eval-sample images may be served from")
	scanOnce := flag.Bool("scan-once", false, "ingest one full pass, print counts, exit (verification)")
	sysmonOnce := flag.Bool("sysmon-once", false, "print one telemetry snapshot as JSON, exit (verification)")
	flag.Parse()

	runsDir := *runs
	if runsDir == "" {
		runsDir = filepath.Join(*repo, "runs")
	}
	dbFile := *dbPath
	if dbFile == "" {
		dbFile = filepath.Join(*repo, "dashboard", "trainboard.db")
	}

	database, err := db.Open(dbFile)
	if err != nil {
		log.Fatalf("db open: %v", err)
	}
	defer database.Close()

	vmBinary := *trainVMBinary
	if vmBinary == "" {
		vmBinary = filepath.Join(*repo, "trainvm", "build", "trainvm")
	}
	vmSchema := *trainVMSchema
	if vmSchema == "" {
		vmSchema = filepath.Join(*repo, "docs", "experiment-vm", "experiment-v1.schema.json")
	}
	vmAuthoring := &trainvmstore.Authoring{
		BinaryPath:  vmBinary,
		SchemaPath:  vmSchema,
		ExamplePath: filepath.Join(*repo, "docs", "experiment-vm", "examples", "mageflow-cache-resume.json"),
	}
	vmSocket := *trainVMSocket
	if vmSocket == "" {
		vmSocket = filepath.Join(*repo, "trainvm.sock")
	}
	var vmCommander *trainvmstore.GRPCCommander
	if vmSocket != "" {
		vmCommander, err = trainvmstore.DialCommander(vmSocket)
		if err != nil {
			log.Fatalf("TrainVM authority client: %v", err)
		}
		defer vmCommander.Close()
		log.Printf("TrainVM command authority configured for %s", vmSocket)
	}
	var vmReader trainvmstore.ReadModel = vmCommander
	if *trainVMPath != "" {
		legacyReader, openErr := trainvmstore.Open(*trainVMPath)
		if openErr != nil {
			log.Fatalf("legacy TrainVM journal open: %v", openErr)
		}
		defer legacyReader.Close()
		vmReader = legacyReader
		log.Printf("legacy TrainVM read model attached explicitly to %s", *trainVMPath)
	} else {
		log.Printf("TrainVM read model uses the native authority at %s", vmSocket)
	}

	ig := ingest.New(database, runsDir, time.Second)

	if *scanOnce {
		n, err := ig.ScanOnce()
		if err != nil {
			log.Fatalf("scan: %v", err)
		}
		fmt.Printf("scan-once ingested %d events into %s\n", n, dbFile)
		return
	}

	if *sysmonOnce {
		sm := sysmon.New(nil, runsDir, "/thearray", 0)
		_ = sm.Probe()                     // prime CPU baseline
		time.Sleep(250 * time.Millisecond) // let a CPU delta accumulate
		snap := sm.Probe()
		out, _ := json.MarshalIndent(snap, "", "  ")
		fmt.Println(string(out))
		return
	}

	// Cancel on Ctrl-C / SIGTERM so the HTTP server + goroutines drain cleanly.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	sampler := sysmon.New(database, runsDir, "/thearray", 0)
	detector := alerts.New(database, sampler, runsDir, 0)

	go ig.Run(ctx)
	go sampler.Run(ctx)
	go detector.Run(ctx)

	srv := server.New(server.Config{
		Addr:      *addr,
		RunsDir:   runsDir,
		RepoRoot:  *repo,
		Static:    web.Static(),
		DB:        database,
		Sampler:   sampler,
		Detector:  detector,
		TrainVM:   vmReader,
		Authoring: vmAuthoring,
		Commander: vmCommander,
		ImageRoots: func() []string {
			var roots []string
			for _, root := range strings.Split(*imageRoots, ",") {
				if root = strings.TrimSpace(root); root != "" {
					roots = append(roots, root)
				}
			}
			return roots
		}(),
	})

	if err := srv.Run(ctx); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
