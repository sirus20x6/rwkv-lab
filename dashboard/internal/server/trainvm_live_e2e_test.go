package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	trainvmstore "trainboard/internal/trainvm"
)

const liveE2EAdapterRegistry = `{
  "api_version": "trainvm.adapters/v2",
  "profiles": [
    {
      "key": {"adapter":"rwkv-lab.mageflow","version":"1.0.0","runtime":"python_worker","operation":"train","contract":"rwkv_lab.mageflow.v1.Train"},
      "effect": "process",
      "idempotency": "receipt_required",
      "code_fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "required_capabilities": ["worker.controls", "worker.metrics"],
      "lifecycle": {"stateful":true,"graceful_stop":true,"checkpoint_now":true,"pause_keep_resources":true,"pause_release_resources":true,"compile":true,"warmup":true,"qualify":true,"profile":true,"resume_grade":"exact"}
    },
    {
      "key": {"adapter":"rwkv-lab.mageflow","version":"1.0.0","runtime":"python_worker","operation":"prepare_cache_span","contract":"rwkv_lab.mageflow.v1.PrepareCacheSpan"},
      "effect": "workspace_write",
      "idempotency": "replay_safe",
      "code_fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "required_capabilities": ["worker.controls", "worker.metrics"],
      "lifecycle": {"stateful":false,"graceful_stop":false,"checkpoint_now":false,"pause_keep_resources":false,"pause_release_resources":false,"compile":false,"warmup":false,"qualify":false,"profile":false,"resume_grade":"none"}
    },
    {
      "key": {"adapter":"rwkv-lab.mageflow","version":"1.0.0","runtime":"python_worker","operation":"cache_encoders","contract":"rwkv_lab.mageflow.v1.CacheEncoders"},
      "effect": "process",
      "idempotency": "receipt_required",
      "code_fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "required_capabilities": ["worker.controls", "worker.metrics"],
      "lifecycle": {"stateful":false,"graceful_stop":true,"checkpoint_now":false,"pause_keep_resources":false,"pause_release_resources":false,"compile":false,"warmup":false,"qualify":false,"profile":false,"resume_grade":"none"}
    }
  ]
}`

const liveE2EHostLaunchRegistry = `{
  "api_version": "trainvm.host-launches/v3",
  "trusted_roots": [],
  "profiles": []
}`

func writeLiveE2EDocument(t *testing.T, path, document string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
}

func liveE2ERequest(t *testing.T, handler http.Handler, method, target string, body any, expected int) map[string]any {
	t.Helper()
	var encoded []byte
	if body != nil {
		var err error
		encoded, err = json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
	}
	request := httptest.NewRequest(method, target, bytes.NewReader(encoded))
	request.Host = "127.0.0.1:9124"
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != expected {
		t.Fatalf("%s %s: status=%d expected=%d body=%s", method, target, response.Code, expected, response.Body.String())
	}
	if response.Body.Len() == 0 {
		return nil
	}
	var result map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("%s %s returned invalid JSON: %v: %s", method, target, err, response.Body.String())
	}
	return result
}

func liveE2EIdentity(t *testing.T, value any) map[string]any {
	t.Helper()
	identity, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("expected identity object, got %#v", value)
	}
	return identity
}

func TestTrainVMLiveStackE2E(t *testing.T) {
	binary := strings.TrimSpace(os.Getenv("TRAINVM_E2E_BINARY"))
	if binary == "" {
		t.Skip("set TRAINVM_E2E_BINARY to the native trainvm executable")
	}
	binary, err := filepath.Abs(binary)
	if err != nil {
		t.Fatal(err)
	}
	repository, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	temporary := t.TempDir()
	journal := filepath.Join(temporary, "journal.db")
	socket := filepath.Join(temporary, "trainvm.sock")
	adapters := filepath.Join(temporary, "adapters.json")
	hostLaunches := filepath.Join(temporary, "host-launches.json")
	writeLiveE2EDocument(t, adapters, liveE2EAdapterRegistry)
	writeLiveE2EDocument(t, hostLaunches, liveE2EHostLaunchRegistry)

	command := exec.Command(binary, "serve",
		"--journal", journal,
		"--socket", socket,
		"--registry", adapters,
		"--host-launch-registry", hostLaunches,
		"--training-component-registry", filepath.Join(repository, "docs", "experiment-vm", "examples", "empty-training-components.json"))
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	waitResult := make(chan error, 1)
	go func() { waitResult <- command.Wait() }()
	stopped := false
	stopAuthority := func() error {
		if stopped {
			return nil
		}
		stopped = true
		if command.Process != nil {
			_ = command.Process.Signal(syscall.SIGTERM)
		}
		select {
		case waitErr := <-waitResult:
			return waitErr
		case <-time.After(5 * time.Second):
			if command.Process != nil {
				_ = command.Process.Kill()
			}
			return <-waitResult
		}
	}
	t.Cleanup(func() { _ = stopAuthority() })

	commander, err := trainvmstore.DialCommander(socket)
	if err != nil {
		t.Fatal(err)
	}
	defer commander.Close()
	deadline := time.Now().Add(10 * time.Second)
	reachable := func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		return commander.Reachable(ctx)
	}
	for !reachable() && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if !reachable() {
		stopErr := stopAuthority()
		t.Fatalf("native authority did not become reachable: wait=%v stdout=%s stderr=%s", stopErr, stdout.String(), stderr.String())
	}

	authoring := &trainvmstore.Authoring{
		BinaryPath:  binary,
		SchemaPath:  filepath.Join(repository, "docs", "experiment-vm", "experiment-v1.schema.json"),
		ExamplePath: filepath.Join(repository, "docs", "experiment-vm", "examples", "mageflow-cache-resume.json"),
	}
	handler := New(Config{
		RepoRoot: repository, RunsDir: filepath.Join(repository, "runs"),
		TrainVM: commander, Commander: commander, Authoring: authoring,
	}).Handler()

	runs := liveE2ERequest(t, handler, http.MethodGet, "/api/trainvm/runs", nil, http.StatusOK)
	journalID, _ := runs["journal_id"].(string)
	if journalID == "" || runs["commands_enabled"] != true {
		t.Fatalf("dashboard did not bind the native read/command authority: %#v", runs)
	}
	source, err := os.ReadFile(authoring.ExamplePath)
	if err != nil {
		t.Fatal(err)
	}
	preview := liveE2ERequest(t, handler, http.MethodPost, "/api/trainvm/compile", json.RawMessage(source), http.StatusOK)
	planHash, _ := preview["plan_hash"].(string)
	adapterLock, _ := preview["adapter_lock_digest"].(string)
	if preview["valid"] != true || planHash == "" || adapterLock == "" {
		t.Fatalf("native compiler/authority preview was not valid: %#v", preview)
	}
	created := liveE2ERequest(t, handler, http.MethodPost, "/api/trainvm/experiments", map[string]any{
		"source_document": string(source), "source_format": "json", "create_run": true,
		"idempotency_key": "live-e2e-root", "expected_journal_id": journalID,
		"expected_plan_hash": planHash, "expected_adapter_lock_digest": adapterLock,
		"expected_training_component_lock_digest": "", "reason": "live boundary acceptance",
	}, http.StatusAccepted)
	root := liveE2EIdentity(t, created["run"])
	rootID, _ := root["run_id"].(string)
	if rootID == "" || root["plan_hash"] != planHash {
		t.Fatalf("native authority returned an invalid root identity: %#v", created)
	}
	plan := liveE2ERequest(t, handler, http.MethodGet, "/api/trainvm/runs/"+rootID+"/plan", nil, http.StatusOK)
	if plan["journal_id"] != journalID || plan["plan_hash"] != planHash {
		t.Fatalf("immutable plan readback lost identity: %#v", plan)
	}

	var proposed map[string]any
	if err := json.Unmarshal(source, &proposed); err != nil {
		t.Fatal(err)
	}
	metadata := proposed["metadata"].(map[string]any)
	metadata["description"] = "live boundary revision fork"
	proposedSource, err := json.Marshal(proposed)
	if err != nil {
		t.Fatal(err)
	}
	proposedPreview := liveE2ERequest(t, handler, http.MethodPost, "/api/trainvm/compile", json.RawMessage(proposedSource), http.StatusOK)
	proposedHash, _ := proposedPreview["plan_hash"].(string)
	proposedAdapterLock, _ := proposedPreview["adapter_lock_digest"].(string)
	if proposedHash == "" || proposedHash == planHash || proposedAdapterLock == "" {
		t.Fatalf("changed preview did not produce a distinct valid plan: %#v", proposedPreview)
	}
	rootView := liveE2ERequest(t, handler, http.MethodGet, "/api/trainvm/runs/"+rootID, nil, http.StatusOK)
	revision, ok := rootView["run_revision"].(float64)
	if !ok || revision < 1 {
		t.Fatalf("root run has no live revision: %#v", rootView)
	}
	diff := liveE2ERequest(t, handler, http.MethodPost, "/api/trainvm/runs/"+rootID+"/diff", map[string]any{
		"expected_run_revision": uint64(revision), "proposed_source_document": string(proposedSource),
		"source_format": "json", "expected_journal_id": journalID,
		"expected_current_plan_hash": planHash, "expected_proposed_plan_hash": proposedHash,
	}, http.StatusOK)
	semanticDiff, ok := diff["semantic_diff"].([]any)
	if !ok || len(semanticDiff) == 0 || diff["adoptable_in_place"] == true {
		t.Fatalf("live semantic diff did not require an immutable fork: %#v", diff)
	}
	forked := liveE2ERequest(t, handler, http.MethodPost, "/api/trainvm/experiments", map[string]any{
		"source_document": string(proposedSource), "source_format": "json", "create_run": true,
		"idempotency_key": "live-e2e-fork", "expected_journal_id": journalID,
		"expected_plan_hash": proposedHash, "expected_adapter_lock_digest": proposedAdapterLock,
		"expected_training_component_lock_digest": "", "reason": "live immutable fork acceptance",
		"forked_from_run_id": rootID, "expected_parent_run_revision": uint64(revision),
		"expected_parent_plan_hash": planHash,
	}, http.StatusAccepted)
	child := liveE2EIdentity(t, forked["run"])
	if child["run_id"] == rootID || child["plan_hash"] != proposedHash {
		t.Fatalf("fork did not create a distinct child identity: %#v", forked)
	}

	if err := stopAuthority(); err != nil {
		t.Fatalf("native authority shutdown failed: %v stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	verify := exec.Command(binary, "journal", "verify", journal)
	verification, err := verify.CombinedOutput()
	var verified struct {
		Events int  `json:"events"`
		Valid  bool `json:"valid"`
	}
	decodeErr := json.Unmarshal(verification, &verified)
	if err != nil || decodeErr != nil || !verified.Valid || verified.Events < 2 {
		t.Fatalf("live journal verification failed: %v: %s", err, verification)
	}
}
