package web

import (
	"io/fs"
	"os/exec"
	"strings"
	"testing"
)

func TestLiveDashboardClientHasRecoveryPaths(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(index), "retry:'always'") {
		t.Fatal("dashboard stream does not retry indefinitely")
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(app), "setInterval(fire, 1000)") {
		t.Fatal("active-run observer has no missed-mutation recovery")
	}
	if !strings.Contains(string(app), "/live`") ||
		!strings.Contains(string(app), "setInterval(refreshSelectedRun, 1000)") {
		t.Fatal("selected-run headers have no independent live refresh")
	}
	if !strings.Contains(string(index), `id="run-kpis"`) {
		t.Fatal("KPI strip has no stable server-patch target")
	}
}

func TestLegacyDashboardDoesNotExposeTrainingMutationControls(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	page := string(index)
	for _, forbidden := range []string{
		`/api/launch`, `/api/queue/`, `/api/autostop`,
		`/stop`, `/checkpoint`, `/control`, `/sample`,
		`id="tune-panel"`, `id="sample-panel"`,
	} {
		if strings.Contains(page, forbidden) {
			t.Fatalf("legacy dashboard still exposes training mutation control %q", forbidden)
		}
	}
	if !strings.Contains(page, `data-vm-action="checkpoint"`) ||
		!strings.Contains(page, `data-vm-action="pause"`) ||
		!strings.Contains(page, `data-vm-action="resume"`) {
		t.Fatal("native TrainVM lifecycle controls were removed with the legacy controls")
	}
}

func TestEvalGalleryDoesNotReloadOnStaleLiveTicks(t *testing.T) {
	assets := Static()
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	pixi, err := fs.ReadFile(assets, "pixi-glue.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(app), `signature !== evalRenderedSignature`) ||
		!strings.Contains(string(app), `image.getAttribute("src") !== item.image_url`) {
		t.Fatal("eval gallery does not preserve unchanged image/card DOM")
	}
	if strings.Contains(string(pixi), `version < lastVersion) loadRun`) {
		t.Fatal("an out-of-order live tick still forces a full gallery reload")
	}
	if !strings.Contains(string(pixi), `version > lastVersion) appendRun`) {
		t.Fatal("new live versions no longer append incrementally")
	}
	if !strings.Contains(string(app), `/eval-samples/latest?at_step=`) {
		t.Fatal("fresh gallery has no fallback for scalar-only eval markers")
	}
	if !strings.Contains(string(app), `const exactRetryLimit = 90`) {
		t.Fatal("eval gallery retry window is too short for cold vision reloads")
	}
}

func TestEvalGalleryRendersTargetAndPredictedBoxOverlays(t *testing.T) {
	assets := Static()
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	css, err := fs.ReadFile(assets, "app.css")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(app), `item.target_boxes`) ||
		!strings.Contains(string(app), `item.predicted_boxes`) ||
		!strings.Contains(string(app), `layoutEvalBoxOverlay`) {
		t.Fatal("eval gallery does not render both box streams over contained images")
	}
	if !strings.Contains(string(css), `.eval-box-target`) ||
		!strings.Contains(string(css), `.eval-box-predicted`) {
		t.Fatal("eval box streams do not have distinct visual styles")
	}
}

func TestEvalGalleryPairsEveryGeneratedImageWithItsOriginal(t *testing.T) {
	assets := Static()
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	css, err := fs.ReadFile(assets, "app.css")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(app), `items.some(item => Boolean(item.target_image_url))`) ||
		!strings.Contains(string(app), `const paired = Boolean(item.target_image_url)`) {
		t.Fatal("target images do not force generated/original paired rendering")
	}
	if !strings.Contains(string(css),
		`.eval-generation .eval-image-comparison { display: grid; grid-template-columns: repeat(2`) {
		t.Fatal("paired eval images are not laid out side by side")
	}
}

func TestEvalGalleryHasTimeScrubberAndPinnedHistoryMode(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	pixi, err := fs.ReadFile(assets, "pixi-glue.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(index), `id="eval-history-range"`) ||
		!strings.Contains(string(index), `id="eval-history-latest"`) {
		t.Fatal("eval gallery has no scrubber or latest control")
	}
	if !strings.Contains(string(app), `/eval-samples`) ||
		!strings.Contains(string(app), `evalHistoryManual`) ||
		!strings.Contains(string(app), `openEvalHistoryIndex`) {
		t.Fatal("eval gallery does not discover or navigate historical snapshots")
	}
	if !strings.Contains(string(pixi),
		`openEvalSamples(curRun, cand.step, cand.ppl, 0, "manual")`) {
		t.Fatal("clicking a historical eval marker does not pin history mode")
	}
}

func TestNativeEvalGalleryUsesPublishedHistoryAndSideBySideViewer(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	css, err := fs.ReadFile(assets, "app.css")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(index), `id="vm-gallery-range"`) ||
		!strings.Contains(string(index), `id="vm-gallery-latest"`) {
		t.Fatal("native gallery has no immutable-history scrubber")
	}
	if !strings.Contains(string(app), `/galleries/${encodeURIComponent(summary.artifact_id)}`) ||
		!strings.Contains(string(app), `item.target_image_url || item.source_image_url`) {
		t.Fatal("native gallery is not driven by published revisions or lacks original-image fallback")
	}
	if !strings.Contains(string(app), `vmObservability?.eval_gallery_artifact`) ||
		!strings.Contains(string(app), `artifact?.logical_name === logicalName`) ||
		!strings.Contains(string(app), `vmIsDeclaredGallery(artifact)`) {
		t.Fatal("native gallery discovery does not follow the compiled observability declaration")
	}
	if !strings.Contains(string(app), `history?.galleries`) ||
		!strings.Contains(string(app), `history?.history_truncated`) ||
		!strings.Contains(string(app), `older history truncated`) {
		t.Fatal("native gallery history does not expose bounded-tail truncation")
	}
	if !strings.Contains(string(app), `immutableRunIdentityChanged`) ||
		!strings.Contains(string(app), `resetVMGallery(selected.run_id)`) ||
		!strings.Contains(string(app), `resetVMControls(selected.run_id)`) ||
		!strings.Contains(string(app), `create a new run for a new plan`) {
		t.Fatal("the browser does not fail closed when an immutable run identity changes")
	}
	if !strings.Contains(string(app), `requestedAfter !== vmAfter`) ||
		!strings.Contains(string(app), `selectionGeneration !== vmSelectionGeneration`) {
		t.Fatal("timeline replay is not fenced against stale in-flight requests")
	}
	if !strings.Contains(string(css), `.vm-gallery-pair { display: grid; grid-template-columns: repeat(2`) {
		t.Fatal("native generated/original images are not rendered side by side")
	}
}

func TestTrainVMGPUProfilesQualifyDeclaredWarmupState(t *testing.T) {
	assets := Static()
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	css, err := fs.ReadFile(assets, "app.css")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		`profile.execution_phases?.overlaps_warmup === baseline.execution_phases?.overlaps_warmup`,
		`warmup overlap unknown`, `steady-state status cannot be determined`,
		`profile.trace_sha256, profile.execution_phases`, `.vm-profile-phase`,
	} {
		if !strings.Contains(string(app)+string(css), required) {
			t.Fatalf("GPU trace phase qualification is missing %q", required)
		}
	}
}

func TestTrainVMPanelUsesIncrementalReadOnlyTimeline(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(index), `id="trainvm-panel"`) ||
		!strings.Contains(string(index), `id="trainvm-timeline"`) {
		t.Fatal("native TrainVM runs and timeline have no dashboard panel")
	}
	if !strings.Contains(string(app), `/api/trainvm/runs`) ||
		!strings.Contains(string(app), `timeline?after=${requestedAfter}&limit=1000`) ||
		!strings.Contains(string(app), `setInterval(refreshTrainVM, 1000)`) {
		t.Fatal("TrainVM panel does not incrementally follow the native journal")
	}
	for _, required := range []string{
		`id="vm-control-catalog"`, `id="vm-control-apply"`,
		`id="trainvm-metrics"`, `id="trainvm-artifacts"`,
		`id="trainvm-observability-state"`,
		`id="vm-execution-phases"`, `renderVMExecutionPhases`,
		`/observability?after=${vmTelemetryAfter}&limit=250`,
		`vmMetricSeries`, `vmArtifacts`, `renderVMMetricCharts`, `renderVMArtifacts`,
		`data-vm-open-artifact`, `/artifacts/${encodeURIComponent(artifact.artifact_id)}/content?v=`,
		`&s=${encodeURIComponent(Number(artifact.sequence || 0))}`,
		`expected_control_revision`, `vmPendingControls`, `randomUUID`,
		`request atomic patch`, `id="vm-control-history"`, `vmSelectionGeneration`,
		`vmControlLoadAbort`, `vmControlRetries`, `outcome unknown · retry exact request`,
	} {
		if !strings.Contains(string(index)+string(app), required) {
			t.Fatalf("TrainVM live-control surface is missing %q", required)
		}
	}
}

func TestTrainVMHostAuthorityPanelUsesReceiptDerivedStatus(t *testing.T) {
	index, err := fs.ReadFile(Static(), "index.html")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(Static(), "app.js")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		`id="vm-host-authority-state"`, `id="vm-host-fences"`,
		`id="vm-host-processes"`, `/api/trainvm/host-authority`,
		`mutation_disabled_reason`, `device_policy_installed`,
		`process_policy_installed`, `active_fences_truncated`,
		`active_processes_truncated`,
	} {
		if !strings.Contains(string(index)+string(app), required) {
			t.Fatalf("host authority panel is missing %q", required)
		}
	}
	for _, forbidden := range []string{"/proc/", "pgrep", "process list"} {
		if strings.Contains(string(app), forbidden) {
			t.Fatalf("host authority UI infers authority from %q", forbidden)
		}
	}
}

func TestTrainVMLifecycleActionsUseNativeAuthorityAndExactRetry(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	combined := string(index) + string(app)
	for _, required := range []string{
		`id="vm-action-reason"`, `data-vm-action="checkpoint"`,
		`data-vm-action="pause"`, `data-release-resources="true"`,
		`data-vm-action="resume"`, `data-vm-action="cancel"`,
		`id="vm-cancel-timeout"`, `/actions`, `expected_run_revision`,
		`checkpoint_first: checkpointFirst`, `release_resources: releaseResources`,
		`graceful_timeout_seconds: gracefulTimeout`, `vmActionRetries`,
		`body: submission.body`, `outcome unknown · retry exact action`,
		`observed === "running" && hasWorker`, `observed === "paused" && hasNode`,
	} {
		if !strings.Contains(combined, required) {
			t.Fatalf("TrainVM lifecycle surface is missing %q", required)
		}
	}
	if strings.Contains(string(app), "process.kill") || strings.Contains(string(app), "SIGSTOP") ||
		strings.Contains(string(app), "SIGCONT") {
		t.Fatal("TrainVM lifecycle UI bypasses native authority with process signalling")
	}
}

func TestTrainVMWorkflowGraphIsAuthorityDrivenAndFamilyAgnostic(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	css, err := fs.ReadFile(assets, "app.css")
	if err != nil {
		t.Fatal(err)
	}
	combined := string(index) + string(app) + string(css)
	for _, required := range []string{
		`id="vm-workflow-graph"`, `/api/trainvm/runs/${encodeURIComponent(runID)}/plan`,
		`plan?.canonical_plan?.spec?.workflow`, `node?.transitions`,
		`data-vm-node=`, `vmSelectedRun.current_node_id`, `vmVisitedNodes`,
		`layoutVMWorkflowEdges`, `marker-end`, `.vm-graph-layers`, `.vm-graph-node.current`,
	} {
		if !strings.Contains(combined, required) {
			t.Fatalf("TrainVM workflow graph is missing %q", required)
		}
	}
	graphStart := strings.Index(string(app), "function vmWorkflowModel")
	graphEnd := strings.Index(string(app), "function sameVMValue")
	if graphStart < 0 || graphEnd <= graphStart {
		t.Fatal("could not isolate the workflow graph implementation")
	}
	graphImplementation := strings.ToLower(string(app)[graphStart:graphEnd])
	for _, forbidden := range []string{"mageflow", "rwkv", "transformer"} {
		if strings.Contains(graphImplementation, forbidden) {
			t.Fatalf("workflow graph embeds a family-specific branch for %q", forbidden)
		}
	}
}

func TestTrainVMEditorIsSchemaDrivenAndNativeCompiled(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	editor, err := fs.ReadFile(assets, "trainvm-editor.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(index), `id="trainvm-authoring"`) ||
		!strings.Contains(string(index), `id="vm-schema-form"`) ||
		!strings.Contains(string(index), `/static/trainvm-editor.js`) {
		t.Fatal("TrainVM authoring panel or editor script is missing")
	}
	for _, required := range []string{
		`/api/trainvm/schema`, `resolveSchema`, `node.oneOf`, `resolved.enum`,
		`resolved.additionalProperties`, `/api/trainvm/compile`, `canonical_plan`,
	} {
		if !strings.Contains(string(editor), required) {
			t.Fatalf("TrainVM schema editor is missing %q", required)
		}
	}
	if strings.Contains(string(index), "launch TrainVM") {
		t.Fatal("preview-only TrainVM editor unexpectedly exposes a launch action")
	}
}

func TestTrainVMSubmissionFreezesPreviewAndRetriesExactIntent(t *testing.T) {
	assets := Static()
	editor, err := fs.ReadFile(assets, "trainvm-editor.js")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		`expected_journal_id: authorityJournalID`,
		`expected_plan_hash: validatedDraft.planHash`,
		`expected_adapter_lock_digest: validatedDraft.adapterLockDigest`,
		`expected_training_component_lock_digest:`,
		`validatedDraft.trainingComponentLockDigest`,
		`result.adapter_lock_digest !== intent.adapterLockDigest`,
		`String(result.training_component_lock_digest || "") !==`,
		`fetch("/api/trainvm/training-components"`,
		`fetch("/api/trainvm/operations"`,
		`descriptor.configuration || []`,
		`const value = hasConfigured ? configuration[field.name] : field.default;`,
		`node.invoke.training.components[composerSlot]`,
		`operation.training_composition.model_family`,
		`compatibleComponents(category, composition.model_family)`,
		`data-vm-operation-slot`,
		`data-vm-add-property`,
		`data-vm-add-map`,
		`data-vm-add-array`,
		`#vm-operation-composer input`,
		`sessionStorage.setItem`,
		`body: intent.body`,
		`submissionBusy`,
		`submittedDraftSource === validatedDraft.source`,
		`compileGeneration += 1`,
		`response.status === 408 || response.status >= 500`,
	} {
		if !strings.Contains(string(editor), required) {
			t.Fatalf("TrainVM submission lifecycle is missing %q", required)
		}
	}
	for _, required := range []string{
		`event.detail?.runID`, `selectVMRun(runID)`, `refreshTrainVM(true)`,
	} {
		if !strings.Contains(string(app), required) {
			t.Fatalf("created TrainVM run selection is missing %q", required)
		}
	}
}

func TestTrainVMDescriptorComposerRoundTripsInBrowser(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is not installed")
	}
	command := exec.Command(python, "testdata/trainvm_editor_browser.py")
	output, err := command.CombinedOutput()
	if exitError, ok := err.(*exec.ExitError); ok && exitError.ExitCode() == 77 {
		t.Skip("Playwright is not installed")
	}
	if err != nil {
		t.Fatalf("descriptor composer browser contract failed: %v\n%s", err, output)
	}
}

func TestTrainVMPlanDiffCreatesExplicitImmutableRevisionFork(t *testing.T) {
	assets := Static()
	index, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	editor, err := fs.ReadFile(assets, "trainvm-editor.js")
	if err != nil {
		t.Fatal(err)
	}
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	combined := string(index) + string(editor) + string(app)
	for _, required := range []string{
		`id="vm-diff"`, `id="vm-plan-diff"`,
		`/api/trainvm/runs/${encodeURIComponent(identity.runID)}/diff`,
		`expected_current_plan_hash: identity.planHash`,
		`expected_proposed_plan_hash: preview.planHash`,
		`Array.isArray(result?.semantic_diff)`, `create revision fork`,
		`payload.forked_from_run_id = fork.runID`,
		`payload.expected_parent_run_revision = fork.runRevision`,
		`payload.expected_parent_plan_hash = fork.planHash`,
		`trainvm-run-selected`, `runRevision: Number(run.run_revision || 0)`,
	} {
		if !strings.Contains(combined, required) {
			t.Fatalf("TrainVM immutable plan-fork workflow is missing %q", required)
		}
	}
	if strings.Contains(string(editor), "adopt_plan") {
		t.Fatal("editor attempts to mutate an active plan instead of creating a fork")
	}
}

func TestTrainVMWaitingStatesAreNotRenderedAsTerminalOrControllable(t *testing.T) {
	assets := Static()
	app, err := fs.ReadFile(assets, "app.js")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		`terminalStates.has(run.observed_state)`,
		`"waiting for assignment"`,
		`(!active && !retryLocked)`,
		`Boolean(vmSelectedRun.current_node_id) && Boolean(vmSelectedRun.current_attempt_id)`,
	} {
		if !strings.Contains(string(app), required) {
			t.Fatalf("TrainVM waiting-state UI is missing %q", required)
		}
	}
}
