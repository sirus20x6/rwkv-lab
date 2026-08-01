package web

import (
	"io/fs"
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
		!strings.Contains(string(app), `timeline?after=${vmAfter}&limit=1000`) ||
		!strings.Contains(string(app), `setInterval(refreshTrainVM, 1000)`) {
		t.Fatal("TrainVM panel does not incrementally follow the native journal")
	}
	for _, required := range []string{
		`id="vm-control-catalog"`, `id="vm-control-apply"`,
		`expected_control_revision`, `vmPendingControls`, `randomUUID`,
		`request atomic patch`, `id="vm-control-history"`, `vmSelectionGeneration`,
		`vmControlLoadAbort`, `vmControlRetries`, `outcome unknown · retry exact request`,
	} {
		if !strings.Contains(string(index)+string(app), required) {
			t.Fatalf("TrainVM live-control surface is missing %q", required)
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
