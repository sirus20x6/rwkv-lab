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
