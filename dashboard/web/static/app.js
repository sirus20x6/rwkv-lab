// trainboard front-end glue.
//
// Datastar (loaded as a module in index.html) drives the reactive shell: it
// opens /api/stream on load and morphs the system header, run list, and
// selected-run header in, plus patches the $kpi / $runVersions signals.
//
// This file is the imperative Pixi side. Phase 4 fills in pixi-glue with the
// real charts; for now it's a stub that waits for the active-run node so the
// wiring is in place.

(function () {
  "use strict";

  // The hidden #active-run node carries data-run + data-v; Datastar updates it
  // every tick. Phase 4's chart layer observes it to drive incremental append.
  function watchActiveRun(cb) {
    let last = "";
    const fire = () => {
      // Datastar's morph may replace the node rather than mutate the original.
      // Resolve it each time so the observer never remains attached to a
      // detached element (which previously froze charts until a run switch).
      const node = document.getElementById("active-run");
      if (!node) return;
      const key = (node.dataset.run || "") + ":" + (node.dataset.v || "");
      if (key !== last) {
        last = key;
        cb(node.dataset.run || "", parseInt(node.dataset.v || "0", 10));
      }
    };
    new MutationObserver(fire).observe(document.body, {
      attributes: true, childList: true, subtree: true,
      attributeFilter: ["data-run", "data-v"],
    });
    // Mutation delivery can be missed while a tab is suspended or while
    // Datastar replaces a larger ancestor. This cheap fallback makes the
    // hidden run/version marker self-healing without requiring a run switch.
    setInterval(fire, 1000);
    fire();
  }

  // Exposed for pixi-glue.js: subscribe to (run, version) changes that Datastar
  // publishes by morphing the hidden #active-run node every stream tick.
  let evalSampleRequest = 0;
  let evalRenderedKey = "";
  let evalRenderedSignature = "";
  let evalHistoryRun = "";
  let evalHistorySteps = [];
  let evalHistoryManual = false;
  let evalHistoryRequest = 0;
  let evalHistoryTimer = 0;

  const svgNS = "http://www.w3.org/2000/svg";

  function evalHistoryIndexAtOrBefore(step) {
    if (!evalHistorySteps.length) return -1;
    const target = Number(step);
    let low = 0, high = evalHistorySteps.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (evalHistorySteps[mid] <= target) low = mid + 1;
      else high = mid;
    }
    return Math.max(0, low - 1);
  }

  function renderEvalHistory(index) {
    const controls = document.getElementById("eval-history-controls");
    const range = document.getElementById("eval-history-range");
    const label = document.getElementById("eval-history-step");
    const previous = document.getElementById("eval-history-prev");
    const next = document.getElementById("eval-history-next");
    const latest = document.getElementById("eval-history-latest");
    if (!controls || !range || !label) return;
    controls.hidden = evalHistorySteps.length === 0;
    if (!evalHistorySteps.length) return;
    index = Math.max(0, Math.min(evalHistorySteps.length - 1, Number(index) || 0));
    range.min = "0";
    range.max = String(evalHistorySteps.length - 1);
    range.value = String(index);
    range.disabled = evalHistorySteps.length < 2;
    label.textContent = `step ${evalHistorySteps[index].toLocaleString()} · ${index + 1}/${evalHistorySteps.length}`;
    if (previous) previous.disabled = index === 0;
    if (next) next.disabled = index === evalHistorySteps.length - 1;
    if (latest) latest.classList.toggle(
      "active", !evalHistoryManual && index === evalHistorySteps.length - 1);
  }

  async function refreshEvalHistory(run, selectedStep = NaN, followLatest = false) {
    const request = ++evalHistoryRequest;
    try {
      const res = await fetch(
        `/api/runs/${encodeURIComponent(run)}/eval-samples`,
        { cache: "no-store" });
      if (!res.ok) {
        if (res.status === 404 && request === evalHistoryRequest) {
          evalHistoryRun = run;
          evalHistorySteps = [];
          renderEvalHistory(-1);
        }
        return;
      }
      const data = await res.json();
      if (request !== evalHistoryRequest) return;
      evalHistoryRun = run;
      evalHistorySteps = (data.steps || [])
        .map(Number)
        .filter(Number.isFinite)
        .sort((a, b) => a - b);
      const index = followLatest || !Number.isFinite(Number(selectedStep))
        ? evalHistorySteps.length - 1
        : evalHistoryIndexAtOrBefore(selectedStep);
      renderEvalHistory(index);
    } catch (_) {
      // The gallery remains usable without its navigation index; a later eval
      // or run refresh retries this small request.
    }
  }

  function openEvalHistoryIndex(index) {
    if (!evalHistorySteps.length || !evalHistoryRun) return;
    index = Math.max(0, Math.min(evalHistorySteps.length - 1, Number(index) || 0));
    evalHistoryManual = true;
    renderEvalHistory(index);
    openEvalSamples(evalHistoryRun, evalHistorySteps[index], NaN, 0, "manual");
  }

  // Optional metrics must read as ABSENT, not as zero. The Python side omits
  // keys conditionally -- ocr_generation_metrics only sets eod_rate/maxout_rate
  // when the caller passes the flag arrays, structured_generation_metrics only
  // sets box_iou/mask_dice when target_count > 0 -- and a confident "0.000" for
  // a metric that was never computed is indistinguishable from a real zero on a
  // dashboard whose whole job is reading those numbers.
  function metricText(value, digits = 3) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(digits) : "—";
  }

  function layoutEvalBoxOverlay(card) {
    const visual = card.querySelector('[data-eval-role="visual"]');
    const image = card.querySelector('[data-eval-role="image"]');
    const overlay = card.querySelector('[data-eval-role="box-overlay"]');
    if (!visual || !image || !overlay || !image.naturalWidth || !image.naturalHeight) return;
    const width = image.clientWidth, height = image.clientHeight;
    if (!width || !height) return;
    const naturalRatio = image.naturalWidth / image.naturalHeight;
    const boxRatio = width / height;
    let drawnWidth, drawnHeight, left, top;
    if (naturalRatio > boxRatio) {
      drawnWidth = width; drawnHeight = width / naturalRatio;
      left = 0; top = (height - drawnHeight) / 2;
    } else {
      drawnHeight = height; drawnWidth = height * naturalRatio;
      top = 0; left = (width - drawnWidth) / 2;
    }
    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${drawnWidth}px`;
    overlay.style.height = `${drawnHeight}px`;
  }

  function renderEvalBoxOverlay(card, item) {
    const overlay = card.querySelector('[data-eval-role="box-overlay"]');
    const legend = card.querySelector('[data-eval-role="box-legend"]');
    const target = Array.isArray(item.target_boxes) ? item.target_boxes : [];
    const predicted = Array.isArray(item.predicted_boxes) ? item.predicted_boxes : [];
    const head = Array.isArray(item.head_boxes) ? item.head_boxes : [];
    if (!overlay || !legend) return;
    overlay.replaceChildren();
    for (const [kind, boxes] of [
      ["target", target], ["predicted", predicted], ["head", head],
    ]) {
      for (const box of boxes) {
        const rect = document.createElementNS(svgNS, "rect");
        rect.setAttribute("x", String(box.x1)); rect.setAttribute("y", String(box.y1));
        rect.setAttribute("width", String(box.x2 - box.x1));
        rect.setAttribute("height", String(box.y2 - box.y1));
        rect.setAttribute("vector-effect", "non-scaling-stroke");
        rect.classList.add(`eval-box-${kind}`);
        const title = document.createElementNS(svgNS, "title");
        title.textContent = `${kind === "target" ? "target" : (kind === "head" ? "native head" : "model text")}${box.label ? ` · ${box.label}` : ""}`;
        rect.append(title); overlay.append(rect);
      }
    }
    legend.hidden = target.length + predicted.length + head.length === 0;
    legend.innerHTML = `<span class="target">target ${target.length}</span>` +
      `<span class="predicted">text ${predicted.length}</span>` +
      `<span class="head">native head ${head.length}</span>`;
    const image = card.querySelector('[data-eval-role="image"]');
    if (image && !image.dataset.boxLayoutBound) {
      image.dataset.boxLayoutBound = "1";
      image.addEventListener("load", () => layoutEvalBoxOverlay(card));
    }
    requestAnimationFrame(() => layoutEvalBoxOverlay(card));
  }

  // Clear the loading guard only for the request that is still current.
  // `complete` is false while a newer request is in flight, so a late event
  // from a superseded one cannot unhide a half-loaded card.
  function settleEvalImage(image) {
    if (!image.complete) return;
    image.classList.remove("eval-image-loading");
  }

  function isEvalImageGeneration(evalKind, items) {
    return evalKind === "image_generation" ||
      evalKind.endsWith("_generation") ||
      items.some(item => Boolean(item.target_image_url));
  }

  function renderEvalItems(body, items, evalKind = "") {
    const imageGeneration = isEvalImageGeneration(evalKind, items);
    const existing = Array.from(body.querySelectorAll(":scope > article.eval-sample"));
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      let card = existing[i];
      if (!card) {
        card = document.createElement("article"); card.className = "eval-sample";
        const comparison = document.createElement("div");
        comparison.className = "eval-image-comparison";
        comparison.dataset.evalRole = "comparison";
        const targetVisual = document.createElement("div");
        targetVisual.className = "eval-target-visual";
        targetVisual.dataset.evalRole = "target-visual";
        const targetImg = document.createElement("img");
        targetImg.loading = "lazy";
        targetImg.alt = "held-out original image";
        targetImg.dataset.evalRole = "target-image";
        targetImg.addEventListener("load", () => settleEvalImage(targetImg));
        targetImg.addEventListener("error", () => settleEvalImage(targetImg));
        const targetLabel = document.createElement("span");
        targetLabel.className = "eval-image-label";
        targetLabel.textContent = "original";
        targetVisual.append(targetImg, targetLabel);
        const visual = document.createElement("div");
        visual.className = "eval-sample-visual"; visual.dataset.evalRole = "visual";
        const img = document.createElement("img");
        img.loading = "lazy"; img.alt = "generated eval image"; img.dataset.evalRole = "image";
        img.addEventListener("load", () => settleEvalImage(img));
        img.addEventListener("error", () => settleEvalImage(img));
        const generatedLabel = document.createElement("span");
        generatedLabel.className = "eval-image-label";
        generatedLabel.textContent = "generated";
        const overlay = document.createElementNS(svgNS, "svg");
        overlay.classList.add("eval-box-overlay"); overlay.dataset.evalRole = "box-overlay";
        overlay.setAttribute("viewBox", "0 0 999 999");
        overlay.setAttribute("preserveAspectRatio", "none");
        overlay.setAttribute("aria-hidden", "true");
        const legend = document.createElement("div");
        legend.className = "eval-box-legend"; legend.dataset.evalRole = "box-legend";
        legend.hidden = true;
        visual.append(img, generatedLabel, overlay, legend);
        comparison.append(targetVisual, visual);
        const copy = document.createElement("div"); copy.className = "eval-sample-copy";
        for (const [tag, role, cls] of [
          ["h3", "prompt-heading", ""], ["p", "prompt", "prompt"],
          ["h3", "generated-heading", ""], ["p", "generated", "generated"],
          ["h3", "structured-head-heading", ""],
          ["p", "structured-head", "generated"],
          ["h3", "reference-heading", ""], ["p", "reference", "reference"],
        ]) {
          const node = document.createElement(tag);
          node.dataset.evalRole = role;
          if (cls) node.className = cls;
          copy.append(node);
        }
        card.append(comparison, copy);
        body.append(card);
      }
      card.dataset.evalIndex = String(i);
      const targetVisual = card.querySelector('[data-eval-role="target-visual"]');
      const targetImage = card.querySelector('[data-eval-role="target-image"]');
      const paired = Boolean(item.target_image_url);
      card.classList.toggle("eval-generation", paired);
      if (targetVisual) targetVisual.hidden = !paired;
      if (targetImage && paired &&
          targetImage.getAttribute("src") !== item.target_image_url) {
        targetImage.classList.add("eval-image-loading");
        targetImage.src = item.target_image_url;
      }
      const image = card.querySelector('[data-eval-role="image"]');
      if (image && image.getAttribute("src") !== item.image_url) {
        // Cards are reused by position across snapshots. The caption/reference
        // text below updates synchronously, but a lazy <img> keeps painting the
        // PREVIOUS snapshot's bitmap until the new one decodes -- and an
        // offscreen card defers the fetch indefinitely. The .eval-image-loading
        // guard (app.css) hides the stale bitmap until the new one is ready.
        //
        // Assigning src is enough to arm it. Clearing the attribute first is
        // actively harmful: per HTML's "update the image data" algorithm that
        // queues an `error` task, which -- running after the new src is
        // installed in this same synchronous task -- would strip the guard
        // before the new bitmap decodes and unhide an empty box.
        image.classList.add("eval-image-loading");
        image.src = item.image_url;
      }
      const promptH = card.querySelector('[data-eval-role="prompt-heading"]');
      const prompt = card.querySelector('[data-eval-role="prompt"]');
      if (promptH) {
        promptH.textContent = imageGeneration ? "generation prompt" : "task prompt";
        promptH.hidden = !item.prompt;
      }
      if (prompt) { prompt.textContent = item.prompt || ""; prompt.hidden = !item.prompt; }
      const genH = card.querySelector('[data-eval-role="generated-heading"]');
      const gen = card.querySelector('[data-eval-role="generated"]');
      const refH = card.querySelector('[data-eval-role="reference-heading"]');
      const ref = card.querySelector('[data-eval-role="reference"]');
      const headH = card.querySelector('[data-eval-role="structured-head-heading"]');
      const head = card.querySelector('[data-eval-role="structured-head"]');
      // The server owns this classification (isNegativeStructuredReference in
      // internal/server/eval_samples.go, which is the tested copy). Re-deriving
      // it here only created a second rule to drift from.
      const negativeReference = item.reference_negative === true;
      if (genH) genH.textContent = imageGeneration
        ? "generation settings"
        : `model caption · ${item.tokens} tokens${item.stopped_at_eod ? " · EOD" : (item.complete === false ? " · generating" : " · capped")}`;
      if (gen) gen.textContent = item.caption || "(empty caption)";
      const headInstances = Array.isArray(item.structured_head?.instances)
        ? item.structured_head.instances : [];
      if (headH) {
        headH.textContent = `native structured head · ${headInstances.length} instances`;
        headH.hidden = !item.structured_head;
      }
      if (head) {
        head.textContent = headInstances.map(instance => {
          const box = Array.isArray(instance.box_xyxy)
            ? instance.box_xyxy.map(value => Number(value).toFixed(3)).join(",")
            : "";
          const shape = Array.isArray(instance.mask_shape)
            ? instance.mask_shape.join("×") : "";
          return `q${instance.query} · p=${Number(instance.objectness).toFixed(3)} · ` +
            `box=[${box}] · mask ${shape}=${instance.mask_spans || "empty"}`;
        }).join("\n");
        head.hidden = !item.structured_head;
      }
      if (refH) refH.textContent = imageGeneration
        ? `held-out target · ${item.source || "unknown"}`
        : (negativeReference
          ? `negative reference · concept absent · ${item.source || "unknown"}`
          : `reference · ${item.source || "unknown"}`);
      if (ref) ref.textContent = item.reference || "";
      renderEvalBoxOverlay(card, item);
    }
    for (let i = items.length; i < existing.length; i++) existing[i].remove();
    const empty = body.querySelector(":scope > .empty");
    if (items.length && empty) empty.remove();
    if (!items.length) {
      if (!empty) body.innerHTML = '<div class="empty">snapshot contains no images</div>';
      else empty.textContent = "snapshot contains no images";
    }
  }

  function resetEvalSamples(run) {
    evalSampleRequest++;
    evalHistoryRequest++;
    evalRenderedKey = "";
    evalRenderedSignature = "";
    evalHistoryRun = run || "";
    evalHistorySteps = [];
    evalHistoryManual = false;
    clearTimeout(evalHistoryTimer);
    renderEvalHistory(-1);
    if (run) refreshEvalHistory(run, NaN, true);
    const title = document.getElementById("eval-inline-title");
    const meta = document.getElementById("eval-inline-meta");
    const body = document.getElementById("eval-inline-body");
    if (title) title.textContent = "eval captions";
    if (meta) meta.textContent = run
      ? `${run} · waiting for a qualitative eval snapshot`
      : "waiting for a qualitative eval snapshot";
    if (body) body.innerHTML = '<div class="empty">waiting for a caption snapshot…</div>';
  }
  async function openEvalSamples(run, step, ppl, attempt = 0, mode = "auto") {
    // restart-before-eval can spend well over ten seconds reloading RADIO before
    // the trainer publishes the caption skeleton. Keep checking in the
    // background, but slow down after the first few attempts because many
    // scalar-only eval points will intentionally never get a gallery.
    const exactRetryLimit = 90;
    const exactRetryDelay = attempt < 5 ? 2000 : 5000;
    if (mode === "auto" && evalHistoryManual && evalHistoryRun === run) {
      const range = document.getElementById("eval-history-range");
      const index = Math.max(0, Math.min(
        evalHistorySteps.length - 1,
        Number(range?.value) || 0,
      ));
      refreshEvalHistory(run, evalHistorySteps[index], false);
      return;
    }
    if (mode === "manual") evalHistoryManual = true;
    const request = ++evalSampleRequest;
    const panel = document.getElementById("eval-sample-inline");
    const title = document.getElementById("eval-inline-title");
    const meta = document.getElementById("eval-inline-meta");
    const body = document.getElementById("eval-inline-body");
    if (!panel || !body) return;
    const snapshotKey = `${run}:${Number(step)}:${Number.isFinite(ppl) ? Number(ppl).toPrecision(12) : ""}`;
    const changingSnapshot = snapshotKey !== evalRenderedKey;
    if (changingSnapshot) {
      evalRenderedKey = snapshotKey;
      evalRenderedSignature = "";
      // Keep the previous completed gallery visible until a newer artifact
      // actually exists. Scalar PPL can run more often than qualitative eval,
      // and a scalar-only marker must not blank the most recent captions.
      if (!body.querySelector(":scope > article.eval-sample")) {
        if (title) title.textContent = "eval captions · step " + Number(step).toLocaleString();
        if (meta) meta.textContent = run + (Number.isFinite(ppl) ? " · ppl " + Number(ppl).toFixed(3) : "");
        body.innerHTML = '<div class="empty">loading qualitative snapshot…</div>';
      }
    }
    const url = `/api/runs/${encodeURIComponent(run)}/eval-samples/${encodeURIComponent(step)}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      let res = await fetch(url, { signal: controller.signal, cache: "no-store" });
      let fallback = false;
      if (res.status === 404 && attempt < exactRetryLimit) {
        setTimeout(() => {
          if (request === evalSampleRequest) {
            openEvalSamples(run, step, ppl, attempt + 1, mode);
          }
        }, exactRetryDelay);
      }
      if (res.status === 404 && !body.querySelector(":scope > article.eval-sample")) {
        // PPL is commonly measured more often than captions are generated.
        // Fill a freshly loaded/reset card immediately with the newest real
        // gallery, while the retries above continue looking for an exact-step
        // artifact that may only just be starting.
        const latestURL = `/api/runs/${encodeURIComponent(run)}/eval-samples/latest?at_step=${encodeURIComponent(step)}`;
        const latest = await fetch(latestURL, { signal: controller.signal, cache: "no-store" });
        if (latest.ok) {
          res = latest;
          fallback = true;
        } else if (attempt < exactRetryLimit) {
          body.innerHTML = '<div class="empty">caption snapshot is starting…</div>';
          return;
        }
      } else if (res.status === 404) {
        return; // scalar-only eval marker; retain the newest qualitative card
      }
      if (!res.ok) throw new Error(res.status === 404
        ? "No caption snapshot was recorded for this older eval. New eval markers will include one."
        : `snapshot request failed (${res.status})`);
      const data = await res.json();
      if (request !== evalSampleRequest) return;
      // Recovery can re-evaluate the same step while the prior generation's
      // complete artifact still exists.  The scalar eval is logged before the
      // new caption skeleton is published, so a blind 200 response would make
      // us display old captions forever (complete=true stops normal polling).
      const artifactStep = Number(data.step), artifactPPL = Number(data.ppl);
      const expectedStep = Number(step), expectedPPL = Number(ppl);
      const pplMismatch = Number.isFinite(expectedPPL) && Number.isFinite(artifactPPL) &&
        Math.abs(artifactPPL - expectedPPL) > 1e-9 * Math.max(1, Math.abs(expectedPPL));
      const stale = !fallback && (artifactStep !== expectedStep || pplMismatch);
      if (stale && attempt < 5) {
        if (meta) meta.textContent = `${run} · ppl ${expectedPPL.toFixed(3)} · replacing stale same-step snapshot`;
        body.innerHTML = '<div class="empty">waiting for this eval generation’s captions…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) {
            openEvalSamples(run, step, ppl, attempt + 1, mode);
          }
        }, 2000);
        return;
      }
      // Retries exhausted on a permanently stale artifact (e.g. dead run whose
      // recovery never republished): show what exists, flagged, and stop polling.
      const pending = data.complete === false && !stale;
      const structured = data.structured_generation;
      const ocr = data.ocr_generation;
      const generatedOCR = !pending && ocr &&
        typeof ocr.normalized_cer === "number"
        ? ` · OCR CER ${metricText(ocr.normalized_cer)}` +
          ` · WER ${metricText(ocr.wer)}` +
          ` · line acc ${metricText(ocr.line_accuracy)}` +
          ` · EOD/max ${metricText(ocr.eod_rate)}/${metricText(ocr.maxout_rate)}`
        : "";
      const generatedGeometry = !pending && structured &&
        typeof structured.box_iou === "number"
        ? ` · greedy box IoU ${metricText(structured.box_iou)}` +
          ` · mask Dice ${metricText(structured.mask_dice)}` +
          ` · P/R@.5 ${metricText(structured.precision_at_50)}/${metricText(structured.recall_at_50)}` +
          (typeof structured.invalid_predictions === "number" &&
           structured.invalid_predictions > 0
            ? ` · invalid ${structured.invalid_predictions.toLocaleString()}` : "")
        : "";
      const items = data.items || [];
      refreshEvalHistory(
        run,
        artifactStep,
        mode === "auto" && !evalHistoryManual,
      );
      const imageGeneration = isEvalImageGeneration(data.eval_kind || "", items);
      if (title) title.textContent = (imageGeneration ? "eval generations · step " : "eval captions · step ") +
        artifactStep.toLocaleString();
      if (meta) meta.textContent = imageGeneration
        ? `${run} · ${Number(data.generation_steps || 0)} denoise steps · ${items.length} fixed held-out prompts`
        : `${run} · ppl ${Number(data.ppl).toFixed(3)} · ${data.decoding || "greedy"} decoding` +
        (pending ? ` · generating ${Number(data.generation_steps || 0)}/${Number(data.max_new || 0)}` : "") +
        generatedOCR +
        generatedGeometry +
        (fallback ? ` · newest gallery before scalar eval ${expectedStep.toLocaleString()}` : "") +
        (stale ? " · stale snapshot from an earlier generation of this step" : "");
      // Whole-payload change detector: every field the card renders belongs
      // here, including the two generation-metric maps. They only happen to be
      // safe to omit today because meta.textContent is rewritten
      // unconditionally below.
      const signature = JSON.stringify([
        data.eval_kind, data.complete, data.generation_steps, ocr, structured,
        items.map(item => [item.image_url, item.tokens, item.stopped_at_eod,
                            item.complete, item.caption, item.reference, item.reference_negative,
                            item.prompt, item.source, item.target_image_url,
                            item.target_boxes, item.predicted_boxes,
                            item.head_boxes, item.structured_head]),
      ]);
      // Preserve image/card DOM across polls. Replacing it forced the browser
      // to repaint every image, which made the gallery visibly flash.
      if (signature !== evalRenderedSignature) {
        renderEvalItems(body, items, data.eval_kind);
        evalRenderedSignature = signature;
      }
      // A scalar eval is published before the expensive greedy captions. Poll
      // only while that same card request remains current; switching run/eval
      // invalidates the request and prevents a stale response from taking over.
      if (pending) {
        setTimeout(() => {
          if (request === evalSampleRequest) {
            openEvalSamples(run, step, ppl, 0, mode);
          }
        }, 2000);
      }
    } catch (err) {
      if (request !== evalSampleRequest) return;
      if (err && err.name === "AbortError") {
        body.innerHTML = '<div class="empty">snapshot request timed out; retrying…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) {
            openEvalSamples(run, step, ppl, attempt, mode);
          }
        }, 2000);
        return;
      }
      body.innerHTML = "";
      const msg = document.createElement("div"); msg.className = "empty"; msg.textContent = err.message || String(err); body.append(msg);
    } finally {
      clearTimeout(timeout);
    }
  }

  function wireEvalHistoryControls() {
    const range = document.getElementById("eval-history-range");
    const previous = document.getElementById("eval-history-prev");
    const next = document.getElementById("eval-history-next");
    const latest = document.getElementById("eval-history-latest");
    if (!range) return;
    range.addEventListener("input", () => {
      const index = Number(range.value);
      renderEvalHistory(index);
      clearTimeout(evalHistoryTimer);
      evalHistoryTimer = setTimeout(() => openEvalHistoryIndex(index), 90);
    });
    if (previous) previous.addEventListener("click", () => {
      openEvalHistoryIndex(Number(range.value) - 1);
    });
    if (next) next.addEventListener("click", () => {
      openEvalHistoryIndex(Number(range.value) + 1);
    });
    if (latest) latest.addEventListener("click", () => {
      if (!evalHistorySteps.length || !evalHistoryRun) return;
      evalHistoryManual = false;
      const index = evalHistorySteps.length - 1;
      renderEvalHistory(index);
      openEvalSamples(evalHistoryRun, evalHistorySteps[index], NaN, 0, "auto");
    });
  }
  wireEvalHistoryControls();

  // ---- TrainVM native read model. The browser only receives read-only
  // projections; lifecycle mutations remain on the native command boundary.
  let vmSelected = "";
  let vmJournalID = "";
  let vmAfter = 0;
  let vmTelemetryAfter = 0;
  let vmTelemetryJournal = "";
  let vmLatestHeartbeat = null;
  let vmObservability = null;
  let vmMetricRenderSignature = null;
  let vmArtifactRenderSignature = null;
  const vmEvalGallerySchema = "rwkv-lab.eval-gallery.v2";
  const vmMetricSeriesLimit = 512;
  const vmMetricSeries = new Map();
  const vmArtifacts = new Map();
  const vmExecutionPhases = new Map();
  const vmCheckpointSummaries = new Map();
  let vmGalleries = [];
  let vmGalleryIndex = -1;
  let vmGalleryManual = false;
  let vmGallerySignature = "";
  let vmGalleryHistoryTruncated = false;
  let vmGalleryLoadGeneration = 0;
  let vmProfileSignature = "";
  let vmProfiles = [];
  let vmProfileBaseline = "";
  let vmBusy = false;
  let vmSelectedRun = null;
  let vmControlView = null;
  let vmControlRun = "";
  let vmPendingControls = {};
  let vmControlIntent = "";
  let vmCommandsEnabled = false;
  let vmSelectionGeneration = 0;
  let vmControlLoadGeneration = 0;
  let vmControlLoadAbort = null;
  let vmSubmitBusy = false;
  let vmActionBusy = false;
  let vmCompiledPlan = null;
  let vmPlanRun = "";
  let vmPlanLoadGeneration = 0;
  let vmWorkflowEdges = [];
  let vmWorkflowSignature = "";
  let vmHostBusy = false;
  const vmVisitedNodes = new Set();
  const vmInvalidControls = new Set();
  const vmControlRetries = new Map();
  const vmActionRetries = new Map();

  function vmEscape(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }

  function renderVMHostAuthority(status) {
    const state = document.getElementById("vm-host-authority-state");
    const summary = document.getElementById("vm-host-authority-summary");
    const fences = document.getElementById("vm-host-fences");
    const processes = document.getElementById("vm-host-processes");
    if (!state || !summary || !fences || !processes) return;
    const resourceHealthy = Boolean(status.resource_inventory_observed) && Number(status.degraded_resource_count || 0) === 0;
    const renewalHealthy = !status.lease_renewal_poisoned && !status.lease_renewal_failure;
    const healthy = Boolean(status.mutation_enabled) && resourceHealthy && renewalHealthy;
    state.textContent = !status.mutation_enabled ?
      `disabled · ${status.mutation_disabled_reason || "authority did not supply a reason"}` :
      !resourceHealthy ? `admitting · ${status.resource_degradation_reason || "resource health unknown"}` :
      !renewalHealthy ? `admitting · renewal degraded · ${status.lease_renewal_failure || "coordinator poisoned"}` :
      "admitting · mutations enabled";
    state.className = healthy ? "ok" : "bad";
    const fact = (label, value, title = "") =>
      `<div class="vm-fact"><span>${vmEscape(label)}</span><strong title="${vmEscape(title || value)}">${vmEscape(value ?? "—")}</strong></div>`;
    const coordinator = status.coordinator || {};
    summary.innerHTML =
      fact("coordinator", coordinator.lifecycle || "unknown", coordinator.poison_reason || "") +
      fact("startup", status.startup_phase || "unknown") +
      fact("ledger", status.ledger_verified ? `verified · #${Number(status.ledger_sequence || 0).toLocaleString()}` : "verification failed", status.ledger_verification_reason || status.ledger_chain_hash || "") +
      fact("launch", status.process_launch_enabled ? "strict enabled" : "disabled") +
      fact("fences", Number(status.active_fence_count || 0).toLocaleString()) +
      fact("resource health", status.resource_inventory_observed ?
        (Number(status.degraded_resource_count || 0) === 0 ?
          `intact · ${Math.round(Number(status.resource_inventory_observation_age_ns || 0) / 1e9)}s old` :
          `${Number(status.degraded_resource_count).toLocaleString()} degraded`) :
        "observation failed", status.resource_degradation_reason || status.current_inventory_receipt_digest || "") +
      fact("processes", Number(status.active_process_count || 0).toLocaleString()) +
      fact("lease renewal", status.lease_renewal_poisoned ? "poisoned" :
        `${Number(status.lease_renewal_tracked_count || 0).toLocaleString()} tracked`, status.lease_renewal_failure || "") +
      fact("recovery backlog", `${Number(status.remaining_unclosed_process_records || 0).toLocaleString()} live · ${Number(status.remaining_terminal_release_records || 0).toLocaleString()} release`) +
      fact("startup audit", coordinator.startup_audit_receipt_digest ? (coordinator.startup_audit_passed ? "passed" : "failed") : "not committed", coordinator.startup_audit_receipt_digest || "");
    const fenceRows = Array.isArray(status.active_fences) ? status.active_fences : [];
    fences.innerHTML = fenceRows.map((fence) =>
      `<div class="vm-authority-row"><span title="${vmEscape(fence.stable_id)}">${vmEscape(fence.stable_id)}</span>` +
      `<span>${vmEscape([fence.kind, fence.vendor].filter(Boolean).join(" · "))}</span><span>gen ${Number(fence.generation || 0).toLocaleString()}</span></div>`
    ).join("") || '<div class="empty">no active resource fences</div>';
    if (status.active_fences_truncated) fences.insertAdjacentHTML("beforeend", '<div class="empty">bounded prefix shown · more fences are active</div>');
    const processRows = Array.isArray(status.active_processes) ? status.active_processes : [];
    processes.innerHTML = processRows.map((process) => {
      const device = process.device_policy_intended ? (process.device_policy_installed ? "device ✓" : "device ✗") : "device n/a";
      const policy = process.process_policy_intended ? (process.process_policy_installed ? "host policy ✓" : "host policy ✗") : "host policy n/a";
      const incomplete = (process.device_policy_intended && !process.device_policy_installed) ||
        (process.process_policy_intended && !process.process_policy_installed);
      const terminal = process.phase === "terminal_pending_release";
      const cleanupComplete = terminal && process.cgroup_empty === true && process.accelerator_contexts_empty === true;
      const cleanup = terminal ? ` · cleanup ${cleanupComplete ? "✓" : "✗"}` : "";
      return `<div class="vm-authority-row"><span title="${vmEscape(process.launch_id)}">${vmEscape(process.run_id || process.launch_id)}</span>` +
        `<span>${vmEscape(process.phase)} · pid ${process.host_pid == null ? "—" : Number(process.host_pid).toLocaleString()}</span>` +
        `<span class="${incomplete || (terminal && !cleanupComplete) ? "bad" : "ok"}">${vmEscape(device)} · ${vmEscape(policy)}${vmEscape(cleanup)}</span></div>`;
    }).join("") || '<div class="empty">no active process authority records</div>';
    if (status.active_processes_truncated) processes.insertAdjacentHTML("beforeend", '<div class="empty">bounded prefix shown · more process records are active</div>');
  }

  function renderVMHostAuthorityUnavailable(message) {
    const state = document.getElementById("vm-host-authority-state");
    const summary = document.getElementById("vm-host-authority-summary");
    const fences = document.getElementById("vm-host-fences");
    const processes = document.getElementById("vm-host-processes");
    if (state) {
      state.textContent = message;
      state.className = "bad";
    }
    if (summary) summary.innerHTML = '<div class="empty">no authority receipt available</div>';
    if (fences) fences.innerHTML = '<div class="empty">not observable</div>';
    if (processes) processes.innerHTML = '<div class="empty">not observable</div>';
  }

  async function refreshVMHostAuthority() {
    if (vmHostBusy) return;
    vmHostBusy = true;
    try {
      const response = await fetch("/api/trainvm/host-authority", { cache: "no-store" });
      if (!response.ok) {
        renderVMHostAuthorityUnavailable(`authority unavailable · HTTP ${response.status}`);
        return;
      }
      const payload = await response.json();
      renderVMHostAuthority(payload);
    } catch (_) {
      renderVMHostAuthorityUnavailable("authority transport unavailable");
    } finally {
      vmHostBusy = false;
    }
  }

  function renderVMRunList(runs) {
    const target = document.getElementById("trainvm-runs");
    if (!target) return;
    target.innerHTML = runs.map((run) => `
      <button class="vm-run${run.run_id === vmSelected ? " active" : ""}"
              type="button" data-vm-run="${vmEscape(run.run_id)}">
        <span class="vm-run-name">${vmEscape(run.experiment_name || run.run_id)}</span>
        <span class="vm-run-state"><span>${vmEscape(run.observed_state)}</span><span>r${Number(run.run_revision || 0)}</span></span>
      </button>`).join("") || '<div class="empty">no native runs yet</div>';
  }

  function renderVMSummary(run) {
    const target = document.getElementById("trainvm-summary");
    if (!target) return;
    const terminalStates = new Set(["completed", "cancelled", "failed"]);
    const nodeFallback = terminalStates.has(run.observed_state) ? "terminal" : "waiting for assignment";
    const fact = (label, value, title = "") =>
      `<div class="vm-fact"><span>${vmEscape(label)}</span><strong title="${vmEscape(title || value)}">${vmEscape(value || "—")}</strong></div>`;
    target.innerHTML =
      fact("desired", run.desired_state) + fact("observed", run.observed_state) +
      fact("node", run.current_node_id || nodeFallback, run.current_node_id) +
      fact("attempt", run.current_attempt_id || "—") +
      fact("optimizer step", Number(run.optimizer_step || 0).toLocaleString()) +
      fact("revision", String(run.run_revision || 0)) +
      fact("experiment", run.experiment_name, run.experiment_name) +
      fact("plan", String(run.plan_hash || "").slice(0, 12), run.plan_hash) +
      // Lineage is shown only when the authority reports it. A run with no
      // parent shows nothing rather than an invented "root", so provenance is
      // never something the view guessed.
      (run.forked_from_run_id
        ? fact(
            "forked from",
            `${run.forked_from_run_id} · r${run.forked_from_run_revision || 0}`,
            `parent plan ${run.forked_from_plan_hash || "unknown"}`)
        : "");
    renderVMActions();
  }

  function vmActionVariant(action, checkpointFirst = false, releaseResources = false) {
    return action === "pause" && checkpointFirst && releaseResources ? "pause-release" : action;
  }

  function renderVMActions() {
    const reason = document.getElementById("vm-action-reason");
    const timeout = document.getElementById("vm-cancel-timeout");
    const state = document.getElementById("vm-action-state");
    const retry = vmSelectedRun ? vmActionRetries.get(vmSelectedRun.run_id) : null;
    const controlRetryLocked = vmSelectedRun ? vmControlRetries.has(vmSelectedRun.run_id) : false;
    const observed = String(vmSelectedRun?.observed_state || "");
    const hasNode = Boolean(vmSelectedRun?.current_node_id);
    const hasWorker = hasNode && Boolean(vmSelectedRun?.current_attempt_id);
    const allowed = new Set();
    if (observed === "running" && hasWorker) {
      allowed.add("checkpoint");
      allowed.add("pause");
      allowed.add("pause-release");
      allowed.add("cancel");
    } else if (observed === "paused" && hasNode) {
      allowed.add("resume");
      if (hasWorker) allowed.add("cancel");
    }
    for (const button of document.querySelectorAll("[data-vm-action]")) {
      const checkpointFirst = button.dataset.checkpointFirst === "true";
      const releaseResources = button.dataset.releaseResources === "true";
      const variant = vmActionVariant(button.dataset.vmAction || "", checkpointFirst, releaseResources);
      const retryMatch = retry?.variant === variant;
      button.hidden = !allowed.has(variant) && !retryMatch;
      button.disabled = !vmCommandsEnabled || vmSubmitBusy || vmActionBusy || controlRetryLocked ||
        Boolean(retry && !retryMatch) || (!allowed.has(variant) && !retryMatch);
      if (retryMatch) button.textContent = `retry ${retry.label}`;
      else if (variant === "pause-release") button.textContent = "pause & release GPU";
      else if (variant === "checkpoint") button.textContent = "checkpoint now";
      else button.textContent = variant;
    }
    if (reason) {
      reason.disabled = !vmSelectedRun || !vmCommandsEnabled || vmSubmitBusy || vmActionBusy || controlRetryLocked || Boolean(retry);
      if (retry && reason.value !== retry.reason) reason.value = retry.reason;
    }
    if (timeout) {
      timeout.disabled = !vmSelectedRun || !vmCommandsEnabled || vmSubmitBusy || vmActionBusy || controlRetryLocked || Boolean(retry) || !allowed.has("cancel");
      if (retry?.variant === "cancel") timeout.value = String(retry.payload.graceful_timeout_seconds);
    }
    if (state && !vmSelectedRun) {
      state.textContent = "select a native run";
      state.className = "sub";
    } else if (state && retry) {
      state.textContent = "outcome unknown · retry exact action";
      state.className = "sub error";
    } else if (state && controlRetryLocked) {
      state.textContent = "retry the exact pending control request first";
      state.className = "sub error";
    } else if (state && !vmCommandsEnabled) {
      state.textContent = "native command socket not attached";
      state.className = "sub error";
    } else if (state && !allowed.size && !vmActionBusy) {
      state.textContent = `no lifecycle action available while ${observed || "unselected"}`;
      state.className = "sub";
    }
  }

  function resetVMWorkflow(runID = "") {
    vmCompiledPlan = null;
    vmPlanRun = "";
    vmPlanLoadGeneration += 1;
    vmWorkflowEdges = [];
    vmWorkflowSignature = "";
    vmVisitedNodes.clear();
    const graph = document.getElementById("vm-workflow-graph");
    const state = document.getElementById("vm-workflow-state");
    if (graph) graph.innerHTML = `<div class="empty">${runID ? "loading authority plan…" : "select a native run"}</div>`;
    if (state) state.textContent = runID ? "loading immutable plan" : "select a native run";
  }

  function resetVMControls(runID = "") {
    vmControlRun = "";
    vmControlView = null;
    const retry = runID ? vmControlRetries.get(runID) : null;
    vmPendingControls = retry ? cloneVMValues(retry.assignments) : {};
    vmControlIntent = retry?.intent || "";
    vmInvalidControls.clear();
    if (vmControlLoadAbort) vmControlLoadAbort.abort();
    vmControlLoadAbort = null;
    vmControlLoadGeneration += 1;
    const target = document.getElementById("vm-control-catalog");
    if (target) target.innerHTML = runID ? '<div class="empty">loading declared controls…</div>' : '<div class="empty">select a native run</div>';
    const history = document.getElementById("vm-control-history");
    if (history) history.innerHTML = '<div class="empty">no command history loaded</div>';
    const apply = document.getElementById("vm-control-apply");
    if (apply) apply.disabled = true;
    const reason = document.getElementById("vm-control-reason");
    if (reason) {
      reason.value = retry?.reason || "";
      reason.disabled = true;
    }
    const state = document.getElementById("vm-control-state");
    if (state) {
      state.textContent = retry ? "outcome unknown · retry exact request" :
        (runID ? "loading selected run…" : "no run selected");
      state.className = retry ? "sub error" : "sub";
    }
  }

  function vmWorkflowModel(plan) {
    const workflow = plan?.canonical_plan?.spec?.workflow;
    const declared = workflow?.nodes && typeof workflow.nodes === "object" ? workflow.nodes : {};
    const entrypoint = String(workflow?.entrypoint || "");
    const nodes = new Map(Object.entries(declared));
    const edges = [];
    for (const [source, node] of nodes) {
      for (const transition of Array.isArray(node?.transitions) ? node.transitions : []) {
        const target = String(transition?.target || "");
        if (!target) continue;
        if (!nodes.has(target)) nodes.set(target, null);
        edges.push({ source, target, event: String(transition?.on || "transition"), where: transition?.where || null });
      }
    }
    const distance = new Map();
    if (nodes.has(entrypoint)) distance.set(entrypoint, 0);
    for (let pass = 0; pass < nodes.size; pass += 1) {
      let changed = false;
      for (const edge of edges) {
        if (!distance.has(edge.source) || edge.target === entrypoint) continue;
        const candidate = distance.get(edge.source) + 1;
        if (!distance.has(edge.target) || candidate < distance.get(edge.target)) {
          distance.set(edge.target, candidate);
          changed = true;
        }
      }
      if (!changed) break;
    }
    const lastLayer = Math.max(0, ...distance.values()) + 1;
    for (const name of nodes.keys()) if (!distance.has(name)) distance.set(name, lastLayer);
    const layers = new Map();
    for (const name of nodes.keys()) {
      const layer = distance.get(name);
      if (!layers.has(layer)) layers.set(layer, []);
      layers.get(layer).push(name);
    }
    for (const names of layers.values()) names.sort((left, right) => left.localeCompare(right));
    for (const edge of edges) edge.back = distance.get(edge.target) <= distance.get(edge.source);
    return { entrypoint, nodes, edges, layers: [...layers.entries()].sort((left, right) => left[0] - right[0]) };
  }

  function renderVMWorkflowGraph() {
    const target = document.getElementById("vm-workflow-graph");
    const state = document.getElementById("vm-workflow-state");
    if (!target || !vmCompiledPlan || !vmSelectedRun || vmCompiledPlan.run_id !== vmSelectedRun.run_id) return;
    const signature = JSON.stringify([
      vmCompiledPlan.plan_hash, vmSelectedRun.current_node_id, vmSelectedRun.current_attempt_id,
      vmSelectedRun.observed_state, [...vmVisitedNodes].sort(),
    ]);
    if (signature === vmWorkflowSignature && target.querySelector(".vm-graph-canvas")) return;
    vmWorkflowSignature = signature;
    const model = vmWorkflowModel(vmCompiledPlan);
    if (!model.nodes.size || !model.entrypoint) {
      target.innerHTML = '<div class="empty">compiled plan has no workflow graph</div>';
      if (state) state.textContent = `plan ${String(vmCompiledPlan.plan_hash || "").slice(0, 12)}`;
      return;
    }
    const outgoing = new Map();
    for (const edge of model.edges) {
      if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
      outgoing.get(edge.source).push(edge);
    }
    const terminalObserved = String(vmSelectedRun.observed_state || "");
    const layers = model.layers.map(([layer, names]) =>
      `<div class="vm-graph-layer" data-vm-layer="${layer}">` + names.map((name) => {
        const node = model.nodes.get(name);
        const terminal = !node || name.startsWith("$");
        const current = name === vmSelectedRun.current_node_id;
        const observedTerminal = terminal && ((name === "$failed" && terminalObserved === "failed") ||
          (name === "$completed" && terminalObserved === "completed") ||
          (name === "$cancelled" && terminalObserved === "cancelled"));
        const visited = current || observedTerminal || vmVisitedNodes.has(name);
        const invoke = node?.invoke || {};
        const operation = terminal ? "terminal state" : `${invoke.component || "?"}.${invoke.operation || "?"}`;
        const effect = terminal ? "terminal" : String(node.effect || "effect");
        const transitions = (outgoing.get(name) || []).map((edge) =>
          `<span title="${vmEscape(edge.where ? JSON.stringify(edge.where) : edge.event)}">${vmEscape(edge.event)} → ${vmEscape(edge.target)}</span>`).join("");
        const classes = ["vm-graph-node", current ? "current" : "", visited ? "visited" : "", terminal ? "terminal" : ""].filter(Boolean).join(" ");
        return `<article class="${classes}" data-vm-node="${vmEscape(name)}" title="${vmEscape(node?.description || name)}">` +
          `<div class="vm-node-head"><strong>${vmEscape(name)}</strong><span>${vmEscape(current ? "current" : effect)}</span></div>` +
          `<div class="vm-node-operation">${vmEscape(operation)}</div>` +
          (current && vmSelectedRun.current_attempt_id ? `<div class="vm-node-attempt">${vmEscape(vmSelectedRun.current_attempt_id)}</div>` : "") +
          (transitions ? `<div class="vm-node-transitions">${transitions}</div>` : "") + `</article>`;
      }).join("") + `</div>`).join("");
    vmWorkflowEdges = model.edges;
    target.innerHTML = `<div class="vm-graph-canvas"><svg class="vm-graph-edges" aria-hidden="true"></svg><div class="vm-graph-layers">${layers}</div></div>`;
    if (state) state.textContent = `${model.nodes.size} nodes · ${model.edges.length} transitions · ${String(vmCompiledPlan.plan_hash || "").slice(0, 12)}`;
    requestAnimationFrame(layoutVMWorkflowEdges);
  }

  function layoutVMWorkflowEdges() {
    const target = document.getElementById("vm-workflow-graph");
    const canvas = target?.querySelector(".vm-graph-canvas");
    const svg = canvas?.querySelector(".vm-graph-edges");
    if (!canvas || !svg || !vmWorkflowEdges.length) return;
    const namespace = "http://www.w3.org/2000/svg";
    const width = Math.max(canvas.scrollWidth, canvas.clientWidth);
    const height = Math.max(canvas.scrollHeight, canvas.clientHeight);
    svg.replaceChildren();
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const defs = document.createElementNS(namespace, "defs");
    const marker = document.createElementNS(namespace, "marker");
    marker.setAttribute("id", "vm-graph-arrow");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "5");
    marker.setAttribute("markerHeight", "5");
    marker.setAttribute("orient", "auto-start-reverse");
    const arrow = document.createElementNS(namespace, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    arrow.setAttribute("fill", "currentColor");
    marker.appendChild(arrow);
    defs.appendChild(marker);
    svg.appendChild(defs);
    const root = canvas.getBoundingClientRect();
    const nodeElements = new Map([...canvas.querySelectorAll("[data-vm-node]")].map((node) => [node.dataset.vmNode, node]));
    for (const edge of vmWorkflowEdges) {
      const source = nodeElements.get(edge.source);
      const destination = nodeElements.get(edge.target);
      if (!source || !destination) continue;
      const from = source.getBoundingClientRect();
      const to = destination.getBoundingClientRect();
      const startX = from.right - root.left;
      const startY = from.top + from.height / 2 - root.top;
      let endX = to.left - root.left;
      let endY = to.top + to.height / 2 - root.top;
      let pathData;
      if (source === destination) {
        endX = from.right - root.left;
        endY = from.top + from.height * .75 - root.top;
        pathData = `M ${startX} ${startY} C ${startX + 42} ${startY - 34}, ${endX + 42} ${endY + 34}, ${endX} ${endY}`;
      } else if (edge.back) {
        const bendY = Math.max(8, Math.min(startY, endY) - 24);
        pathData = `M ${startX} ${startY} C ${startX + 32} ${bendY}, ${endX - 32} ${bendY}, ${endX} ${endY}`;
      } else {
        const control = Math.max(28, (endX - startX) * .45);
        pathData = `M ${startX} ${startY} C ${startX + control} ${startY}, ${endX - control} ${endY}, ${endX} ${endY}`;
      }
      const path = document.createElementNS(namespace, "path");
      path.setAttribute("d", pathData);
      path.setAttribute("marker-end", "url(#vm-graph-arrow)");
      if (edge.back) path.setAttribute("class", "back");
      const title = document.createElementNS(namespace, "title");
      title.textContent = `${edge.source} · ${edge.event} → ${edge.target}`;
      path.appendChild(title);
      svg.appendChild(path);
    }
  }

  async function refreshVMCompiledPlan(run, force = false) {
    if (!run) return;
    if (!force && vmPlanRun === run.run_id && vmCompiledPlan?.plan_hash === run.plan_hash) {
      renderVMWorkflowGraph();
      return;
    }
    const runID = run.run_id;
    const selectionGeneration = vmSelectionGeneration;
    const loadGeneration = ++vmPlanLoadGeneration;
    const state = document.getElementById("vm-workflow-state");
    if (state) state.textContent = "loading immutable plan";
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(runID)}/plan`, { cache: "no-store" });
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmPlanLoadGeneration || runID !== vmSelected) return;
      if (!response.ok) throw new Error((await response.text()).trim() || `HTTP ${response.status}`);
      const plan = await response.json();
      if (plan.run_id !== runID || plan.plan_hash !== run.plan_hash ||
          (vmJournalID && plan.journal_id !== vmJournalID) ||
          !plan.canonical_plan || typeof plan.canonical_plan !== "object") {
        throw new Error("compiled-plan identity mismatch");
      }
      vmPlanRun = runID;
      vmCompiledPlan = plan;
      renderVMWorkflowGraph();
    } catch (error) {
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmPlanLoadGeneration || runID !== vmSelected) return;
      const graph = document.getElementById("vm-workflow-graph");
      if (graph) graph.innerHTML = `<div class="empty">workflow unavailable · ${vmEscape(error.message)}</div>`;
      if (state) state.textContent = "authority plan unavailable";
    }
  }

  function sameVMValue(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
  }

  function cloneVMValues(values) {
    return JSON.parse(JSON.stringify(values || {}));
  }

  function publishVMRunSelection(run) {
    const detail = run ? {
      runID: String(run.run_id || ""),
      runRevision: Number(run.run_revision || 0),
      planHash: String(run.plan_hash || ""),
    } : null;
    const previous = window.__trainVMSelectedRun || null;
    if (previous?.runID === detail?.runID && previous?.runRevision === detail?.runRevision &&
        previous?.planHash === detail?.planHash) return;
    window.__trainVMSelectedRun = detail;
    window.dispatchEvent(new CustomEvent("trainvm-run-selected", { detail }));
  }

  function selectVMRun(runID) {
    if (vmSelected === runID) return false;
    vmSelected = runID;
    vmSelectionGeneration += 1;
    vmSelectedRun = null;
    publishVMRunSelection(null);
    resetVMWorkflow(runID);
    resetVMGallery(runID);
    resetVMProfiles(runID);
    resetVMControls(runID);
    const actionRetry = vmActionRetries.get(runID);
    const actionReason = document.getElementById("vm-action-reason");
    if (actionReason) {
      actionReason.value = actionRetry?.reason || "";
      actionReason.disabled = true;
    }
    const actionState = document.getElementById("vm-action-state");
    if (actionState) {
      actionState.textContent = actionRetry ? "outcome unknown · retry exact action" :
        (runID ? "loading selected run…" : "select a native run");
      actionState.className = actionRetry ? "sub error" : "sub";
    }
    renderVMActions();
    return true;
  }

  function resetVMGallery(runID = "") {
    vmGalleries = [];
    vmGalleryIndex = -1;
    vmGalleryManual = false;
    vmGallerySignature = "";
    vmGalleryHistoryTruncated = false;
    vmGalleryLoadGeneration += 1;
    const range = document.getElementById("vm-gallery-range");
    const latest = document.getElementById("vm-gallery-latest");
    const state = document.getElementById("vm-gallery-state");
    const revision = document.getElementById("vm-gallery-revision");
    const items = document.getElementById("vm-gallery-items");
    if (range) { range.min = "0"; range.max = "0"; range.value = "0"; range.disabled = true; }
    if (latest) latest.disabled = true;
    if (state) state.textContent = "no published revisions";
    if (revision) revision.textContent = runID ? "loading immutable history…" : "select a native run";
    if (items) items.innerHTML = `<div class="empty">${runID ? "loading published eval galleries…" : "select a native run"}</div>`;
  }

  function resetVMProfiles(runID = "") {
    vmProfileSignature = "";
    vmProfiles = [];
    vmProfileBaseline = "";
    const state = document.getElementById("vm-profile-state");
    const items = document.getElementById("vm-profile-items");
    if (state) state.textContent = "no published traces";
    if (items) items.innerHTML = `<div class="empty">${runID ? "loading verified trace summaries…" : "select a native run"}</div>`;
  }

  function vmDurationUS(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0) return "—";
    if (numeric >= 1e6) return `${(numeric / 1e6).toFixed(2)} s`;
    if (numeric >= 1e3) return `${(numeric / 1e3).toFixed(2)} ms`;
    return `${numeric.toFixed(1)} μs`;
  }

  function vmBytes(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let scaled = numeric;
    let unit = 0;
    while (scaled >= 1024 && unit < units.length - 1) {
      scaled /= 1024;
      unit += 1;
    }
    return `${scaled.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
  }

  function vmProfileHasTimingSummary(profile) {
    const summary = profile?.summary || {};
    return ["accelerator_launch_count", "captured_step_wall_time_us"]
      .every((field) => Number.isFinite(Number(summary[field])) && Number(summary[field]) >= 0) &&
      Number(profile?.capture_steps) > 0;
  }

  function vmProfilesComparable(profile, baseline) {
    if (!vmProfileHasTimingSummary(profile) || !vmProfileHasTimingSummary(baseline)) return false;
    return profile.node_id === baseline.node_id &&
      profile.backend === baseline.backend &&
      Number(profile.capture_steps) === Number(baseline.capture_steps) &&
      Number(profile.skip_steps) === Number(baseline.skip_steps) &&
      Number(profile.warmup_steps) === Number(baseline.warmup_steps) &&
      JSON.stringify(profile.activities || []) === JSON.stringify(baseline.activities || []) &&
      JSON.stringify(profile.options || {}) === JSON.stringify(baseline.options || {}) &&
      profile.execution_phases?.overlaps_warmup === baseline.execution_phases?.overlaps_warmup;
  }

  function vmProfilePhaseSummary(profile) {
    const phases = profile?.execution_phases;
    if (!phases) return "";
    const declared = (field) => Object.prototype.hasOwnProperty.call(phases, field);
    const compile = declared("compile_enabled") ? `compile ${phases.compile_enabled ? "on" : "off"}` : "compile undeclared";
    let warmup = "warmup undeclared";
    if (declared("warmup_enabled")) {
      warmup = phases.warmup_enabled ?
        (declared("warmup_steps_declared") ? `warmup ${Number(phases.warmup_steps_declared).toLocaleString()} steps` : "warmup steps unknown") :
        "warmup off";
    }
    let qualify = "qualify undeclared";
    if (declared("qualify_enabled")) {
      qualify = phases.qualify_enabled ?
        (declared("qualify_steps") ? `qualify ${Number(phases.qualify_steps).toLocaleString()} steps` : "qualify steps unknown") :
        "qualify off";
    }
    return `${compile} · ${warmup} · ${qualify}`;
  }

  function vmSignedPercent(current, baseline) {
    current = Number(current);
    baseline = Number(baseline);
    if (!Number.isFinite(current) || !Number.isFinite(baseline) || baseline === 0) return "—";
    const delta = ((current / baseline) - 1) * 100;
    return `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`;
  }

  function vmSignedPoints(current, baseline) {
    const delta = (Number(current) - Number(baseline)) * 100;
    if (!Number.isFinite(delta)) return "—";
    return `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`;
  }

  function renderVMProfiles(profiles) {
    const state = document.getElementById("vm-profile-state");
    const target = document.getElementById("vm-profile-items");
    const baseline = profiles.find((profile) => profile.artifact_id === vmProfileBaseline);
    const phaseSummary = vmProfilePhaseSummary(profiles.find((profile) => profile.execution_phases));
    if (state) state.textContent = profiles.length ?
      `${profiles.length} restricted trace${profiles.length === 1 ? "" : "s"} · ${phaseSummary ? `${phaseSummary} · ` : ""}${baseline ? "baseline selected" : "choose a baseline"} · explicit download only` :
      "no published traces";
    if (!target) return;
    target.innerHTML = profiles.map((profile) => {
      const summary = profile.summary || {};
      const phases = profile.execution_phases;
      const operators = Array.isArray(summary.top_operators) ? summary.top_operators.slice(0, 5) : [];
      const windowLabel = `${Number(profile.first_optimizer_step || 0).toLocaleString()}–${Number(profile.last_optimizer_step || 0).toLocaleString()}`;
      const timingSummary = vmProfileHasTimingSummary(profile);
      const facts = [
        ["accelerator op time", vmDurationUS(summary.accelerator_time_us)],
        ["CPU op time", vmDurationUS(summary.cpu_time_us)],
        ["raw trace", vmBytes(profile.trace_size_bytes)],
      ];
      if (timingSummary) {
        facts.splice(0, 0,
          ["captured wall", vmDurationUS(summary.captured_step_wall_time_us)],
          ["GPU launches", `${(Number(summary.accelerator_launch_count) / Math.max(1, Number(profile.capture_steps))).toFixed(1)}/step`],
        );
        if (Number.isFinite(Number(summary.gpu_active_ratio)) && Number(summary.gpu_active_ratio) >= 0) {
          facts.splice(0, 0, ["GPU active", `${(Number(summary.gpu_active_ratio) * 100).toFixed(1)}%`]);
        }
        if (Number.isFinite(Number(summary.allocator_peak_allocated_bytes)) && Number(summary.allocator_peak_allocated_bytes) >= 0) {
          facts.splice(3, 0, ["peak allocated", vmBytes(summary.allocator_peak_allocated_bytes)]);
        }
        if (Number.isFinite(Number(summary.allocator_peak_reserved_bytes)) && Number(summary.allocator_peak_reserved_bytes) >= 0) {
          facts.splice(4, 0, ["peak reserved", vmBytes(summary.allocator_peak_reserved_bytes)]);
        }
        if (Number.isFinite(Number(summary.input_stall_ratio)) && Number(summary.input_stall_ratio) >= 0) {
          facts.splice(3, 0, ["input stall", `${(Number(summary.input_stall_ratio) * 100).toFixed(1)}%`]);
        }
      }
      let comparison = "";
      let phaseQualifier = "";
      if (phases?.overlaps_warmup === true) {
        phaseQualifier = '<div class="vm-profile-phase"><strong>warmup overlap</strong><span>capture includes declared execution warmup; treat metrics as non-steady-state</span></div>';
      } else if (phases && phases.overlaps_warmup === null) {
        phaseQualifier = '<div class="vm-profile-phase"><strong>warmup overlap unknown</strong><span>warmup is enabled without a declared step count; steady-state status cannot be determined</span></div>';
      }
      if (baseline && profile.artifact_id === baseline.artifact_id) {
        comparison = '<div class="vm-profile-comparison"><strong>comparison baseline</strong><span>other compatible windows are normalized against this trace</span></div>';
      } else if (baseline && vmProfilesComparable(profile, baseline)) {
        const baseSummary = baseline.summary || {};
        comparison = `<div class="vm-profile-comparison"><strong>Δ from steps ${Number(baseline.first_optimizer_step).toLocaleString()}–${Number(baseline.last_optimizer_step).toLocaleString()}</strong>` +
          `<span>wall/step ${vmEscape(vmSignedPercent(Number(summary.captured_step_wall_time_us) / Number(profile.capture_steps), Number(baseSummary.captured_step_wall_time_us) / Number(baseline.capture_steps)))}</span>` +
          `<span>launches/step ${vmEscape(vmSignedPercent(Number(summary.accelerator_launch_count) / Number(profile.capture_steps), Number(baseSummary.accelerator_launch_count) / Number(baseline.capture_steps)))}</span>` +
          `${Number.isFinite(Number(summary.gpu_active_ratio)) && Number.isFinite(Number(baseSummary.gpu_active_ratio)) ? `<span>GPU active ${vmEscape(vmSignedPoints(summary.gpu_active_ratio, baseSummary.gpu_active_ratio))}</span>` : ""}` +
          `${Number.isFinite(Number(summary.allocator_peak_allocated_bytes)) && Number.isFinite(Number(baseSummary.allocator_peak_allocated_bytes)) ? `<span>peak allocated ${vmEscape(vmSignedPercent(summary.allocator_peak_allocated_bytes, baseSummary.allocator_peak_allocated_bytes))}</span>` : ""}` +
          `${Number.isFinite(Number(summary.input_stall_ratio)) && Number.isFinite(Number(baseSummary.input_stall_ratio)) ? `<span>input stall ${vmEscape(vmSignedPoints(summary.input_stall_ratio, baseSummary.input_stall_ratio))}</span>` : ""}</div>`;
      } else if (baseline) {
        comparison = '<div class="vm-profile-comparison incompatible"><strong>not comparable</strong><span>node, backend, schedule, activities, profiler options, or warmup status differ</span></div>';
      }
      const downloadLabel = profile.trace_file_name === "trace.sqlite" ? "download restricted Nsight Systems SQLite" :
        (profile.trace_file_name === "trace.ncu-rep" ? "download restricted Nsight Compute report" : "download restricted Chrome trace");
      return `<article class="vm-profile-card${baseline && profile.artifact_id === baseline.artifact_id ? " baseline" : ""}">` +
        `<div class="vm-profile-head"><strong title="${vmEscape(profile.artifact_id)}">${vmEscape(profile.backend)} · steps ${vmEscape(windowLabel)}</strong><span>#${Number(profile.sequence || 0).toLocaleString()}</span></div>` +
        `<div class="vm-profile-facts">${facts.map(([label, value]) =>
          `<div class="vm-profile-fact"><span>${vmEscape(label)}</span><strong>${vmEscape(value)}</strong></div>`
        ).join("")}</div>` +
        `<div class="vm-profile-operators">${operators.map((operator) =>
          `<div class="vm-profile-operator"><span title="${vmEscape(operator.name)}">${vmEscape(operator.name)}</span><span>${Number(operator.calls || 0).toLocaleString()}×</span><span>${vmEscape(vmDurationUS(operator.accelerator_time_us))}</span></div>`
        ).join("") || '<span class="vm-profile-meta">no operator rows in summary</span>'}</div>` +
        phaseQualifier +
        comparison +
        `<div class="vm-profile-meta">${vmEscape(profile.attempt_id)} · ${Number(profile.capture_steps || 0)} captured · ${Number(profile.skip_steps || 0)} skipped · ${Number(profile.warmup_steps || 0)} warmup</div>` +
        `<button type="button" class="vm-profile-baseline" data-artifact="${vmEscape(profile.artifact_id)}" aria-pressed="${baseline && profile.artifact_id === baseline.artifact_id ? "true" : "false"}" ${timingSummary ? "" : "disabled"}>${baseline && profile.artifact_id === baseline.artifact_id ? "selected baseline" : "use as comparison baseline"}</button>` +
        `<a class="vm-profile-download" href="${vmEscape(profile.trace_download_url)}" download>${downloadLabel}</a>` +
      `</article>`;
    }).join("") || '<div class="empty">no bounded GPU traces published yet</div>';
    target.querySelectorAll(".vm-profile-baseline").forEach((button) => {
      button.addEventListener("click", () => {
        vmProfileBaseline = button.dataset.artifact || "";
        renderVMProfiles(vmProfiles);
      });
    });
  }

  async function refreshVMProfiles(force = false) {
    if (!vmSelected) return;
    const runID = vmSelected;
    const selectionGeneration = vmSelectionGeneration;
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(runID)}/profiles`, { cache: "no-store" });
      if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected) return;
      if (!response.ok) throw new Error((await response.text()).trim() || `HTTP ${response.status}`);
      const profiles = await response.json();
      if (!Array.isArray(profiles)) throw new Error("profile response is not an array");
      const signature = JSON.stringify(profiles.map((profile) => [profile.sequence, profile.artifact_id, profile.trace_sha256, profile.execution_phases]));
      if (!force && signature === vmProfileSignature) return;
      vmProfileSignature = signature;
      vmProfiles = profiles;
      if (!profiles.some((profile) => profile.artifact_id === vmProfileBaseline)) {
        vmProfileBaseline = profiles.length > 1 ?
          (profiles.find(vmProfileHasTimingSummary)?.artifact_id || "") : "";
      }
      renderVMProfiles(vmProfiles);
    } catch (error) {
      if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected) return;
      const target = document.getElementById("vm-profile-items");
      if (target) target.innerHTML = `<div class="empty">GPU trace history unavailable · ${vmEscape(error.message)}</div>`;
    }
  }

  function renderVMGalleryControls() {
    const range = document.getElementById("vm-gallery-range");
    const latest = document.getElementById("vm-gallery-latest");
    const state = document.getElementById("vm-gallery-state");
    const revision = document.getElementById("vm-gallery-revision");
    const gallery = vmGalleries[vmGalleryIndex];
    if (range) {
      range.min = "0";
      range.max = String(Math.max(0, vmGalleries.length - 1));
      range.value = String(Math.max(0, vmGalleryIndex));
      range.disabled = vmGalleries.length < 2;
    }
    if (latest) latest.disabled = !vmGalleries.length || vmGalleryIndex === vmGalleries.length - 1;
    if (state) state.textContent = vmGalleries.length ?
      `${vmGalleryIndex + 1}/${vmGalleries.length} retained immutable revisions${vmGalleryHistoryTruncated ? " · older history truncated" : ""}${vmGalleryManual ? " · pinned" : " · following latest"}` :
      (vmGalleryHistoryTruncated ? "no gallery in retained history · older artifacts truncated" : "no published revisions");
    if (revision) revision.textContent = gallery ?
      `${vmEscape(gallery.step_domain)} ${Number(gallery.step || 0).toLocaleString()} · ${vmEscape(gallery.attempt_id)} · #${Number(gallery.sequence || 0).toLocaleString()}` :
      "no declared eval gallery artifact published";
  }

  function vmIsDeclaredGallery(artifact) {
    const logicalName = String(vmObservability?.eval_gallery_artifact || "");
    return Boolean(logicalName) && artifact?.kind === "image_gallery" &&
      artifact?.schema === vmEvalGallerySchema && artifact?.logical_name === logicalName;
  }

  function renderVMGallery(gallery) {
    const target = document.getElementById("vm-gallery-items");
    if (!target) return;
    const items = Array.isArray(gallery.items) ? gallery.items : [];
    target.innerHTML = items.map((item) => {
      const originalURL = item.target_image_url || item.source_image_url || "";
      const originalLabel = item.target_image_url ? "original / target" : "source";
      const attributes = Object.entries(item.sampling_attributes || {})
        .map(([key, value]) => `${key}=${value}`).join(" · ");
      return `<article class="vm-gallery-item">` +
        `<div class="vm-gallery-pair${originalURL ? "" : " single"}">` +
          (originalURL ? `<div class="vm-gallery-image"><img src="${vmEscape(originalURL)}" alt="${vmEscape(originalLabel)}" loading="lazy"><span>${originalLabel}</span></div>` : "") +
          `<div class="vm-gallery-image"><img src="${vmEscape(item.generated_image_url)}" alt="generated eval image" loading="lazy"><span>generated</span></div>` +
        `</div>` +
        `<div class="vm-gallery-copy"><strong title="${vmEscape(item.item_id)}">${vmEscape(item.item_id)}</strong>` +
          `<span title="${vmEscape(item.heldout_item_id)}">held-out ${vmEscape(item.heldout_item_id)}</span>` +
          `<span title="${vmEscape(item.prompt_or_condition_digest)}">condition ${vmEscape(String(item.prompt_or_condition_digest || "").slice(0, 20))} · seed ${Number(item.seed || 0).toLocaleString()}</span>` +
          (attributes ? `<span title="${vmEscape(attributes)}">${vmEscape(attributes)}</span>` : "") +
        `</div></article>`;
    }).join("") || '<div class="empty">published gallery contains no displayable items</div>';
  }

  async function loadVMGallery(index) {
    const summary = vmGalleries[index];
    if (!summary || !vmSelected) return;
    const runID = vmSelected;
    const selectionGeneration = vmSelectionGeneration;
    const loadGeneration = ++vmGalleryLoadGeneration;
    const target = document.getElementById("vm-gallery-items");
    if (target) target.innerHTML = '<div class="empty">verifying published manifest and image identities…</div>';
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(runID)}/galleries/${encodeURIComponent(summary.artifact_id)}`,
        { cache: "no-store" });
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmGalleryLoadGeneration || runID !== vmSelected) return;
      if (!response.ok) throw new Error((await response.text()).trim() || `HTTP ${response.status}`);
      const gallery = await response.json();
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmGalleryLoadGeneration || runID !== vmSelected) return;
      renderVMGallery(gallery);
    } catch (error) {
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmGalleryLoadGeneration || runID !== vmSelected) return;
      if (target) target.innerHTML = `<div class="empty">gallery verification failed · ${vmEscape(error.message)}</div>`;
    }
  }

  async function refreshVMGalleries(force = false) {
    if (!vmSelected) return;
    const runID = vmSelected;
    const selectionGeneration = vmSelectionGeneration;
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(runID)}/galleries`, { cache: "no-store" });
      if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected) return;
      if (!response.ok) throw new Error((await response.text()).trim() || `HTTP ${response.status}`);
      const history = await response.json();
      const galleries = history?.galleries;
      if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected) return;
      if (!Array.isArray(galleries) || typeof history?.history_truncated !== "boolean") {
        throw new Error("gallery history response is malformed");
      }
      const signature = JSON.stringify([
        history.history_truncated,
        galleries.map((gallery) => [gallery.sequence, gallery.artifact_id, gallery.step, gallery.item_count]),
      ]);
      if (!force && signature === vmGallerySignature) return;
      const pinnedArtifact = vmGalleryManual ? vmGalleries[vmGalleryIndex]?.artifact_id : "";
      vmGalleries = galleries;
      vmGallerySignature = signature;
      vmGalleryHistoryTruncated = history.history_truncated;
      vmGalleryIndex = pinnedArtifact ? galleries.findIndex((gallery) => gallery.artifact_id === pinnedArtifact) : galleries.length - 1;
      if (vmGalleryIndex < 0 && galleries.length) {
        vmGalleryIndex = galleries.length - 1;
        vmGalleryManual = false;
      }
      renderVMGalleryControls();
      if (vmGalleryIndex >= 0) await loadVMGallery(vmGalleryIndex);
      else {
        const target = document.getElementById("vm-gallery-items");
        if (target) target.innerHTML = '<div class="empty">no immutable eval galleries published yet</div>';
      }
    } catch (error) {
      if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected) return;
      resetVMGallery(runID);
      const target = document.getElementById("vm-gallery-items");
      if (target) target.innerHTML = `<div class="empty">gallery history unavailable · ${vmEscape(error.message)}</div>`;
    }
  }

  function vmCompactJSON(value, limit = 260) {
    const encoded = JSON.stringify(value ?? {});
    return encoded.length > limit ? `${encoded.slice(0, limit - 1)}…` : encoded;
  }

  function vmControlViewSignature(view) {
    if (!view) return "";
    return JSON.stringify({
      catalog: view.catalog || {},
      effective_values: view.effective_values || {},
      latest_requested_revision: view.latest_requested_revision || 0,
      latest_effective_revision: view.latest_effective_revision || 0,
      commands: view.commands || [],
    });
  }

  function renderVMCommandHistory() {
    const target = document.getElementById("vm-control-history");
    if (!target || !vmControlView) return;
    const commands = Array.isArray(vmControlView.commands) ? vmControlView.commands.slice(0, 12) : [];
    target.innerHTML = commands.map((command) => {
      const status = String(command.status || "unknown").toLowerCase();
      const diagnostics = Array.isArray(command.diagnostics) ? command.diagnostics : [];
      const detail = [
        command.apply_point || "unknown boundary",
        command.effective_step == null ? "" : `step ${Number(command.effective_step).toLocaleString()}`,
        command.reason || "",
      ].filter(Boolean).join(" · ");
      const diagnostic = diagnostics.map((item) => item.message || item.code || String(item)).join(" · ");
      return `<article class="vm-command ${vmEscape(status)}">` +
        `<div class="vm-command-head"><strong>r${Number(command.control_revision || 0)}</strong><span>${vmEscape(status)}</span></div>` +
        `<div class="vm-command-values" title="${vmEscape(vmCompactJSON(command.assignments))}">${vmEscape(vmCompactJSON(command.assignments))}</div>` +
        `<div class="vm-command-meta">${vmEscape(detail || "—")}</div>` +
        (diagnostic ? `<div class="vm-command-diagnostic">${vmEscape(diagnostic)}</div>` : "") +
        `</article>`;
    }).join("") || '<div class="empty">no control commands requested</div>';
  }

  function vmControlInput(name, descriptor, value, disabled) {
    const identity = ` data-vm-control="${vmEscape(name)}"`;
    const blocked = disabled ? " disabled" : "";
    if (Array.isArray(descriptor.values)) {
      return `<select${identity}${blocked}>${descriptor.values.map((option) =>
        `<option value="${vmEscape(option)}"${sameVMValue(option, value) ? " selected" : ""}>${vmEscape(option)}</option>`).join("")}</select>`;
    }
    if (descriptor.type === "boolean") {
      return `<input type="checkbox"${identity}${value ? " checked" : ""}${blocked} />`;
    }
    if (descriptor.type === "number" || descriptor.type === "integer") {
      const step = descriptor.type === "integer" ? "1" : "any";
      const minimum = descriptor.minimum == null ? "" : ` min="${vmEscape(descriptor.minimum)}"`;
      const maximum = descriptor.maximum == null ? "" : ` max="${vmEscape(descriptor.maximum)}"`;
      return `<input type="number" step="${step}"${minimum}${maximum} value="${vmEscape(value)}"${identity}${blocked} />`;
    }
    return `<input type="text" value="${vmEscape(value)}"${identity}${blocked} />`;
  }

  function renderVMControls() {
    const target = document.getElementById("vm-control-catalog");
    const revision = document.getElementById("vm-control-revision");
    const apply = document.getElementById("vm-control-apply");
    if (!target || !vmControlView || !vmSelectedRun) return;
    const catalog = vmControlView.catalog || {};
    const effective = vmControlView.effective_values || {};
    const active = ["running", "paused"].includes(vmSelectedRun.observed_state) &&
      Boolean(vmSelectedRun.current_node_id) && Boolean(vmSelectedRun.current_attempt_id);
    const retryLocked = vmControlRetries.has(vmSelectedRun.run_id);
    const requested = {};
    for (const command of Array.isArray(vmControlView.commands) ? vmControlView.commands : []) {
      if (command.status !== "requested" || !command.assignments) continue;
      for (const [name, value] of Object.entries(command.assignments)) {
        if (!(name in requested)) requested[name] = value;
      }
    }
    target.innerHTML = Object.keys(catalog).sort().map((name) => {
      const descriptor = catalog[name] || {};
      const invalid = vmInvalidControls.has(name);
      const value = invalid ? "" : (name in vmPendingControls ? vmPendingControls[name] : effective[name]);
      const pending = name in vmPendingControls;
      const pauseBlocked = descriptor.requires_pause && vmSelectedRun.observed_state !== "paused";
      const disabled = !vmCommandsEnabled || vmSubmitBusy || vmActionBusy || vmActionRetries.has(vmSelectedRun.run_id) || retryLocked || !active ||
        !descriptor.mutable_after_start || pauseBlocked;
      const meta = [
        descriptor.apply || "unknown boundary",
        descriptor.unit || "",
        descriptor.requires_pause ? (pauseBlocked ? "pause required" : "paused") : "",
        name in requested ? `requested ${String(requested[name])}` : "",
        `effective ${String(effective[name] ?? "—")}`,
      ].filter(Boolean).join(" · ");
      return `<label class="vm-control-row${pending ? " pending" : ""}${invalid ? " invalid" : ""}" title="${vmEscape(descriptor.description || "")}">` +
        `<span class="vm-control-copy"><span class="vm-control-name">${vmEscape(name)}</span><span class="vm-control-meta">${vmEscape(meta)}</span></span>` +
        vmControlInput(name, descriptor, value, disabled) + `</label>`;
    }).join("") || '<div class="empty">this plan declares no live controls</div>';
    if (revision) revision.textContent =
      `requested r${Number(vmControlView.latest_requested_revision || 0)} · effective r${Number(vmControlView.latest_effective_revision || 0)}`;
    if (apply) apply.disabled = !vmCommandsEnabled || vmSubmitBusy || vmActionBusy || vmActionRetries.has(vmSelectedRun.run_id) || (!active && !retryLocked) ||
      !Object.keys(vmPendingControls).length || vmInvalidControls.size > 0;
    const reason = document.getElementById("vm-control-reason");
    if (reason) reason.disabled = vmSubmitBusy || vmActionBusy || vmActionRetries.has(vmSelectedRun.run_id) || retryLocked || (!active && !retryLocked);
    renderVMCommandHistory();
  }

  async function refreshVMControlView(run, force = false) {
    if (!run) return;
    if (vmControlRun !== run.run_id) {
      vmControlRun = run.run_id;
      const retry = vmControlRetries.get(run.run_id);
      vmPendingControls = retry ? cloneVMValues(retry.assignments) : {};
      vmControlIntent = retry?.intent || "";
      vmInvalidControls.clear();
      vmControlView = null;
      const reason = document.getElementById("vm-control-reason");
      if (reason) reason.value = retry?.reason || "";
      const state = document.getElementById("vm-control-state");
      if (state) {
        state.textContent = retry ? "outcome unknown · retry exact request" : "no pending changes";
        state.className = retry ? "sub error" : "sub";
      }
      const target = document.getElementById("vm-control-catalog");
      if (target) target.innerHTML = '<div class="empty">loading declared controls…</div>';
    }
    if (!force && vmControlView &&
        Number(vmControlView.source_event_sequence || 0) === Number(run.last_event_sequence || 0)) {
      return;
    }
    if (vmControlLoadAbort) vmControlLoadAbort.abort();
    const controller = new AbortController();
    vmControlLoadAbort = controller;
    const selectionGeneration = vmSelectionGeneration;
    const loadGeneration = ++vmControlLoadGeneration;
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(run.run_id)}/controls`,
        { cache: "no-store", signal: controller.signal });
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmControlLoadGeneration ||
          vmSelected !== run.run_id || controller.signal.aborted) return;
      if (!response.ok) {
        const target = document.getElementById("vm-control-catalog");
        if (target) target.innerHTML = '<div class="empty">persisted control catalog unavailable</div>';
        return;
      }
      const view = await response.json();
      if (selectionGeneration !== vmSelectionGeneration || loadGeneration !== vmControlLoadGeneration ||
          vmSelected !== run.run_id || controller.signal.aborted) return;
      const viewChanged = vmControlViewSignature(view) !== vmControlViewSignature(vmControlView);
      vmControlView = view;
      vmControlView.source_event_sequence = run.last_event_sequence;
      if (viewChanged) renderVMControls();
    } catch (error) {
      if (error.name === "AbortError" || selectionGeneration !== vmSelectionGeneration ||
          loadGeneration !== vmControlLoadGeneration || vmSelected !== run.run_id) return;
      const target = document.getElementById("vm-control-catalog");
      if (target) target.innerHTML = '<div class="empty">persisted control catalog unavailable</div>';
    } finally {
      if (loadGeneration === vmControlLoadGeneration) vmControlLoadAbort = null;
    }
  }

  function vmInputValue(input, descriptor) {
    if (descriptor.type === "boolean") return input.tagName === "SELECT" ? input.value === "true" : input.checked;
    if (descriptor.type === "integer") return Number(input.value);
    if (descriptor.type === "number") return Number(input.value);
    return input.value;
  }

  async function requestVMControls() {
    const state = document.getElementById("vm-control-state");
    if (vmSubmitBusy || !vmSelectedRun || !vmControlView ||
        !Object.keys(vmPendingControls).length || vmInvalidControls.size) return;
    const runID = vmSelectedRun.run_id;
    const retry = vmControlRetries.get(runID);
    const reason = retry?.reason || document.getElementById("vm-control-reason")?.value.trim() || "";
    if (!retry && !reason) {
      state.textContent = "reason is required";
      state.className = "sub error";
      document.getElementById("vm-control-reason")?.focus();
      return;
    }
    const intent = retry?.intent || vmControlIntent ||
      (globalThis.crypto?.randomUUID?.() ||
        `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    vmControlIntent = intent;
    const assignments = retry ? cloneVMValues(retry.assignments) : cloneVMValues(vmPendingControls);
    const payload = retry?.payload || {
      expected_run_revision: Number(vmSelectedRun.run_revision || 0),
      expected_control_revision: Number(vmControlView.latest_requested_revision || 0),
      idempotency_key: intent,
      reason,
      assignments,
    };
    const submission = {
      runID,
      selectionGeneration: vmSelectionGeneration,
      intent,
      reason,
      assignments,
      payload,
      body: retry?.body || JSON.stringify(payload),
    };
    vmSubmitBusy = true;
    renderVMControls();
    state.textContent = "requesting native revision…";
    state.className = "sub";
    try {
      const response = await fetch(
        `/api/trainvm/runs/${encodeURIComponent(submission.runID)}/controls`, {
          method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
          body: submission.body,
        });
      const text = await response.text();
      let result = {};
      try { result = JSON.parse(text); } catch (_) { /* HTTP error text is shown below. */ }
      if (!response.ok) {
        const ambiguous = response.status >= 500 || response.status === 408;
        if (ambiguous) vmControlRetries.set(submission.runID, submission);
        else vmControlRetries.delete(submission.runID);
        if (submission.selectionGeneration === vmSelectionGeneration && vmSelected === submission.runID) {
          vmControlIntent = ambiguous ? submission.intent : "";
          vmPendingControls = cloneVMValues(submission.assignments);
          state.textContent = ambiguous ?
            `outcome unknown · retry exact request · ${text.trim() || `HTTP ${response.status}`}` :
            result.diagnostics?.[0]?.message || text.trim() || `HTTP ${response.status}`;
          state.className = "sub error";
          if (!ambiguous && vmSelectedRun) await refreshVMControlView(vmSelectedRun, true);
        }
        return;
      }
      vmControlRetries.delete(submission.runID);
      if (submission.selectionGeneration === vmSelectionGeneration && vmSelected === submission.runID) {
        vmPendingControls = {};
        vmControlIntent = "";
        const reasonInput = document.getElementById("vm-control-reason");
        if (reasonInput) reasonInput.value = "";
        state.textContent = `${String(result.status || "requested").toLowerCase()} · control r${Number(result.control_revision || 0)}`;
        state.className = "sub ok";
        if (vmSelectedRun) await refreshVMControlView(vmSelectedRun, true);
        await appendVMTimeline();
      }
    } catch (error) {
      vmControlRetries.set(submission.runID, submission);
      if (submission.selectionGeneration === vmSelectionGeneration && vmSelected === submission.runID) {
        vmControlIntent = submission.intent;
        vmPendingControls = cloneVMValues(submission.assignments);
        state.textContent = `outcome unknown · retry exact request · ${error.message}`;
        state.className = "sub error";
      }
    } finally {
      vmSubmitBusy = false;
      if (vmSelectedRun && vmControlView) renderVMControls();
      renderVMActions();
    }
  }

  async function requestVMAction(button) {
    const state = document.getElementById("vm-action-state");
    if (vmSubmitBusy || vmActionBusy || !vmSelectedRun || !button || vmControlRetries.has(vmSelectedRun.run_id)) return;
    const runID = vmSelectedRun.run_id;
    const retry = vmActionRetries.get(runID);
    const action = button.dataset.vmAction || "";
    const checkpointFirst = button.dataset.checkpointFirst === "true";
    const releaseResources = button.dataset.releaseResources === "true";
    const variant = vmActionVariant(action, checkpointFirst, releaseResources);
    if (retry && retry.variant !== variant) return;
    const reason = retry?.reason || document.getElementById("vm-action-reason")?.value.trim() || "";
    if (!retry && !reason) {
      state.textContent = "reason is required";
      state.className = "sub error";
      document.getElementById("vm-action-reason")?.focus();
      return;
    }
    const timeoutInput = document.getElementById("vm-cancel-timeout");
    const gracefulTimeout = action === "cancel" ? Number(timeoutInput?.value || 0) : 0;
    if (!retry && action === "cancel" && (!Number.isInteger(gracefulTimeout) || gracefulTimeout < 0 || gracefulTimeout > 86400)) {
      state.textContent = "cancel grace must be an integer from 0 to 86400 seconds";
      state.className = "sub error";
      timeoutInput?.focus();
      return;
    }
    const intent = retry?.intent || (globalThis.crypto?.randomUUID?.() ||
      `browser-action-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    const payload = retry?.payload || {
      expected_run_revision: Number(vmSelectedRun.run_revision || 0),
      idempotency_key: intent,
      reason,
      action,
      checkpoint_first: checkpointFirst,
      release_resources: releaseResources,
      graceful_timeout_seconds: gracefulTimeout,
    };
    const submission = retry || {
      runID, selectionGeneration: vmSelectionGeneration, variant,
      label: button.textContent.trim(), intent, reason, payload, body: JSON.stringify(payload),
    };
    vmActionBusy = true;
    renderVMActions();
    if (vmControlView) renderVMControls();
    state.textContent = `requesting ${submission.label}…`;
    state.className = "sub";
    try {
      const response = await fetch(`/api/trainvm/runs/${encodeURIComponent(submission.runID)}/actions`, {
        method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
        body: submission.body,
      });
      const text = await response.text();
      let result = {};
      try { result = JSON.parse(text); } catch (_) { /* HTTP error text is shown below. */ }
      if (!response.ok) {
        const ambiguous = response.status === 408 || response.status >= 500;
        if (ambiguous) vmActionRetries.set(submission.runID, submission);
        else vmActionRetries.delete(submission.runID);
        if (submission.selectionGeneration === vmSelectionGeneration && vmSelected === submission.runID) {
          state.textContent = ambiguous ?
            `outcome unknown · retry exact action · ${text.trim() || `HTTP ${response.status}`}` :
            result.diagnostics?.[0]?.message || text.trim() || `HTTP ${response.status}`;
          state.className = "sub error";
        }
        return;
      }
      vmActionRetries.delete(submission.runID);
      if (submission.selectionGeneration === vmSelectionGeneration && vmSelected === submission.runID) {
        const artifact = result.artifact_id ? ` · ${result.artifact_id}` : "";
        state.textContent = `${String(result.status || "requested").toLowerCase()} · ${result.action || action}${artifact}`;
        state.className = "sub ok";
        const reasonInput = document.getElementById("vm-action-reason");
        if (reasonInput) reasonInput.value = "";
        await appendVMTimeline();
        await refreshTrainVM(true);
      }
    } catch (error) {
      vmActionRetries.set(submission.runID, submission);
      if (submission.selectionGeneration === vmSelectionGeneration && vmSelected === submission.runID) {
        state.textContent = `outcome unknown · retry exact action · ${error.message}`;
        state.className = "sub error";
      }
    } finally {
      vmActionBusy = false;
      renderVMActions();
      if (vmSelectedRun && vmControlView) renderVMControls();
    }
  }

  async function appendVMTimeline() {
    if (!vmSelected) return;
    const runID = vmSelected;
    const selectionGeneration = vmSelectionGeneration;
    const requestedAfter = vmAfter;
    const response = await fetch(
      `/api/trainvm/runs/${encodeURIComponent(runID)}/timeline?after=${requestedAfter}&limit=1000`,
      { cache: "no-store" });
    if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected ||
        requestedAfter !== vmAfter) return;
    if (!response.ok) return;
    const events = await response.json();
    if (selectionGeneration !== vmSelectionGeneration || runID !== vmSelected ||
        requestedAfter !== vmAfter) return;
    if (!Array.isArray(events) || !events.length) return;
    const target = document.getElementById("trainvm-timeline");
    if (!target) return;
    if (vmAfter === 0) target.textContent = "";
    const fragment = document.createDocumentFragment();
    for (const event of events) {
      if (event.run_id !== runID) continue;
      if (event.node_id) vmVisitedNodes.add(event.node_id);
      const row = document.createElement("div");
      row.className = "vm-event";
      row.innerHTML = `<span class="vm-seq">#${Number(event.sequence).toLocaleString()}</span>` +
        `<span class="vm-type" title="${vmEscape(event.event_type)}">${vmEscape(event.event_type)}</span>` +
        `<span class="vm-attempt" title="${vmEscape(event.attempt_id)}">${vmEscape(event.attempt_id || "—")}</span>` +
        `<span class="vm-payload" title="${vmEscape(JSON.stringify(event.payload || {}))}">${vmEscape(JSON.stringify(event.payload || {}))}</span>`;
      fragment.appendChild(row);
      vmAfter = Math.max(vmAfter, Number(event.sequence) || 0);
    }
    target.appendChild(fragment);
    while (target.children.length > 500) target.firstElementChild.remove();
    const cursor = document.getElementById("trainvm-cursor");
    if (cursor) cursor.textContent = `sequence ${vmAfter.toLocaleString()}`;
    target.scrollTop = target.scrollHeight;
    if (vmCompiledPlan) renderVMWorkflowGraph();
  }

  function resetVMTelemetry() {
    vmTelemetryAfter = 0;
    vmTelemetryJournal = "";
    vmLatestHeartbeat = null;
    vmObservability = null;
    vmMetricRenderSignature = null;
    vmArtifactRenderSignature = null;
    vmMetricSeries.clear();
    vmArtifacts.clear();
    vmExecutionPhases.clear();
    vmCheckpointSummaries.clear();
    const metrics = document.getElementById("trainvm-metrics");
    const artifacts = document.getElementById("trainvm-artifacts");
    if (metrics) metrics.innerHTML = '<div class="empty">no metric samples loaded</div>';
    if (artifacts) artifacts.innerHTML = '<div class="empty">no artifacts loaded</div>';
    const metricCursor = document.getElementById("trainvm-metric-cursor");
    const artifactCursor = document.getElementById("trainvm-artifact-cursor");
    if (metricCursor) metricCursor.textContent = "waiting for snapshot";
    if (artifactCursor) artifactCursor.textContent = "waiting for snapshot";
    const state = document.getElementById("trainvm-observability-state");
    if (state) {
      state.className = "vm-observability-state waiting";
      state.dataset.vmSemanticState = "waiting";
      state.setAttribute("aria-live", "polite");
      state.textContent = "no telemetry snapshot";
    }
    const phases = document.getElementById("vm-execution-phases");
    const phaseState = document.getElementById("vm-execution-phases-state");
    if (phases) phases.innerHTML = '<div class="empty">no execution-phase receipts loaded</div>';
    if (phaseState) phaseState.textContent = "no receipts";
  }

  async function refreshVMCheckpointRows() {
    if (!vmSelected) return;
    const selected = vmSelected;
    const response = await fetch(
      `/api/trainvm/runs/${encodeURIComponent(selected)}/checkpoints?limit=250`,
      { cache: "no-store" });
    if (!response.ok || selected !== vmSelected) return;
    const checkpoints = await response.json();
    if (!Array.isArray(checkpoints) || selected !== vmSelected) return;
    vmCheckpointSummaries.clear();
    for (const checkpoint of checkpoints) {
      if (!checkpoint?.artifact_id) continue;
      vmCheckpointSummaries.set(checkpoint.artifact_id, checkpoint);
      if (!vmArtifacts.has(checkpoint.artifact_id)) {
        vmArtifacts.set(checkpoint.artifact_id, {
          artifact_id: checkpoint.artifact_id,
          logical_name: checkpoint.logical_name || "checkpoint",
          kind: "checkpoint", schema: checkpoint.checkpoint_schema || "",
          size_bytes: checkpoint.payload_size_bytes || 0,
          sequence: checkpoint.sequence || 0,
          producer_node_id: checkpoint.node_id || "",
          producer_attempt_id: checkpoint.attempt_id || "",
          parent_artifact_ids: checkpoint.parent_artifact_ids || [],
        });
      }
    }
    while (vmArtifacts.size > 1000) vmArtifacts.delete(vmArtifacts.keys().next().value);
    while (vmCheckpointSummaries.size > 1000) {
      vmCheckpointSummaries.delete(vmCheckpointSummaries.keys().next().value);
    }
    renderVMArtifacts();
  }

  function vmMetricKey(metric) {
    const labels = Object.entries(metric.labels || {}).sort(([left], [right]) => left.localeCompare(right));
    return JSON.stringify([
      metric.name, metric.unit, metric.step_domain, labels,
      metric.node_id || "", metric.attempt_id || "",
    ]);
  }

  function ingestVMMetric(metric) {
    const key = vmMetricKey(metric);
    let series = vmMetricSeries.get(key);
    if (!series) {
      if (vmMetricSeries.size >= vmMetricSeriesLimit) return false;
      series = { key, descriptor: metric, points: [] };
      vmMetricSeries.set(key, series);
    }
    if (series.points.some((point) => Number(point.sequence) === Number(metric.sequence))) return true;
    series.descriptor = metric;
    series.points.push(metric);
    series.points.sort((left, right) => Number(left.sequence) - Number(right.sequence));
    if (series.points.length > 240) series.points.splice(0, series.points.length - 240);
    return true;
  }

  function vmSparkline(points) {
    const numeric = points.filter((point) => typeof point.value === "number" && Number.isFinite(point.value));
    if (numeric.length < 2) return '<div class="vm-metric-no-chart">numeric history pending</div>';
    const width = 360, height = 92, inset = 6;
    const xs = numeric.map((point) => Number(point.step));
    const ys = numeric.map((point) => Number(point.value));
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = Math.max(maxX - minX, 1);
    const spanY = Math.max(maxY - minY, Math.abs(maxY) * 1e-9, 1e-12);
    const coordinates = numeric.map((point) => {
      const x = inset + ((Number(point.step) - minX) / spanX) * (width - 2 * inset);
      const y = height - inset - ((Number(point.value) - minY) / spanY) * (height - 2 * inset);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    return `<svg class="vm-metric-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="metric history from ${vmEscape(minY)} to ${vmEscape(maxY)}">` +
      `<path class="vm-metric-gridline" d="M6 46H354"></path>` +
      `<polyline points="${coordinates}"></polyline></svg>` +
      `<div class="vm-metric-range"><span>${vmEscape(minY.toPrecision(4))}</span><span>${vmEscape(maxY.toPrecision(4))}</span></div>`;
  }

  function vmReducedMetricPoints(series) {
    const aggregation = series.descriptor.aggregation || "last";
    if (aggregation === "histogram") return [];
    const groups = new Map();
    for (const point of series.points) {
      if (typeof point.value !== "number" || !Number.isFinite(point.value)) continue;
      const step = Number(point.step);
      if (!groups.has(step)) groups.set(step, []);
      groups.get(step).push(point);
    }
    return [...groups.entries()].sort(([left], [right]) => left - right).map(([step, points]) => {
      const latest = points.at(-1);
      const values = points.map((point) => Number(point.value));
      let value = values.at(-1);
      if (aggregation === "sum") value = values.reduce((total, item) => total + item, 0);
      if (aggregation === "mean") value = values.reduce((total, item) => total + item, 0) / values.length;
      if (aggregation === "weighted_mean") {
        const weight = points.reduce((total, point) => total + Number(point.sample_weight || 1), 0);
        value = points.reduce((total, point) =>
          total + Number(point.value) * Number(point.sample_weight || 1), 0) / Math.max(weight, 1e-12);
      }
      if (aggregation === "min") value = Math.min(...values);
      if (aggregation === "max") value = Math.max(...values);
      return { ...latest, step, value };
    });
  }

  function renderVMMetricCharts() {
    const target = document.getElementById("trainvm-metrics");
    if (!target) return;
    const series = [...vmMetricSeries.values()].sort((left, right) =>
      String(left.descriptor.name).localeCompare(String(right.descriptor.name)) ||
      left.key.localeCompare(right.key));
    const signature = JSON.stringify(series.map((item) => [
      item.key, item.points.length, Number(item.points.at(-1)?.sequence || 0),
    ]));
    if (signature === vmMetricRenderSignature) return;
    vmMetricRenderSignature = signature;
    if (!series.length) {
      target.innerHTML = '<div class="empty">no declared metric samples in the retained tail</div>';
      return;
    }
    target.innerHTML = series.map((item) => {
      const metric = item.descriptor;
      const reduced = vmReducedMetricPoints(item);
      const latest = reduced.at(-1) || item.points.at(-1) || metric;
      const labels = Object.entries(metric.labels || {}).sort(([left], [right]) => left.localeCompare(right));
      const labelText = labels.map(([key, value]) => `${key}=${value}`).join(" · ");
      return `<article class="vm-metric-card" data-vm-metric-key="${vmEscape(item.key)}">` +
        `<header><strong title="${vmEscape(metric.description || metric.name)}">${vmEscape(metric.name)}</strong>` +
        `<span>${vmEscape(metric.aggregation || "last")}</span></header>` +
        `<div class="vm-metric-latest"><b>${vmEscape(JSON.stringify(latest.value))}</b> ${vmEscape(metric.unit)}</div>` +
        vmSparkline(reduced) +
        `<footer title="${vmEscape(labelText)}">${vmEscape(metric.step_domain)} ${Number(latest.step || 0).toLocaleString()} · ${vmEscape(labelText || metric.attempt_id || "unlabelled")}</footer>` +
        `</article>`;
    }).join("");
  }

  function vmArtifactDepth(artifactID, cache, visiting) {
    if (cache.has(artifactID)) return cache.get(artifactID);
    if (visiting.has(artifactID)) return { depth: 0, cycle: true };
    visiting.add(artifactID);
    const artifact = vmArtifacts.get(artifactID);
    let depth = 0, cycle = false;
    for (const parentID of artifact?.parent_artifact_ids || []) {
      if (!vmArtifacts.has(parentID)) continue;
      const parent = vmArtifactDepth(parentID, cache, visiting);
      depth = Math.max(depth, Math.min(parent.depth + 1, 8));
      cycle ||= parent.cycle;
    }
    visiting.delete(artifactID);
    const result = { depth, cycle };
    cache.set(artifactID, result);
    return result;
  }

  function renderVMArtifacts() {
    const target = document.getElementById("trainvm-artifacts");
    if (!target) return;
    const artifacts = [...vmArtifacts.values()].sort((left, right) =>
      Number(left.sequence) - Number(right.sequence));
    const signature = JSON.stringify(artifacts.map((artifact) => {
      const checkpoint = vmCheckpointSummaries.get(artifact.artifact_id);
      return [
        artifact.artifact_id, Number(artifact.sequence || 0),
        artifact.fingerprint || "", artifact.parent_artifact_ids || [],
        checkpoint?.valid, checkpoint?.validation_error || "",
      ];
    }));
    if (signature === vmArtifactRenderSignature) return;
    vmArtifactRenderSignature = signature;
    if (!artifacts.length) {
      target.innerHTML = '<div class="empty">no immutable artifacts in the retained tail</div>';
      return;
    }
    const depthCache = new Map();
    const externalParents = [...new Set(artifacts.flatMap((artifact) =>
      (artifact.parent_artifact_ids || []).filter((parentID) => !vmArtifacts.has(parentID))))].sort();
    const externalNodes = externalParents.map((artifactID) =>
      `<article class="vm-artifact-node external" style="--vm-indent:0px">` +
      `<div class="vm-artifact-rail" aria-hidden="true"></div><div class="vm-artifact-copy">` +
      `<header><strong title="${vmEscape(artifactID)}">${vmEscape(artifactID)}</strong><span>external parent</span></header>` +
      `<small>outside the retained artifact window</small></div></article>`).join("");
    const focused = target.contains(document.activeElement) ? {
      action: document.activeElement?.dataset?.vmOpenArtifact ||
        document.activeElement?.dataset?.vmArtifactAction || "",
      artifact: document.activeElement?.dataset?.artifact || "",
    } : null;
    target.innerHTML = externalNodes + artifacts.map((artifact) => {
      const lineage = vmArtifactDepth(artifact.artifact_id, depthCache, new Set());
      const checkpoint = vmCheckpointSummaries.get(artifact.artifact_id);
      const parents = Array.isArray(artifact.parent_artifact_ids) ? artifact.parent_artifact_ids : [];
      const fingerprint = String(artifact.fingerprint || "").replace(/^sha256:/i, "").toLowerCase();
      const downloadable = ["sha256", "manifest_sha256"].includes(artifact.fingerprint_algorithm) &&
        /^[0-9a-f]{64}$/.test(fingerprint);
      const contentURL = `/api/trainvm/runs/${encodeURIComponent(vmSelected)}/artifacts/${encodeURIComponent(artifact.artifact_id)}/content?v=${encodeURIComponent(fingerprint)}&s=${encodeURIComponent(Number(artifact.sequence || 0))}`;
      const special = vmIsDeclaredGallery(artifact) ?
        `<button class="btn sm" type="button" data-vm-open-artifact="gallery" data-artifact="${vmEscape(artifact.artifact_id)}">open gallery</button>` :
        artifact.kind === "opaque" && artifact.schema === "trainvm.gpu-trace.v1" ?
          `<button class="btn sm" type="button" data-vm-open-artifact="profile" data-artifact="${vmEscape(artifact.artifact_id)}">open trace</button>` : "";
      const checkpointText = checkpoint?.valid === false ?
        checkpoint.validation_error || "checkpoint manifest failed verification" : checkpoint ?
        `${checkpoint.resume_grade || "checkpoint"} · step ${Number(checkpoint.optimizer_step || 0).toLocaleString()} · ${Number(checkpoint.file_count || 0).toLocaleString()} files` :
        `${Number(artifact.size_bytes || 0).toLocaleString()} B`;
      return `<article class="vm-artifact-node${lineage.cycle || checkpoint?.valid === false ? " invalid" : ""}" style="--vm-indent:${lineage.depth * 12}px" data-vm-checkpoint="${artifact.kind === "checkpoint" ? vmEscape(artifact.artifact_id) : ""}">` +
        `<div class="vm-artifact-rail" aria-hidden="true"></div><div class="vm-artifact-copy">` +
        `<header><strong title="${vmEscape(artifact.logical_name)}">${vmEscape(artifact.logical_name)}</strong><span>${vmEscape(artifact.kind)}</span></header>` +
        `<div>${vmEscape(checkpointText)} · #${Number(artifact.sequence || 0).toLocaleString()}</div>` +
        `<small title="${vmEscape(parents.join(", "))}">${lineage.cycle ? "invalid cyclic lineage" : parents.length ? `from ${vmEscape(parents.join(", "))}` : "lineage root"}</small>` +
        `<footer>${special}${downloadable ? `<a class="btn sm" download href="${vmEscape(contentURL)}" data-vm-artifact-action="download" data-artifact="${vmEscape(artifact.artifact_id)}">download verified file</a>` : ""}</footer>` +
        `</div></article>`;
    }).join("");
    if (focused?.action && focused.artifact) {
      const replacement = [...target.querySelectorAll("[data-artifact]")].find((candidate) =>
        candidate.dataset.artifact === focused.artifact &&
        (candidate.dataset.vmOpenArtifact || candidate.dataset.vmArtifactAction || "") === focused.action);
      replacement?.focus({ preventScroll: true });
    }
  }

  function renderVMObservabilityState(snapshot) {
    const target = document.getElementById("trainvm-observability-state");
    if (!target) return;
    const run = snapshot?.run || vmSelectedRun || {};
    const terminal = new Set(["completed", "cancelled", "failed"]);
    const heartbeatSeconds = Number(vmObservability?.heartbeat_seconds || 0);
    let className = "waiting", text = "waiting for worker assignment";
    if (terminal.has(run.observed_state)) {
      className = "terminal";
      text = `${run.observed_state} · telemetry immutable`;
    } else if (run.current_attempt_id && vmLatestHeartbeat &&
        vmLatestHeartbeat.attempt_id === run.current_attempt_id &&
        vmLatestHeartbeat.node_id === run.current_node_id) {
      const ageSeconds = Math.max(0, (Date.now() * 1e6 - Number(vmLatestHeartbeat.accepted_at_ns || 0)) / 1e9);
      const stale = heartbeatSeconds > 0 && ageSeconds > heartbeatSeconds * 3;
      className = stale ? "stale" : "live";
      text = `${stale ? "stale" : "live"} · ${vmLatestHeartbeat.phase} · heartbeat ${ageSeconds < 10 ? ageSeconds.toFixed(1) : Math.round(ageSeconds)}s ago`;
    } else if (run.current_attempt_id) {
      text = "worker assigned · awaiting first heartbeat";
    }
    const replay = snapshot?.replay_pending ?
      ` · replaying ${Number(snapshot.next_sequence || 0).toLocaleString()}/${Number(snapshot.target_sequence || 0).toLocaleString()}` :
      ` · caught up at ${Number(vmTelemetryAfter || 0).toLocaleString()}`;
    const semanticState = [className, run.observed_state || "", run.current_attempt_id || "",
      vmLatestHeartbeat?.phase || "", Boolean(snapshot?.replay_pending)].join(":");
    const changed = target.dataset.vmSemanticState !== semanticState;
    target.dataset.vmSemanticState = semanticState;
    target.setAttribute("aria-live", changed ? "polite" : "off");
    target.className = `vm-observability-state ${className}`;
    target.textContent = text + replay;
  }

  function renderVMExecutionPhases() {
    const target = document.getElementById("vm-execution-phases");
    const state = document.getElementById("vm-execution-phases-state");
    if (!target || !state) return;
    const receipts = [...vmExecutionPhases.values()]
      .sort((left, right) => Number(right.sequence || 0) - Number(left.sequence || 0))
      .slice(0, 12);
    if (!receipts.length) {
      target.innerHTML = '<div class="empty">no execution-phase receipts loaded</div>';
      state.textContent = "no receipts";
      return;
    }
    const failures = receipts.filter((receipt) => receipt.disposition === "failed").length;
    state.textContent = failures ? `${failures} failed · ${receipts.length} visible` : `${receipts.length} receipted`;
    target.innerHTML = receipts.map((receipt) => {
      const durationMS = Math.max(0,
        (Number(receipt.completed_at_ns || 0) - Number(receipt.started_at_ns || 0)) / 1e6);
      const restored = receipt.state_fingerprint_before === receipt.state_fingerprint_after;
      const requested = Number(receipt.requested_steps || 0);
      const executed = Number(receipt.steps_executed || 0);
      const diagnostic = Array.isArray(receipt.diagnostics) ? receipt.diagnostics[0] : null;
      const disposition = String(receipt.disposition || "unknown");
      return `<article class="vm-execution-phase ${vmEscape(disposition)}">` +
        `<header><strong>${vmEscape(receipt.phase || "phase")}</strong><span>${vmEscape(disposition)}</span></header>` +
        `<div>${executed.toLocaleString()}/${requested.toLocaleString()} steps · ${durationMS < 1000 ? durationMS.toFixed(1) + " ms" : (durationMS / 1000).toFixed(2) + " s"}</div>` +
        `<div class="vm-phase-proof" title="${vmEscape(receipt.request_digest || "")}">${restored ? "state restored" : "state changed after failure"} · ${vmEscape(receipt.node_id || "—")} · ${vmEscape(receipt.attempt_id || "—")}</div>` +
        (diagnostic ? `<div class="vm-phase-error" title="${vmEscape(diagnostic.message || "")}">${vmEscape(diagnostic.code || "phase failure")} · ${vmEscape(diagnostic.message || "")}</div>` : "") +
        `</article>`;
    }).join("");
  }

  function renderVMObservabilityError(message) {
    const target = document.getElementById("trainvm-observability-state");
    if (!target) return;
    target.dataset.vmSemanticState = `error:${message}`;
    target.setAttribute("aria-live", "polite");
    target.className = "vm-observability-state stale";
    target.textContent = message;
  }

  async function appendVMTelemetry() {
    if (!vmSelected) return;
    const selected = vmSelected;
    const coldLoad = vmTelemetryAfter === 0;
    let galleryPublished = false, profilePublished = false, checkpointPublished = false;
    let lastSnapshot = null;
    for (let page = 0; page < 4; page += 1) {
      let response;
      try {
        response = await fetch(
          `/api/trainvm/runs/${encodeURIComponent(selected)}/observability?after=${vmTelemetryAfter}&limit=250`,
          { cache: "no-store" });
      } catch (_) {
        renderVMObservabilityError("telemetry transport unavailable · retrying");
        return;
      }
      if (selected !== vmSelected) return;
      if (!response.ok) {
        renderVMObservabilityError(`telemetry authority rejected snapshot · HTTP ${response.status}`);
        return;
      }
      let snapshot;
      try {
        snapshot = await response.json();
      } catch (_) {
        renderVMObservabilityError("telemetry snapshot was not valid JSON");
        return;
      }
      if (selected !== vmSelected) return;
      if (snapshot?.run?.run_id !== selected) {
        renderVMObservabilityError("telemetry snapshot belongs to a different run");
        return;
      }
      if (vmTelemetryJournal && snapshot.journal_id !== vmTelemetryJournal) {
        resetVMTelemetry();
      }
      if (Number(snapshot.after_sequence || 0) !== vmTelemetryAfter) {
        renderVMObservabilityError("telemetry cursor disagrees with the requested snapshot");
        return;
      }
      vmTelemetryJournal = String(snapshot.journal_id || "");
      vmObservability = snapshot.observability || vmObservability;
      const currentAttempt = String(snapshot.run?.current_attempt_id || "");
      const currentNode = String(snapshot.run?.current_node_id || "");
      if (!vmLatestHeartbeat || vmLatestHeartbeat.attempt_id !== currentAttempt ||
          vmLatestHeartbeat.node_id !== currentNode) {
        vmLatestHeartbeat = null;
      }
      for (const heartbeat of snapshot.heartbeats || []) {
        if (heartbeat.attempt_id === currentAttempt && heartbeat.node_id === currentNode &&
            (!vmLatestHeartbeat || Number(heartbeat.sequence) > Number(vmLatestHeartbeat.sequence))) {
          vmLatestHeartbeat = heartbeat;
        }
      }
      for (const metric of snapshot.metrics || []) {
        if (!ingestVMMetric(metric)) {
          renderVMObservabilityError(`run exceeds the ${vmMetricSeriesLimit}-series live metric bound`);
          return;
        }
      }
      for (const artifact of snapshot.artifacts || []) {
        vmArtifacts.set(artifact.artifact_id, artifact);
        galleryPublished ||= vmIsDeclaredGallery(artifact);
        profilePublished ||= artifact.kind === "opaque" && artifact.schema === "trainvm.gpu-trace.v1";
        checkpointPublished ||= artifact.kind === "checkpoint";
      }
      for (const receipt of snapshot.execution_phases || []) {
        const key = `${receipt.node_id || ""}\u0000${receipt.attempt_id || ""}\u0000${receipt.phase || ""}`;
        const previous = vmExecutionPhases.get(key);
        if (!previous || Number(receipt.sequence || 0) > Number(previous.sequence || 0)) {
          vmExecutionPhases.set(key, receipt);
        }
      }
      while (vmArtifacts.size > 1000) vmArtifacts.delete(vmArtifacts.keys().next().value);
      vmTelemetryAfter = Number(snapshot.next_sequence || vmTelemetryAfter);
      lastSnapshot = snapshot;
      if (!snapshot.replay_pending) break;
    }
    renderVMMetricCharts();
    renderVMArtifacts();
    renderVMExecutionPhases();
    renderVMObservabilityState(lastSnapshot);
    const state = lastSnapshot?.replay_pending ? "replaying" : "caught up";
    const metricCursor = document.getElementById("trainvm-metric-cursor");
    const artifactCursor = document.getElementById("trainvm-artifact-cursor");
    if (metricCursor) metricCursor.textContent = `sequence ${vmTelemetryAfter.toLocaleString()} · ${state}`;
    if (artifactCursor) artifactCursor.textContent = `sequence ${vmTelemetryAfter.toLocaleString()} · ${state}`;
    if (galleryPublished) await refreshVMGalleries(true);
    if (profilePublished) await refreshVMProfiles(true);
    if (checkpointPublished || coldLoad) {
      await refreshVMCheckpointRows();
    }
  }

  async function refreshTrainVM(force = false) {
    const panel = document.getElementById("trainvm-panel");
    if (!panel || !panel.open || vmBusy || (document.hidden && !force)) return;
    vmBusy = true;
    try {
      void refreshVMHostAuthority();
      const response = await fetch("/api/trainvm/runs", { cache: "no-store" });
      if (!response.ok) {
        renderVMObservabilityError(`run authority unavailable · HTTP ${response.status} · retrying`);
        return;
      }
      let payload;
      try {
        payload = await response.json();
      } catch (_) {
        renderVMObservabilityError("run authority returned invalid JSON · retrying");
        return;
      }
      const journalID = String(payload.journal_id || "");
      if (vmJournalID && journalID && journalID !== vmJournalID) {
        vmAfter = 0;
        resetVMTelemetry();
        resetVMGallery(vmSelected);
        resetVMProfiles(vmSelected);
        resetVMWorkflow(vmSelected);
        const timeline = document.getElementById("trainvm-timeline");
        if (timeline) timeline.innerHTML = '<div class="empty">native journal changed · reloading authoritative history…</div>';
      }
      if (journalID) vmJournalID = journalID;
      const commandsEnabled = Boolean(payload.commands_enabled);
      const commandAvailabilityChanged = commandsEnabled !== vmCommandsEnabled;
      vmCommandsEnabled = commandsEnabled;
      const runs = Array.isArray(payload.runs) ? payload.runs : [];
      if (!payload.enabled) {
        renderVMObservabilityError("native TrainVM journal is not attached");
        document.getElementById("trainvm-runs").innerHTML =
          '<div class="empty">native journal not attached · start with -trainvm-db PATH</div>';
        return;
      }
      if (!vmCommandsEnabled) {
        const state = document.getElementById("vm-control-state");
        if (state) state.textContent = "native command socket not attached";
      }
      if (!runs.some((run) => run.run_id === vmSelected)) {
        selectVMRun(runs[0]?.run_id || "");
        vmAfter = 0;
        resetVMTelemetry();
        const timeline = document.getElementById("trainvm-timeline");
        if (timeline) timeline.innerHTML = '<div class="empty">no events loaded</div>';
      }
      const selected = runs.find((run) => run.run_id === vmSelected);
      if (selected && Number(selected.last_event_sequence || 0) < vmAfter) {
        vmAfter = 0;
        const timeline = document.getElementById("trainvm-timeline");
        if (timeline) timeline.textContent = "";
      }
      if (selected && Number(selected.last_event_sequence || 0) < vmTelemetryAfter) {
        resetVMTelemetry();
      }
      const previousSelectedRun = vmSelectedRun;
      const immutableRunIdentityChanged = previousSelectedRun?.run_id === selected?.run_id &&
        (String(previousSelectedRun?.plan_hash || "") !== String(selected?.plan_hash || "") ||
         Number(selected?.run_revision || 0) < Number(previousSelectedRun?.run_revision || 0));
      if (immutableRunIdentityChanged) {
        // TrainVM requires a changed plan to create a new run. If one run ID
        // changes plan hash or regresses revision, fail closed instead of
        // combining histories whose declarations have different meanings.
        vmSelectionGeneration += 1;
        vmAfter = 0;
        resetVMTelemetry();
        resetVMGallery(selected.run_id);
        resetVMProfiles(selected.run_id);
        resetVMWorkflow(selected.run_id);
        resetVMControls(selected.run_id);
        vmSelectedRun = null;
        publishVMRunSelection(null);
        renderVMRunList(runs);
        const timeline = document.getElementById("trainvm-timeline");
        if (timeline) timeline.innerHTML = '<div class="empty">immutable run identity changed · history rejected</div>';
        renderVMObservabilityError("authority changed an immutable run identity · create a new run for a new plan");
        return;
      }
      renderVMRunList(runs);
      const previousObservedState = previousSelectedRun?.run_id === selected?.run_id ?
        previousSelectedRun?.observed_state : "";
      vmSelectedRun = selected || null;
      publishVMRunSelection(vmSelectedRun);
      if (selected) {
        renderVMSummary(selected);
        await Promise.all([refreshVMControlView(selected), refreshVMCompiledPlan(selected)]);
        if ((commandAvailabilityChanged || previousObservedState !== selected.observed_state) &&
            vmSelectedRun && vmControlView) renderVMControls();
        if (commandAvailabilityChanged || previousObservedState !== selected.observed_state) renderVMActions();
      }
      await appendVMTimeline();
      await appendVMTelemetry();
      if (selected && !vmGallerySignature) await refreshVMGalleries(true);
      if (selected && !vmProfileSignature) await refreshVMProfiles(true);
    } catch (_) {
      renderVMObservabilityError("run authority transport unavailable · retrying");
    } finally {
      vmBusy = false;
    }
  }

  const vmPanel = document.getElementById("trainvm-panel");
  if (vmPanel) vmPanel.addEventListener("toggle", () => refreshTrainVM(true));
  document.getElementById("trainvm-artifacts")?.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-vm-open-artifact]");
    if (!button || !vmSelected) return;
    const artifactID = button.dataset.artifact || "";
    if (button.dataset.vmOpenArtifact === "gallery") {
      if (!vmGalleries.some((gallery) => gallery.artifact_id === artifactID)) {
        await refreshVMGalleries(true);
      }
      const index = vmGalleries.findIndex((gallery) => gallery.artifact_id === artifactID);
      if (index >= 0) {
        vmGalleryManual = true;
        vmGalleryIndex = index;
        renderVMGalleryControls();
        await loadVMGallery(index);
        document.getElementById("vm-gallery-title")?.scrollIntoView({ block: "start", behavior: "smooth" });
      }
    } else if (button.dataset.vmOpenArtifact === "profile") {
      await refreshVMProfiles(true);
      vmProfileBaseline = artifactID;
      renderVMProfiles(vmProfiles);
      document.getElementById("vm-profile-title")?.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  });
  window.addEventListener("trainvm-refresh", (event) => {
    const runID = String(event.detail?.runID || "");
    if (runID) {
      selectVMRun(runID);
      vmAfter = 0;
      resetVMTelemetry();
      const timeline = document.getElementById("trainvm-timeline");
      if (timeline) timeline.innerHTML = '<div class="empty">loading timeline…</div>';
    }
    refreshTrainVM(true);
  });
  document.getElementById("vm-control-catalog")?.addEventListener("change", (event) => {
    const input = event.target.closest("[data-vm-control]");
    if (!input || !vmControlView) return;
    const name = input.dataset.vmControl;
    const descriptor = vmControlView.catalog?.[name] || {};
    const value = vmInputValue(input, descriptor);
    const numeric = descriptor.type === "number" || descriptor.type === "integer";
    if (numeric && (!input.value.trim() || !Number.isFinite(value) ||
        (descriptor.type === "integer" && !Number.isInteger(value)) || !input.checkValidity())) {
      delete vmPendingControls[name];
      vmInvalidControls.add(name);
      vmControlIntent = "";
      const state = document.getElementById("vm-control-state");
      if (state) {
        state.textContent = `${name} has an invalid numeric value`;
        state.className = "sub error";
      }
      renderVMControls();
      return;
    }
    vmInvalidControls.delete(name);
    if (sameVMValue(value, vmControlView.effective_values?.[name])) delete vmPendingControls[name];
    else vmPendingControls[name] = value;
    vmControlIntent = "";
    const state = document.getElementById("vm-control-state");
    if (state) {
      const count = Object.keys(vmPendingControls).length;
      state.textContent = count ? `${count} pending · atomic patch` : "no pending changes";
      state.className = "sub";
    }
    renderVMControls();
  });
  document.getElementById("vm-control-apply")?.addEventListener("click", requestVMControls);
  document.querySelector(".vm-lifecycle")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-vm-action]");
    if (button) requestVMAction(button);
  });
  document.getElementById("vm-gallery-range")?.addEventListener("input", (event) => {
    const index = Number(event.target.value);
    if (!Number.isInteger(index) || index < 0 || index >= vmGalleries.length) return;
    vmGalleryManual = index !== vmGalleries.length - 1;
    vmGalleryIndex = index;
    renderVMGalleryControls();
    loadVMGallery(index);
  });
  document.getElementById("vm-gallery-latest")?.addEventListener("click", () => {
    if (!vmGalleries.length) return;
    vmGalleryManual = false;
    vmGalleryIndex = vmGalleries.length - 1;
    renderVMGalleryControls();
    loadVMGallery(vmGalleryIndex);
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-vm-run]");
    if (!button) return;
    selectVMRun(button.dataset.vmRun || "");
    vmAfter = 0;
    resetVMTelemetry();
    const timeline = document.getElementById("trainvm-timeline");
    if (timeline) timeline.innerHTML = '<div class="empty">loading timeline…</div>';
    refreshTrainVM(true);
  });
  setInterval(refreshTrainVM, 1000);

  window.trainboard = { watchActiveRun, openEvalSamples, resetEvalSamples };
  window.addEventListener("resize", () => {
    document.querySelectorAll("article.eval-sample").forEach(layoutEvalBoxOverlay);
    layoutVMWorkflowEdges();
  });

  // SSE remains the efficient primary transport. This tiny selected-run poll
  // is an independent recovery path for the two headline regions: browser tab
  // suspension, a missed nested-signal patch, or an SSE reconnect must never
  // leave eval PPL / best PPL stale until the user switches runs.
  let liveRefreshInFlight = false;
  async function refreshSelectedRun() {
    if (liveRefreshInFlight || document.hidden) return;
    const marker = document.getElementById("active-run");
    const run = marker && marker.dataset.run;
    if (!run) return;
    liveRefreshInFlight = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run)}/live`, {
        cache: "no-store",
        headers: { "Accept": "application/json" },
        signal: controller.signal,
      });
      if (!res.ok) return;
      const data = await res.json();
      // A run switch may have happened while this request was in flight.
      const current = document.getElementById("active-run");
      if (!current || current.dataset.run !== run || data.run !== run) return;
      const header = document.getElementById("run-header");
      const kpis = document.getElementById("run-kpis");
      if (header && data.header_html) header.outerHTML = data.header_html;
      if (kpis && data.kpi_html) kpis.outerHTML = data.kpi_html;
      const version = String(data.version || 0);
      if (current.dataset.v !== version) current.dataset.v = version;
    } catch (_) {
      // The live SSE still owns normal rendering; the next poll retries.
    } finally {
      clearTimeout(timeout);
      liveRefreshInFlight = false;
    }
  }
  setInterval(refreshSelectedRun, 1000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshSelectedRun();
  });

  // ---- keyboard: "/" focuses the run filter, j/k walk the (filtered) run list.
  // Esc (leaderboard close) is handled declaratively on <body> via Datastar.
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
                         t.tagName === "SELECT" || t.isContentEditable);
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") {
      const s = document.getElementById("run-search");
      if (s) { e.preventDefault(); s.focus(); s.select(); }
      return;
    }
    if (e.key === "j" || e.key === "k") {
      const items = Array.from(document.querySelectorAll("#run-list .run-item"))
        .filter((el) => el.style.display !== "none");
      if (!items.length) return;
      let idx = items.findIndex((el) => el.classList.contains("active"));
      if (idx < 0) idx = e.key === "j" ? 0 : items.length - 1;
      else idx = Math.min(items.length - 1, Math.max(0, idx + (e.key === "j" ? 1 : -1)));
      items[idx].click();
      items[idx].scrollIntoView({ block: "nearest" });
      e.preventDefault();
    }
  });

  // ---- heartbeat driver for state pulses (converting cell, stalling dot, …).
  // Deliberately JS-driven rather than @keyframes: OS/browser "disable
  // animations" settings can suppress CSS animations wholesale, and these
  // pulses carry live state. Writes a 0..1 sine to --pulse only while a
  // pulse-carrying element is actually on the page.
  const PULSE_SEL = ".conv-cell.converting, .dot.stalling, .alert.critical";
  setInterval(() => {
    const root = document.documentElement;
    if (!document.querySelector(PULSE_SEL)) {
      if (root.style.getPropertyValue("--pulse") !== "") root.style.removeProperty("--pulse");
      return;
    }
    const t = (Date.now() % 2600) / 2600;
    const s = 0.5 - 0.5 * Math.cos(2 * Math.PI * t);
    root.style.setProperty("--pulse", s.toFixed(3));
  }, 90);

  // ---- section jump chips in the sticky toolbar (data-jump="#panel-id").
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-jump]");
    if (!btn) return;
    const el = document.querySelector(btn.dataset.jump);
    if (!el) return;
    // jumping to a collapsed panel opens it (which also triggers its lazy load)
    if (el.tagName === "DETAILS" && !el.open) el.open = true;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
