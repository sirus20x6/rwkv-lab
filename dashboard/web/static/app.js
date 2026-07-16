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
    fire();
  }

  // Exposed for pixi-glue.js: subscribe to (run, version) changes that Datastar
  // publishes by morphing the hidden #active-run node every stream tick.
  let evalSampleRequest = 0;
  function resetEvalSamples(run) {
    evalSampleRequest++;
    const title = document.getElementById("eval-inline-title");
    const meta = document.getElementById("eval-inline-meta");
    const body = document.getElementById("eval-inline-body");
    if (title) title.textContent = "eval captions";
    if (meta) meta.textContent = run
      ? `${run} · waiting for a qualitative eval snapshot`
      : "waiting for a qualitative eval snapshot";
    if (body) body.innerHTML = '<div class="empty">waiting for a caption snapshot…</div>';
  }
  async function openEvalSamples(run, step, ppl, attempt = 0) {
    const request = ++evalSampleRequest;
    const panel = document.getElementById("eval-sample-inline");
    const title = document.getElementById("eval-inline-title");
    const meta = document.getElementById("eval-inline-meta");
    const body = document.getElementById("eval-inline-body");
    if (!panel || !body) return;
    title.textContent = "eval captions · step " + Number(step).toLocaleString();
    meta.textContent = run + (Number.isFinite(ppl) ? " · ppl " + Number(ppl).toFixed(3) : "");
    body.innerHTML = '<div class="empty">loading qualitative snapshot…</div>';
    const url = `/api/runs/${encodeURIComponent(run)}/eval-samples/${encodeURIComponent(step)}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const res = await fetch(url, { signal: controller.signal, cache: "no-store" });
      if (res.status === 404 && attempt < 5) {
        body.innerHTML = '<div class="empty">caption snapshot is starting…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl, attempt + 1);
        }, 2000);
        return;
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
      if (artifactStep !== expectedStep || pplMismatch) {
        meta.textContent = `${run} · ppl ${expectedPPL.toFixed(3)} · replacing stale same-step snapshot`;
        body.innerHTML = '<div class="empty">waiting for this eval generation’s captions…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl);
        }, 2000);
        return;
      }
      const pending = data.complete === false;
      meta.textContent = `${run} · ppl ${Number(data.ppl).toFixed(3)} · ${data.decoding || "greedy"} decoding` +
        (pending ? ` · generating ${Number(data.generation_steps || 0)}/${Number(data.max_new || 0)}` : "");
      body.innerHTML = "";
      for (const item of (data.items || [])) {
        const card = document.createElement("article"); card.className = "eval-sample";
        const img = document.createElement("img"); img.src = item.image_url; img.loading = "lazy"; img.alt = "held-out eval image";
        const copy = document.createElement("div"); copy.className = "eval-sample-copy";
        if (item.prompt) {
          const promptH = document.createElement("h3"); promptH.textContent = "task prompt";
          const prompt = document.createElement("p"); prompt.className = "prompt"; prompt.textContent = item.prompt;
          copy.append(promptH, prompt);
        }
        const genH = document.createElement("h3"); genH.textContent = `model caption · ${item.tokens} tokens${item.stopped_at_eod ? " · EOD" : " · capped"}`;
        const gen = document.createElement("p"); gen.className = "generated"; gen.textContent = item.caption || "(empty caption)";
        const refH = document.createElement("h3"); refH.textContent = `reference · ${item.source || "unknown"}`;
        const ref = document.createElement("p"); ref.className = "reference"; ref.textContent = item.reference || "";
        copy.append(genH, gen, refH, ref); card.append(img, copy); body.append(card);
      }
      if (!(data.items || []).length) body.innerHTML = '<div class="empty">snapshot contains no images</div>';
      // A scalar eval is published before the expensive greedy captions. Poll
      // only while that same card request remains current; switching run/eval
      // invalidates the request and prevents a stale response from taking over.
      if (pending) {
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl);
        }, 2000);
      }
    } catch (err) {
      if (request !== evalSampleRequest) return;
      if (err && err.name === "AbortError") {
        body.innerHTML = '<div class="empty">snapshot request timed out; retrying…</div>';
        setTimeout(() => {
          if (request === evalSampleRequest) openEvalSamples(run, step, ppl, attempt);
        }, 2000);
        return;
      }
      body.innerHTML = "";
      const msg = document.createElement("div"); msg.className = "empty"; msg.textContent = err.message || String(err); body.append(msg);
    } finally {
      clearTimeout(timeout);
    }
  }
  window.trainboard = { watchActiveRun, openEvalSamples, resetEvalSamples };

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
  const PULSE_SEL = ".conv-cell.converting, .dot.stalling, .queue-item.running, .alert.critical";
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
