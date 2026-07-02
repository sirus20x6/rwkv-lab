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
    const node = document.getElementById("active-run");
    if (!node) return;
    let last = "";
    const fire = () => {
      const key = (node.dataset.run || "") + ":" + (node.dataset.v || "");
      if (key !== last) {
        last = key;
        cb(node.dataset.run || "", parseInt(node.dataset.v || "0", 10));
      }
    };
    new MutationObserver(fire).observe(node, { attributes: true });
    fire();
  }

  // Exposed for pixi-glue.js: subscribe to (run, version) changes that Datastar
  // publishes by morphing the hidden #active-run node every stream tick.
  window.trainboard = { watchActiveRun };

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
