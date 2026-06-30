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
})();
