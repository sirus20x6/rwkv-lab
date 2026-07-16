// trainboard Pixi chart layer.
//
// Each Chart owns a PIXI.Application on a <canvas>, holds its own data, and
// redraws on the GPU. The ChartController fetches /api/series for the selected
// run (one union request), distributes columns to charts, and on each
// runVersion bump overlaps the current tip, UPSERTS corrections there, and
// appends newer points — no full re-render of the existing curve. Interactions: wheel = zoom-x,
// drag = pan-x, double-click = reset, hover = crosshair + readout.
//
// Pixi v8 API notes (see STACK.md): new Application(); await app.init({...});
// Graphics build-then-fill/stroke ({width,color,alpha}); Text({text,style}).

(function () {
  "use strict";
  if (!window.PIXI) { console.error("[trainboard] PIXI not loaded"); return; }

  // Okabe-Ito colorblind-safe palette (gray swapped in for black on the dark theme).
  const OI = {
    orange: 0xe69f00, sky: 0x56b4e9, green: 0x009e73, yellow: 0xf0e442,
    blue: 0x0072b2, vermillion: 0xd55e00, purple: 0xcc79a7, gray: 0x999999,
  };
  const COL = {
    loss: OI.sky, evalLoss: OI.vermillion, ppl: OI.purple, baseline: OI.green,
    tok_per_sec: OI.blue, gnorm: OI.orange, lr: OI.purple,
    lm_ce: OI.sky, block: OI.orange, smt_mem: OI.green, dmt_mem: OI.purple,
    dmt_state_rms: OI.yellow, top1: OI.sky, top5: OI.green,
  };
  // Mirrors the app.css tokens (no build step): AXIS≈--border-hi lifted a step,
  // GRID=--divider, TXT=--text-dim, INK=--bg-panel.
  const AXIS = 0x4a5663, GRID = 0x1b2027, TXT = 0xa8b2be, INK = 0x13161b;
  const PAD = { l: 52, r: 52, t: 14, b: 26 };

  // user smoothing preference + per-chart hidden-series set, persisted to localStorage
  const SMOOTH = (() => { try { return Object.assign({ on: true, alpha: 0.10 }, JSON.parse(localStorage.getItem("tb_smooth") || "{}")); } catch (e) { return { on: true, alpha: 0.10 }; } })();
  function saveSmooth() { try { localStorage.setItem("tb_smooth", JSON.stringify(SMOOTH)); } catch (e) {} }
  // log-y (applied per axis when its whole domain is positive) + time-x mode
  const LOGY = (() => { try { return { on: localStorage.getItem("tb_logy") === "1" }; } catch (e) { return { on: false }; } })();
  function saveLogy() { try { localStorage.setItem("tb_logy", LOGY.on ? "1" : "0"); } catch (e) {} }
  const XM = (() => { try { return { time: localStorage.getItem("tb_xmode") === "time" }; } catch (e) { return { time: false }; } })();
  function saveXmode() { try { localStorage.setItem("tb_xmode", XM.time ? "time" : "step"); } catch (e) {} }

  function fmtElapsed(v) {
    if (v == null || !isFinite(v)) return "—";
    if (v < 0) v = 0;
    if (v < 120) return Math.round(v) + "s";
    if (v < 7200) return Math.round(v / 60) + "m";
    return (v / 3600).toFixed(1) + "h";
  }
  // earliest positive wall-clock ts across series = the run's t0 for elapsed-x
  function firstTs(...seriesList) {
    let t0 = Infinity;
    for (const s of seriesList) {
      if (!s || !s.ts) continue;
      for (const v of s.ts) { if (v > 0) { t0 = Math.min(t0, v); break; } }
    }
    return isFinite(t0) ? t0 : 0;
  }
  // elapsed-seconds x-array for a series (monotonized via running max so restart
  // rows can't send the x-axis backwards). Falls back to steps when wall-clock is
  // absent OR degenerate — rows backfilled in one ingest gulp all share the file
  // mtime, which would collapse the whole run onto one x position.
  function xArrOf(s, t0) {
    if (!s) return null;
    if (s.__hasTs == null || s.__hasTsN !== s.step.length) {
      let lo = Infinity, hi = -Infinity;
      if (s.ts) for (const v of s.ts) { if (v > 0) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      s.__hasTs = isFinite(lo) && (hi - lo) >= 30;
      s.__hasTsN = s.step.length;
    }
    if (!s.__hasTs) return s.step;
    if (s.__ela && s.__elaN === s.step.length && s.__elaT0 === t0) return s.__ela;
    const out = new Array(s.step.length);
    let prev = 0;
    for (let i = 0; i < s.step.length; i++) {
      const t = s.ts ? s.ts[i] : 0;
      if (t > 0) prev = Math.max(prev, t - t0);
      out[i] = prev;
    }
    s.__ela = out; s.__elaN = s.step.length; s.__elaT0 = t0;
    return out;
  }
  function loadHidden(id) { try { return new Set(JSON.parse(localStorage.getItem("tb_hidden_" + id) || "[]")); } catch (e) { return new Set(); } }
  function saveHidden(id, set) { try { localStorage.setItem("tb_hidden_" + id, JSON.stringify([...set])); } catch (e) {} }
  function saveRun(run) {
    try { localStorage.setItem("tb_run", run || ""); } catch (e) {}
    try { if (run) history.replaceState(null, "", "#run=" + encodeURIComponent(run)); } catch (e) {}
  }
  function desiredRun() {
    const m = (location.hash || "").match(/run=([^&]+)/);
    if (m) { try { return decodeURIComponent(m[1]); } catch (e) { return m[1]; } }
    try { return localStorage.getItem("tb_run") || ""; } catch (e) { return ""; }
  }

  function clearKids(c) { for (const ch of c.removeChildren()) ch.destroy(); }
  function fmtNum(v) {
    if (v == null || !isFinite(v)) return "—";
    if (v !== 0 && Math.abs(v) < 1e-4) return v.toExponential(2);
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
    return (+v).toFixed(3);
  }

  // Return a copy of a {step,cols} series with STRICTLY-INCREASING steps: sort by
  // step, drop duplicates/out-of-order (the grok-autopilot re-emits a row at the
  // same step on each restart — without this the x-axis zig-zags back and forth).
  function monotonicSeries(s) {
    if (!s || !Array.isArray(s.step) || !s.step.length) return null;
    const n = s.step.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => s.step[a] - s.step[b]);
    const step = [], ts = [], cols = Object.create(null);
    for (const k in (s.cols || {})) cols[k] = [];
    let last = -Infinity;
    for (const i of idx) {
      const st = s.step[i];
      if (!(st > last)) continue;            // skip <= previous (dup / non-increasing)
      last = st; step.push(st);
      ts.push(s.ts ? (s.ts[i] || 0) : 0);
      for (const k in cols) cols[k].push(s.cols[k][i]);
    }
    return { step, ts, cols };
  }

  function cloneSeries(s) {
    if (!s) return null;
    const cols = Object.create(null);
    for (const k in (s.cols || {})) cols[k] = s.cols[k].slice();
    return { step: s.step.slice(), ts: (s.ts || []).slice(), cols };
  }

  function appendSeries(cur, inc) {
    if (!inc || !inc.step || !inc.step.length) return cur;
    if (!cur) return monotonicSeries(inc);
    let maxStep = cur.step.length ? cur.step[cur.step.length - 1] : -Infinity;
    for (let i = 0; i < inc.step.length; i++) {
      const st = inc.step[i];
      if (st < maxStep) continue;
      if (st === maxStep && cur.step.length) {
        // The log may append a corrected row for its current step. SQLite
        // upserts it, so an append-only client must replace that tip rather
        // than silently keeping the value loaded before the correction.
        const at = cur.step.length - 1;
        if (!cur.ts) cur.ts = new Array(cur.step.length).fill(0);
        cur.ts[at] = inc.ts ? (inc.ts[i] || 0) : 0;
        for (const k in inc.cols) {
          if (!cur.cols[k]) cur.cols[k] = new Array(cur.step.length).fill(null);
          cur.cols[k][at] = inc.cols[k][i];
        }
        continue;
      }
      maxStep = st;
      const oldLength = cur.step.length;
      cur.step.push(st);
      if (!cur.ts) cur.ts = new Array(oldLength).fill(0);
      cur.ts.push(inc.ts ? (inc.ts[i] || 0) : 0);
      for (const k in cur.cols) {
        cur.cols[k].push(Object.prototype.hasOwnProperty.call(inc.cols, k)
          ? inc.cols[k][i] : null);
      }
      for (const k in inc.cols) {
        if (cur.cols[k]) continue;
        cur.cols[k] = new Array(oldLength).fill(null);
        cur.cols[k].push(inc.cols[k][i]);
      }
    }
    return cur;
  }

  // ---- one chart ----
  class Chart {
    // spec: { series:[{src,key,label,color,axis,type,log,dash,horizontal}], xlabel }
    constructor(canvasId, spec) {
      this.canvas = document.getElementById(canvasId);
      this.id = canvasId;
      this.hidden = loadHidden(canvasId);
      this.spec = spec;
      this.timeline = [];
      this._markHits = [];
      this.compare = null;
      this.compareLabel = "";
      this.suppressBest = false;
      this.data = { train: null, eval: null, baseline: null };
      this.view = null;            // {min,max} zoom window over x (step), null = full
      this.scales = null;          // computed in drawStatic, used by overlay
      this.ready = false;
    }

    async init() {
      if (!this.canvas) return;
      this.app = new PIXI.Application();
      await this.app.init({
        canvas: this.canvas, antialias: true, backgroundAlpha: 0,
        resolution: window.devicePixelRatio || 1, autoDensity: true,
        width: this.canvas.clientWidth || 600, height: this.canvas.clientHeight || 280,
        preference: "webgl",
        // render-on-demand: the scene only changes on data ticks / interaction,
        // so a free-running 60fps ticker per canvas would just burn GPU cycles
        // on the box that is training. We call _render() explicitly instead.
        autoStart: false, sharedTicker: false,
      });
      if (this.app.ticker) this.app.ticker.stop();
      this.gStatic = new PIXI.Graphics();
      this.labels = new PIXI.Container();
      this.gOverlay = new PIXI.Graphics();
      this.tip = new PIXI.Container();
      this.app.stage.addChild(this.gStatic, this.labels, this.gOverlay, this.tip);
      this._tpool = new Map();   // pooled label Texts (rasterizing text is the CPU cost)
      this._ephem = [];          // same-frame duplicates that can't share a pooled Text
      this._epoch = 0;
      this._visible = true;      // offscreen charts skip geometry until scrolled to
      this._dirty = false;

      this._wire();
      const ro = new ResizeObserver(() => this._resize());
      ro.observe(this.canvas.parentElement);
      const io = new IntersectionObserver((entries) => {
        for (const en of entries) {
          const was = this._visible;
          this._visible = en.isIntersecting;
          if (this._visible && !was && this._dirty) { this._dirty = false; this.drawStatic(); }
        }
      }, { rootMargin: "120px" });
      io.observe(this.canvas.parentElement);
      this.ready = true;
    }

    _render() { if (this.ready && this.app && this._visible) this.app.render(); }

    // pooled Text: reuse label rasterizations across redraws; a key already used
    // this draw (duplicate value callout) gets a fresh ephemeral instance
    _text(text, color, fontSize, fontWeight) {
      const key = text + "|" + color + "|" + fontSize + "|" + (fontWeight || "");
      let t = this._tpool.get(key);
      if (t && t.__epoch === this._epoch) {
        const e = new PIXI.Text({ text, style: { fill: color, fontSize, fontFamily: "monospace", ...(fontWeight ? { fontWeight } : {}) } });
        this._ephem.push(e);
        return e;
      }
      if (!t) {
        t = new PIXI.Text({ text, style: { fill: color, fontSize, fontFamily: "monospace", ...(fontWeight ? { fontWeight } : {}) } });
        this._tpool.set(key, t);
      }
      t.__epoch = this._epoch;
      return t;
    }

    // active x-array for a source: steps, or monotonized elapsed seconds in time mode
    _xa(src) {
      const s = this.data[src];
      if (!s) return null;
      return XM.time ? xArrOf(s, this._t0 || 0) : s.step;
    }
    // does any series here have a usable wall-clock axis right now?
    _timeUsable() {
      for (const src of ["train", "eval"]) {
        const s = this.data[src];
        if (s && s.step.length && this._xa(src) !== s.step) return true;
      }
      return false;
    }
    // map a step to the active x-unit (and back) via the densest series present
    _stepToX(step) {
      if (!XM.time) return step;
      const src = this.data.train ? "train" : (this.data.eval ? "eval" : null);
      if (!src) return step;
      const s = this.data[src], xs = this._xa(src);
      if (!s.step.length) return step;
      return xs[nearestIdx(s.step, step)];
    }
    _xToStep(xv) {
      if (!XM.time) return Math.round(xv);
      const src = this.data.train ? "train" : (this.data.eval ? "eval" : null);
      if (!src) return Math.round(xv);
      const s = this.data[src], xs = this._xa(src);
      if (!s.step.length) return Math.round(xv);
      return s.step[nearestIdx(xs, xv)];
    }

    _resize() {
      const box = this.canvas.parentElement;
      if (!box) return;
      this.app.renderer.resize(box.clientWidth, box.clientHeight);
      this.drawStatic();
    }

    setData(d) {
      this.suppressBest = !!d.suppress_best;
      this._fullData = {
        train: monotonicSeries(d.train), eval: monotonicSeries(d.eval),
        baseline: d.baseline || null,
      };
      this.data = {
        train: cloneSeries(this._fullData.train), eval: cloneSeries(this._fullData.eval),
        baseline: this._fullData.baseline,
      };
      this._t0 = firstTs(this.data.train, this.data.eval);
      this.view = null;
      this.drawStatic();
    }

    // setWindow swaps in a full-resolution slice for the current zoom (keeps view
    // AND the run's t0, so elapsed-x coordinates stay in the same frame).
    setWindow(d) {
      this.suppressBest = !!d.suppress_best;
      const base = this.data ? this.data.baseline : null;
      this.data = { train: monotonicSeries(d.train), eval: monotonicSeries(d.eval), baseline: (d.baseline || base) || null };
      this.drawStatic();
    }

    restoreFull() {
      if (!this._fullData) { this.view = null; this.drawStatic(); return; }
      const d = this._fullData;
      this.data = { train: cloneSeries(d.train), eval: cloneSeries(d.eval), baseline: d.baseline || null };
      this._t0 = firstTs(this.data.train, this.data.eval);
      this.view = null;
      this.drawStatic();
    }

    _emitView() { if (this.onView && this.view) this.onView(this.view); }

    // Upsert incremental rows while keeping the step axis strictly increasing:
    // replace a corrected current tip and append only genuinely newer steps.
    append(d) {
      this.suppressBest = !!d.suppress_best;
      for (const src of ["train", "eval"]) {
        const inc = d[src];
        if (!inc || !inc.step || !inc.step.length) continue;
        this._fullData[src] = appendSeries(this._fullData[src], inc);
        this.data[src] = appendSeries(this.data[src], inc);
      }
      if (!this._t0) this._t0 = firstTs(this.data.train, this.data.eval);
      this.drawStatic();
    }

    _xDomain() {
      if (this.view) return [this.view.min, this.view.max];
      let lo = Infinity, hi = -Infinity;
      for (const src of ["train", "eval"]) {
        const s = this.data[src];
        const xs = this._xa(src);
        if (s && xs && s.step.length) { lo = Math.min(lo, xs[0]); hi = Math.max(hi, xs[xs.length - 1]); }
      }
      if (!isFinite(lo)) return [0, 1];
      if (lo === hi) return [lo - 1, hi + 1];
      return [lo, hi];
    }

    _yDomain(axis) {
      let lo = Infinity, hi = -Infinity;
      const [xmin, xmax] = this._xDomain();
      for (const sp of this.spec.series) {
        if (sp.axis !== axis || sp.horizontal) continue;
        const s = this.data[sp.src];
        if (!s || !s.cols[sp.key]) continue;
        const col = s.cols[sp.key], steps = this._xa(sp.src);
        // collect in-window finite values (no spread: train cols can be 50k+ long)
        const vals = [];
        for (let i = 0; i < col.length; i++) {
          const v = col[i];
          if (v == null || !isFinite(v)) continue;
          if (steps[i] < xmin || steps[i] > xmax) continue;
          if (sp.log && v <= 0) continue;
          vals.push(v);
        }
        if (!vals.length) continue;
        let vlo, vhi;
        if (sp.robust && vals.length > 40) {
          // percentile fence: a few spike steps must not crush the axis around the
          // real signal (ports v1 trimSpikes — same intent, simpler quantile form).
          vals.sort((a, b) => a - b);
          vlo = vals[Math.floor(vals.length * 0.012)];
          vhi = vals[Math.ceil(vals.length * 0.988) - 1];
        } else {
          vlo = Infinity; vhi = -Infinity;
          for (const v of vals) { if (v < vlo) vlo = v; if (v > vhi) vhi = v; }
        }
        lo = Math.min(lo, vlo); hi = Math.max(hi, vhi);
      }
      // include baseline horizontal lines on this axis
      for (const sp of this.spec.series) {
        if (sp.axis === axis && sp.horizontal && this.data.baseline) {
          const v = this.data.baseline[sp.key];
          if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        }
      }
      if (!isFinite(lo)) return null;
      if (lo === hi) { lo -= 1; hi += 1; }
      const pad = (hi - lo) * 0.08;
      return [lo - pad, hi + pad];
    }

    _mkScale(axis) {
      const dom = this._yDomain(axis);
      if (!dom) return null;
      const W = this.app.screen.width, H = this.app.screen.height;
      const [lo, hi] = dom;
      // per-series log flags always apply; the global log-y toggle applies to any
      // axis whose whole visible domain is positive (log of accuracies is fine,
      // log of a signed delta is not)
      const log = this.spec.series.some(s => s.axis === axis && s.log) || (LOGY.on && lo > 0);
      const y0 = H - PAD.b, y1 = PAD.t;
      if (log) {
        const llo = Math.log10(Math.max(lo, 1e-9)), lhi = Math.log10(Math.max(hi, 1e-8));
        return { lo, hi, log: true, px: v => y0 + (y1 - y0) * (Math.log10(Math.max(v, 1e-9)) - llo) / (lhi - llo || 1) };
      }
      return { lo, hi, log: false, px: v => y0 + (y1 - y0) * (v - lo) / (hi - lo || 1) };
    }

    // fixed 0..1 axis for normalized (custom-metric) charts
    _unitScale() {
      const H = this.app.screen.height, y0 = H - PAD.b, y1 = PAD.t;
      return { lo: 0, hi: 1, log: false, px: v => y0 + (y1 - y0) * v };
    }

    // per-series min-max scale over the visible window (lets disparate-scale
    // metrics share one chart); hover still reports the raw value.
    _normScale(col, xmin, xmax, steps) {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < col.length; i++) {
        const st = steps[i]; if (st < xmin || st > xmax) continue;
        const v = col[i]; if (v == null || !isFinite(v)) continue;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (hi <= lo) hi = lo + (Math.abs(lo) || 1) * 1e-3 + 1e-9;
      const H = this.app.screen.height, y0 = H - PAD.b, y1 = PAD.t;
      const pad = (hi - lo) * 0.06;
      lo -= pad; hi += pad;
      return { lo, hi, log: false, px: v => y0 + (y1 - y0) * (v - lo) / (hi - lo || 1) };
    }

    drawStatic() {
      if (!this.ready || !this.app) return;
      if (!this._visible) { this._dirty = true; return; } // offscreen: skip the work
      const W = this.app.screen.width, H = this.app.screen.height;
      // new label epoch: detach (don't destroy) pooled texts, drop last draw's dupes
      this._epoch++;
      for (const t of this._ephem) t.destroy();
      this._ephem = [];
      this.labels.removeChildren();
      const g = this.gStatic; g.clear();
      clearKids(this.tip); this.gOverlay.clear();

      const [xmin, xmax] = this._xDomain();
      const xpx = step => PAD.l + (W - PAD.l - PAD.r) * (step - xmin) / (xmax - xmin || 1);
      // time-x is only "on" for this chart if some series actually has usable ts
      this._timeLabels = XM.time && this._timeUsable();
      const norm = !!this.spec.normalize;
      this._normScales = {};
      const sy = norm ? this._unitScale() : this._mkScale("y");
      const sy1 = norm ? null : this._mkScale("y1");
      this.scales = { xmin, xmax, xpx, sy, sy1, norm };

      // grid + y ticks (left axis drives the gridlines)
      const ticks = 5;
      if (sy) {
        for (let i = 0; i <= ticks; i++) {
          const t = i / ticks;
          const val = sy.log ? Math.pow(10, Math.log10(sy.lo) + t * (Math.log10(sy.hi) - Math.log10(sy.lo))) : sy.lo + t * (sy.hi - sy.lo);
          const y = sy.px(val);
          g.moveTo(PAD.l, y).lineTo(W - PAD.r, y).stroke({ width: 1, color: GRID });
          this._label(fmtNum(val), PAD.l - 6, y, "right", TXT);
        }
      }
      if (sy1) {
        for (let i = 0; i <= ticks; i++) {
          const t = i / ticks;
          const val = sy1.log ? Math.pow(10, Math.log10(sy1.lo) + t * (Math.log10(sy1.hi) - Math.log10(sy1.lo))) : sy1.lo + t * (sy1.hi - sy1.lo);
          this._label(fmtNum(val), W - PAD.r + 6, sy1.px(val), "left", TXT);
        }
      }
      // x ticks — clamped inside the plot span so the first/last labels don't
      // collide with the bottom-most left/right axis labels
      for (let i = 0; i <= 4; i++) {
        const xv = xmin + (i / 4) * (xmax - xmin);
        const x = xpx(xv);
        const txt = this._timeLabels ? fmtElapsed(xv) : Math.round(xv).toLocaleString();
        this._label(txt, x, H - PAD.b + 4, "center", TXT, true,
          [PAD.l - 2, W - PAD.r + 2]);
      }
      // axis frame
      g.moveTo(PAD.l, PAD.t).lineTo(PAD.l, H - PAD.b).lineTo(W - PAD.r, H - PAD.b).stroke({ width: 1, color: AXIS, alpha: 0.6 });

      // timeline markers (checkpoints / alerts / controls / actions), behind the curves
      this._drawMarkers(W, H, xpx, xmin, xmax);

      // series
      for (const sp of this.spec.series) {
        if (this.hidden.has(sp.label)) continue;
        let scale = sp.axis === "y1" ? sy1 : sy;
        if (sp.horizontal) { if (scale) this._drawBaseline(sp, scale, xpx, xmin, xmax); continue; }
        const s = this.data[sp.src];
        if (!s || !s.cols[sp.key]) continue;
        const xs = this._xa(sp.src);
        if (norm) { scale = this._normScale(s.cols[sp.key], xmin, xmax, xs); this._normScales[sp.key] = scale; }
        if (!scale) continue;
        if (sp.type === "scatter") this._drawScatter(s, sp, scale, xpx, xs);
        else this._drawLine(s, sp, scale, xpx, xs);
      }
      // cross-run compare overlay (run B as a dimmed EMA trend on the same axes)
      if (this.compare && !norm) this._drawCompare(sy, sy1, xpx, xmin, xmax);
      // value callout labels (sparse series only — drawn on top of the curves)
      for (const sp of this.spec.series) {
        if (!sp.labels || sp.horizontal || this.hidden.has(sp.label)) continue;
        const scale = sp.axis === "y1" ? sy1 : sy;
        const s = this.data[sp.src];
        if (scale && s && s.cols[sp.key]) this._drawLabels(s, sp, scale, xpx);
      }
      // best-point markers (★ + ring at the series' best value, e.g. best eval ppl)
      if (!norm) this._drawBest(sy, sy1, xpx, xmin, xmax);
      // legend
      this._legend();
      // evict pooled labels that no draw is using anymore
      if (this._tpool.size > 600) {
        for (const [k, t] of this._tpool) {
          if (t.__epoch !== this._epoch) { t.destroy(); this._tpool.delete(k); }
        }
      }
      this._render();
    }

    // ★ at the best (min) point of any series flagged best:"min" — ties the
    // curve to the "best ppl" the KPI strip reports
    _drawBest(sy, sy1, xpx, xmin, xmax) {
      if (this.suppressBest) return;
      for (const sp of this.spec.series) {
        if (sp.best !== "min" || sp.horizontal || this.hidden.has(sp.label)) continue;
        const s = this.data[sp.src];
        if (!s || !s.cols[sp.key]) continue;
        const scale = sp.axis === "y1" ? sy1 : sy;
        if (!scale) continue;
        const col = s.cols[sp.key], xs = this._xa(sp.src);
        let bi = -1;
        for (let i = 0; i < col.length; i++) {
          const v = col[i];
          if (v == null || !isFinite(v)) continue;
          if (bi < 0 || v < col[bi]) bi = i;
        }
        if (bi < 0 || xs[bi] < xmin || xs[bi] > xmax) continue;
        const x = xpx(xs[bi]), y = scale.px(col[bi]);
        this.gStatic.circle(x, y, 7.5).stroke({ width: 2, color: 0x3fd07a, alpha: 0.95 });
        const star = this._text("★", 0x3fd07a, 16);
        star.x = x + 9; star.y = y - 20;
        this.labels.addChild(star);
      }
    }

    setTimeline(evs) { this.timeline = Array.isArray(evs) ? evs : []; this.drawStatic(); }

    setCompare(d, label) {
      this.compare = d ? { train: monotonicSeries(d.train), eval: monotonicSeries(d.eval) } : null;
      // compare run gets its own t0, so in time mode both runs align at their start
      this._t0c = this.compare ? firstTs(this.compare.train, this.compare.eval) : 0;
      this.compareLabel = d ? (label || "") : "";
      this.drawStatic();
    }

    _drawCompare(sy, sy1, xpx, xmin, xmax) {
      const g = this.gStatic;
      for (const sp of this.spec.series) {
        if (sp.horizontal || this.hidden.has(sp.label)) continue;
        const scale = sp.axis === "y1" ? sy1 : sy;
        if (!scale) continue;
        const s = this.compare[sp.src];
        const col = s && s.cols[sp.key];
        if (!col || !s.step.length) continue;
        const xs = XM.time ? xArrOf(s, this._t0c || 0) : s.step;
        const sm = emaCol(col, 0.12);                 // clean trend so B reads as a faint underlay
        let pen = false;
        for (let i = 0; i < s.step.length; i++) {
          const st = xs[i];
          if (st < xmin || st > xmax) { pen = false; continue; }   // clip to the visible window
          const v = sm[i];
          if (v == null || !isFinite(v) || (sp.log && v <= 0)) { pen = false; continue; }
          const x = xpx(st), y = scale.px(v);
          if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
        }
        g.stroke({ width: 1.4, color: sp.color, alpha: 0.4 });
      }
      if (this.compareLabel) {
        const W = this.app.screen.width;
        const t = this._text("dim = " + this.compareLabel, TXT, 14.5);
        t.x = Math.max(PAD.l, W - PAD.r - t.width); t.y = 1;
        this.labels.addChild(t);
      }
    }

    _drawMarkers(W, H, xpx, xmin, xmax) {
      this._markHits = [];
      const evs = this.timeline;
      if (!evs || !evs.length) return;
      const g = this.gStatic;
      for (const e of evs) {
        const st = this._stepToX(e.step || 0);
        if (st < xmin || st > xmax) continue;
        const x = xpx(st), c = markColor(e);
        for (let yy = PAD.t; yy < H - PAD.b; yy += 9) {
          g.moveTo(x, yy).lineTo(x, Math.min(yy + 4, H - PAD.b)).stroke({ width: 1, color: c, alpha: 0.3 });
        }
        g.moveTo(x - 4, PAD.t - 1).lineTo(x + 4, PAD.t - 1).lineTo(x, PAD.t + 5).fill({ color: c, alpha: 0.95 });
        this._markHits.push({ x, e, color: c });
      }
    }

    _drawLine(s, sp, scale, xpx, xs) {
      const g = this.gStatic, col = s.cols[sp.key], steps = xs || s.step;
      const trace = (valOf, width, alpha) => {
        let pen = false;
        for (let i = 0; i < col.length; i++) {
          const v = valOf(i);
          if (v == null || !isFinite(v) || (sp.log && v <= 0)) { pen = false; continue; }
          const x = xpx(steps[i]), y = scale.px(v);
          if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
        }
        g.stroke({ width, color: sp.color, alpha });
      };
      if (sp.smooth && SMOOTH.on) {
        // raw trace faint behind the EMA trend so per-step noise is visible but quiet
        trace(i => col[i], 0.8, 0.10);
        const ema = emaCol(col, SMOOTH.alpha);
        trace(i => ema[i], sp.width || 2.0, 1.0);
      } else {
        // Pixi v8 stroke has no native dash; render "dashed" specs lighter/thinner
        // so secondary series (top-5, lr) stay visually distinct from primaries.
        const width = sp.dash ? 1 : (sp.width || 1.6);
        const alpha = sp.dash ? 0.5 : 0.95;
        trace(i => col[i], width, alpha);
      }
      if (sp.points) {
        for (let i = 0; i < col.length; i++) {
          const v = col[i];
          if (v == null || !isFinite(v)) continue;
          g.circle(xpx(steps[i]), scale.px(v), sp.r || 2.6).fill({ color: sp.color });
        }
      }
    }

    _drawScatter(s, sp, scale, xpx, xs) {
      const g = this.gStatic, col = s.cols[sp.key], steps = xs || s.step;
      for (let i = 0; i < col.length; i++) {
        const v = col[i];
        if (v == null || !isFinite(v)) continue;
        g.circle(xpx(steps[i]), scale.px(v), sp.r || 3).fill({ color: sp.color });
      }
    }

    _drawBaseline(sp, scale, xpx, xmin, xmax) {
      if (!this.data.baseline) return;
      const v = this.data.baseline[sp.key];
      if (v == null || !isFinite(v)) return;
      const y = scale.px(v);
      this.gStatic.moveTo(PAD.l, y).lineTo(this.app.screen.width - PAD.r, y)
        .stroke({ width: 1.4, color: sp.color, alpha: 0.7 });
    }

    // Value callouts with leader lines on a sparse series (e.g. eval ppl). Sampled
    // to ~maxLabels, alternating above/below the point, clamped into the plot.
    _drawLabels(s, sp, scale, xpx) {
      const col = s.cols[sp.key], steps = this._xa(sp.src) || s.step, g = this.gStatic;
      const W = this.app.screen.width, H = this.app.screen.height;
      const pts = [];
      for (let i = 0; i < col.length; i++) {
        const v = col[i];
        if (v == null || !isFinite(v)) continue;
        const x = xpx(steps[i]);
        if (x < PAD.l || x > W - PAD.r) continue;
        pts.push({ x, y: scale.px(v), v });
      }
      if (!pts.length) return;
      const maxN = sp.maxLabels || 22;
      const stride = pts.length > maxN ? Math.ceil(pts.length / maxN) : 1;
      const sel = pts.filter((_, i) => i % stride === 0 || i === pts.length - 1);
      sel.forEach((p, k) => {
        const txt = sp.labelFmt ? sp.labelFmt(p.v) : fmtNum(p.v);
        const t = this._text(txt, sp.color, 17, "600");
        const bw = t.width + 14, bh = 24;
        // v1 style: offset the box to the side (toward open space) and above the
        // point, stagger by parity, then run a diagonal leader from point to box.
        const rightSide = p.x < W - PAD.r - bw - 34;
        const dx = rightSide ? 26 : -26;
        const dy = -32 - (k % 2) * 13;
        let bx = rightSide ? p.x + dx : p.x + dx - bw;
        let by = p.y + dy - bh / 2;
        bx = Math.max(PAD.l + 2, Math.min(W - PAD.r - bw - 2, bx));
        by = Math.max(PAD.t + 2, Math.min(H - PAD.b - bh - 2, by));
        const anchorX = rightSide ? bx : bx + bw, anchorY = by + bh / 2;
        g.moveTo(p.x, p.y).lineTo(anchorX, anchorY).stroke({ width: 1, color: sp.color, alpha: 0.6 });
        g.circle(p.x, p.y, 2.2).fill({ color: sp.color });
        g.roundRect(bx, by, bw, bh, 5).fill({ color: INK, alpha: 0.96 }).stroke({ width: 1, color: sp.color, alpha: 0.9 });
        t.x = bx + (bw - t.width) / 2; t.y = by + (bh - t.height) / 2;
        this.labels.addChild(t);
      });
    }

    _legend() {
      let x = PAD.l + 4;
      this._legendHits = [];
      for (const sp of this.spec.series) {
        const hidden = this.hidden.has(sp.label);
        const item = new PIXI.Container();
        this._ephem.push(item); // container + its children are rebuilt per draw
        const t = new PIXI.Text({ text: sp.label, style: { fill: sp.color, fontSize: 15.5, fontFamily: "monospace" } });
        t.x = 14; t.y = 1; t.alpha = hidden ? 0.4 : 1;
        const dot = new PIXI.Graphics();
        if (hidden) dot.rect(0, 5, 10, 10).stroke({ width: 1.5, color: sp.color, alpha: 0.75 });
        else dot.rect(0, 5, 10, 10).fill({ color: sp.color });
        item.addChild(dot, t);
        item.x = x; item.y = 1;
        this.labels.addChild(item);
        // hit rect in CSS px (== pixi world units w/ autoDensity) for DOM-level toggle
        this._legendHits.push({ x0: x - 3, x1: x + t.width + 19, y0: 0, y1: 22, label: sp.label });
        x += 26 + t.width;
      }
    }

    _label(text, x, y, align, color, below, clampX) {
      const t = this._text(text, color, 14.5);
      if (align === "right") t.x = x - t.width;
      else if (align === "center") t.x = x - t.width / 2;
      else t.x = x;
      if (clampX) t.x = Math.max(clampX[0], Math.min(t.x, clampX[1] - t.width));
      t.y = below ? y : y - 6;
      this.labels.addChild(t);
    }

    // x-domain value under a canvas-local pixel x
    _xAt(px) {
      const [xmin, xmax] = this._xDomain();
      const W = this.app.screen.width;
      return xmin + (xmax - xmin) * (px - PAD.l) / (W - PAD.l - PAD.r);
    }

    _evalPointAt(px, py) {
      if (!this.scales) return null;
      const sp = this.spec.series.find(s => s.src === "eval" && s.key === "ppl");
      const s = sp && this.data.eval;
      const col = s && s.cols && s.cols.ppl;
      const scale = sp && (sp.axis === "y1" ? this.scales.sy1 : this.scales.sy);
      if (!sp || !s || !col || !col.length || !scale) return null;
      const xs = this._xa("eval"), i = nearestIdx(xs, this._xAt(px));
      const v = col[i];
      if (v == null || !isFinite(v)) return null;
      const x = this.scales.xpx(xs[i]), y = scale.px(v);
      return Math.hypot(x - px, y - py) <= 12 ? { step: s.step[i], ppl: v } : null;
    }

    _wire() {
      const cv = this.canvas;
      let dragging = false, dragX = 0, dragDom = null;
      let boxing = false, boxX0 = 0;
      let evalCand = null, downX = 0, downY = 0;
      cv.addEventListener("wheel", (e) => {
        e.preventDefault();
        const [xmin, xmax] = this._xDomain();
        const rect = cv.getBoundingClientRect();
        const at = this._xAt(e.clientX - rect.left);
        if (!isFinite(at)) return;
        const f = e.deltaY < 0 ? 0.82 : 1 / 0.82;
        const nr = (xmax - xmin) * f, frac = (at - xmin) / (xmax - xmin || 1);
        this.view = { min: at - frac * nr, max: at + (1 - frac) * nr };
        this.drawStatic();
        this._emitView();
      }, { passive: false });
      cv.addEventListener("pointerdown", (e) => {
        const r = cv.getBoundingClientRect();
        const px = e.clientX - r.left, py = e.clientY - r.top;
        for (const h of (this._legendHits || [])) {
          if (px >= h.x0 && px <= h.x1 && py >= h.y0 && py <= h.y1) {
            if (this.hidden.has(h.label)) this.hidden.delete(h.label); else this.hidden.add(h.label);
            saveHidden(this.id, this.hidden); this.drawStatic(); return;
          }
        }
        // A press near an eval marker is only a *candidate* click: opening on
        // pointerdown would steal drags/box-zooms that start within the 12px
        // hit radius. Decide on pointerup, when movement is known.
        evalCand = e.shiftKey ? null : this._evalPointAt(px, py);
        downX = e.clientX; downY = e.clientY;
        if (e.shiftKey) { boxing = true; boxX0 = px; return; }   // shift+drag = box zoom (x)
        dragging = true; dragX = e.clientX; dragDom = this._xDomain();
      });
      window.addEventListener("pointerup", (e) => {
        const cand = evalCand; evalCand = null;
        if (cand && !boxing &&
            Math.hypot(e.clientX - downX, e.clientY - downY) < 4 &&
            window.trainboard && window.trainboard.openEvalSamples) {
          dragging = false;
          window.trainboard.openEvalSamples(curRun, cand.step, cand.ppl);
          return;
        }
        if (boxing) {
          boxing = false;
          const r = cv.getBoundingClientRect();
          const px = e.clientX - r.left;
          if (Math.abs(px - boxX0) >= 8) {
            const a = this._xAt(Math.min(boxX0, px)), b = this._xAt(Math.max(boxX0, px));
            if (isFinite(a) && isFinite(b) && b > a) {
              this.view = { min: a, max: b };
              this.drawStatic();
              this._emitView();
            }
          } else { this.gOverlay.clear(); this._render(); }
          return;
        }
        const was = dragging; dragging = false; if (was) this._emitView();
      });
      cv.addEventListener("pointermove", (e) => {
        const rect = cv.getBoundingClientRect();
        if (boxing) {
          const px = e.clientX - rect.left;
          const H = this.app.screen.height;
          this.gOverlay.clear(); clearKids(this.tip);
          this.gOverlay.rect(Math.min(boxX0, px), PAD.t, Math.abs(px - boxX0), H - PAD.t - PAD.b)
            .fill({ color: 0x6fa8ff, alpha: 0.12 })
            .stroke({ width: 1, color: 0x6fa8ff, alpha: 0.6 });
          this._render();
        } else if (dragging && dragDom) {
          const W = this.app.screen.width;
          const dpx = (e.clientX - dragX) / (W - PAD.l - PAD.r);
          const span = dragDom[1] - dragDom[0];
          this.view = { min: dragDom[0] - dpx * span, max: dragDom[1] - dpx * span };
          this.drawStatic();
        } else {
          this._hover(e.clientX - rect.left, e.clientY - rect.top);
        }
      });
      cv.addEventListener("pointerleave", () => { this.gOverlay.clear(); clearKids(this.tip); this._render(); });
      cv.addEventListener("dblclick", () => { this.view = null; this.drawStatic(); if (this.onReset) this.onReset(); });
      cv.title = "click eval ppl: view captions · scroll: zoom-x · drag: pan · shift+drag: box zoom · dbl-click: reset";
    }

    _hover(mouseX, mouseY) {
      if (!this.scales || !this._visible) return;
      const { xmin, xmax, xpx } = this.scales;
      const W = this.app.screen.width, H = this.app.screen.height;
      if (mouseX < PAD.l || mouseX > W - PAD.r) { this.gOverlay.clear(); clearKids(this.tip); this._render(); return; }
      // marker hover: pointer in the top band -> describe the nearest timeline event
      if (mouseY != null && mouseY <= PAD.t + 9 && this._markHits && this._markHits.length) {
        let best = null, bd = 9;
        for (const m of this._markHits) { const d = Math.abs(m.x - mouseX); if (d < bd) { bd = d; best = m; } }
        if (best) {
          this.gOverlay.clear(); clearKids(this.tip);
          this.gOverlay.moveTo(best.x, PAD.t).lineTo(best.x, H - PAD.b).stroke({ width: 1.5, color: best.color, alpha: 0.85 });
          const e = best.e;
          const head = `${e.type}${e.severity ? " \u00b7 " + e.severity : ""} @ ${(e.step || 0).toLocaleString()}`;
          const txt = head + (e.label ? "\n" + e.label : "") + (e.detail ? "\n" + wrap(e.detail, 46) : "");
          const box = new PIXI.Text({ text: txt, style: { fill: TXT, fontSize: 15.5, fontFamily: "monospace", lineHeight: 18 } });
          const bx = Math.min(best.x + 10, W - box.width - 8), by = PAD.t + 6;
          const bg = new PIXI.Graphics();
          bg.roundRect(bx - 5, by - 4, box.width + 10, box.height + 8, 4).fill({ color: INK, alpha: 0.96 }).stroke({ width: 1, color: best.color, alpha: 0.7 });
          box.x = bx; box.y = by; this.tip.addChild(bg, box); this._render(); return;
        }
      }
      let xv = xmin + (xmax - xmin) * (mouseX - PAD.l) / (W - PAD.l - PAD.r);
      // snap to the nearest marked (eval) point so a sparse value is easy to land on
      const _snap = this.spec.series.find(sp => sp.labels || sp.points);
      if (_snap) { const xs = this._xa(_snap.src); if (xs && xs.length) xv = xs[nearestIdx(xs, xv)]; }
      this.gOverlay.clear(); clearKids(this.tip);
      this.gOverlay.moveTo(mouseX, PAD.t).lineTo(mouseX, H - PAD.b).stroke({ width: 1, color: 0x4a5663, alpha: 0.8 });

      // header: step (+ elapsed when wall-clock is usable)
      const timeAxis = !!this._timeLabels;
      const psrc = this.data.train ? "train" : (this.data.eval ? "eval" : null);
      let header = timeAxis ? `t +${fmtElapsed(xv)}` : `step ${Math.round(xv).toLocaleString()}`;
      if (psrc) {
        const ps = this.data[psrc], pxs = this._xa(psrc);
        if (ps.step.length) {
          const pi = nearestIdx(pxs, xv);
          if (timeAxis) header = `t +${fmtElapsed(xv)} · step ${ps.step[pi].toLocaleString()}`;
          else if (ps.__hasTs && ps.ts[pi] > 0 && this._t0) header = `step ${ps.step[pi].toLocaleString()} · t +${fmtElapsed(ps.ts[pi] - this._t0)}`;
        }
      }
      const lines = [header];
      for (const sp of this.spec.series) {
        if (sp.horizontal || this.hidden.has(sp.label)) continue;
        const s = this.data[sp.src];
        if (!s || !s.cols[sp.key] || !s.step.length) continue;
        const xs = this._xa(sp.src);
        const i = nearestIdx(xs, xv);
        const v = s.cols[sp.key][i];
        if (v == null || !isFinite(v)) continue;
        const scale = this.scales.norm ? (this._normScales && this._normScales[sp.key]) : (sp.axis === "y1" ? this.scales.sy1 : this.scales.sy);
        if (scale) { this.gOverlay.circle(xpx(xs[i]), scale.px(v), 7).stroke({ width: 2.5, color: sp.color }); }
        let line = `${sp.label}: ${fmtNum(v)}`;
        if (sp.best === "min") {
          // distance from the series' best so a regression is readable in place
          let bv = Infinity;
          const col = s.cols[sp.key];
          for (let j = 0; j < col.length; j++) { const w = col[j]; if (w != null && isFinite(w) && w < bv) bv = w; }
          if (isFinite(bv)) { const d = v - bv; line += d <= 0 ? "  (= best)" : `  (+${fmtNum(d)} vs best)`; }
        }
        lines.push(line);
      }
      const box = new PIXI.Text({ text: lines.join("\n"), style: { fill: TXT, fontSize: 15.5, fontFamily: "monospace", lineHeight: 18 } });
      const bx = Math.min(mouseX + 10, W - box.width - 8), by = PAD.t + 6;
      const bg = new PIXI.Graphics();
      bg.roundRect(bx - 5, by - 4, box.width + 10, box.height + 8, 4).fill({ color: INK, alpha: 0.9 }).stroke({ width: 1, color: 0x2a323b });
      box.x = bx; box.y = by;
      this.tip.addChild(bg, box);
      this._render();
    }
  }

  function nearestIdx(steps, target) {
    // binary search nearest
    let lo = 0, hi = steps.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (steps[m] < target) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(steps[lo - 1] - target) < Math.abs(steps[lo] - target)) return lo - 1;
    return lo;
  }

  // EMA over a column (carries the last value across gaps) — the trend line that
  // makes a noisy per-step metric (train loss, tok/s) actually readable.
  function emaCol(col, alpha) {
    const out = new Array(col.length).fill(null);
    let e = null;
    for (let i = 0; i < col.length; i++) {
      const v = col[i];
      if (v == null || !isFinite(v)) { out[i] = e; continue; }
      e = (e == null) ? v : alpha * v + (1 - alpha) * e;
      out[i] = e;
    }
    return out;
  }

  // ---- chart specs ----
  const CHARTS = [
    {
      id: "chart-loss", panel: null,
      series: [
        { src: "train", key: "loss", label: "train loss (ema)", color: COL.loss, axis: "y", type: "line", smooth: 0.1, robust: true, width: 2 },
        { src: "eval", key: "loss", label: "eval loss", color: COL.evalLoss, axis: "y", type: "scatter", r: 6.5 },
        { src: "eval", key: "ppl", label: "eval ppl", color: COL.ppl, axis: "y1", type: "line", points: true, r: 6.5, width: 2.6, labels: true, maxLabels: 24, labelFmt: v => v.toFixed(2), best: "min" },
        { src: "baseline", key: "ppl", label: "orig ppl", color: COL.baseline, axis: "y1", horizontal: true },
      ],
    },
    {
      id: "chart-training", panel: null,
      series: [
        { src: "train", key: "tok_per_sec", label: "tok/s (ema)", color: COL.tok_per_sec, axis: "y", type: "line", smooth: 0.08, robust: true, width: 2 },
        { src: "train", key: "gnorm", label: "gnorm(log)", color: COL.gnorm, axis: "y1", type: "line", log: true },
        { src: "train", key: "lr", label: "lr", color: COL.lr, axis: "y1", type: "line", dash: true },
      ],
    },
    {
      id: "chart-conv", panel: "conv-panel",
      requires: { src: "train", key: "lm_ce" },
      series: [
        { src: "train", key: "lm_ce", label: "lm_ce", color: COL.lm_ce, axis: "y", type: "line" },
        { src: "train", key: "block", label: "block", color: COL.block, axis: "y", type: "line" },
        { src: "train", key: "smt_mem", label: "smt", color: COL.smt_mem, axis: "y1", type: "line" },
        { src: "train", key: "dmt_mem", label: "dmt", color: COL.dmt_mem, axis: "y1", type: "line" },
      ],
    },
    {
      id: "chart-horizons", panel: "horizons-panel",
      requires: { src: "eval", key: "h4_top1" },
      series: [
        { src: "eval", key: "h1_top1", label: "h1", color: OI.sky, axis: "y", type: "line", width: 2 },
        { src: "eval", key: "h2_top1", label: "h2", color: OI.green, axis: "y", type: "line", width: 2 },
        { src: "eval", key: "h3_top1", label: "h3", color: OI.yellow, axis: "y", type: "line", width: 2 },
        { src: "eval", key: "h4_top1", label: "h4", color: OI.vermillion, axis: "y", type: "line", width: 2 },
        { src: "eval", key: "h1_top5", label: "h1·5", color: OI.sky, axis: "y", type: "line", dash: true },
        { src: "eval", key: "h2_top5", label: "h2·5", color: OI.green, axis: "y", type: "line", dash: true },
        { src: "eval", key: "h3_top5", label: "h3·5", color: OI.yellow, axis: "y", type: "line", dash: true },
        { src: "eval", key: "h4_top5", label: "h4·5", color: OI.vermillion, axis: "y", type: "line", dash: true },
        { src: "eval", key: "h1_ppl", label: "h1 ppl", color: OI.sky, axis: "y1", type: "line" },
        { src: "eval", key: "h4_ppl", label: "h4 ppl", color: OI.vermillion, axis: "y1", type: "line" },
      ],
    },
    {
      id: "chart-grounding", panel: "grounding-panel",
      requires: { src: "train", key: "grounding_contrastive_loss" },
      series: [
        { src: "train", key: "ce_loss", label: "raw CE", color: OI.gray, axis: "y", type: "line", smooth: 0.1, robust: true },
        { src: "train", key: "grounded_ce_loss", label: "opening-weighted CE", color: OI.sky, axis: "y", type: "line", smooth: 0.1, robust: true, width: 2 },
        { src: "train", key: "grounding_contrastive_loss", label: "image/text contrastive", color: OI.vermillion, axis: "y", type: "line", smooth: 0.1, robust: true },
        { src: "train", key: "grounding_retrieval_accuracy", label: "batch retrieval", color: OI.green, axis: "y1", type: "line", smooth: 0.1 },
      ],
    },
    {
      // grokking: ROSA/Engram recall paths are identity-init no-ops that "grok on"
      // (injection RMS rises off ~0). Panel hides until a run emits rosa_inj_rms.
      id: "chart-memact", panel: "memact-panel",
      requires: { src: "train", key: "rosa_inj_rms" },
      series: [
        { src: "train", key: "rosa_inj_rms", label: "ROSA inj", color: OI.sky, axis: "y", type: "line" },
        { src: "train", key: "engram_inj_rms", label: "Engram inj", color: OI.green, axis: "y", type: "line" },
        { src: "train", key: "rosa_e_gap", label: "ROSA |e1-e0|", color: OI.yellow, axis: "y1", type: "line", dash: true },
      ],
    },
    {
      // grokking: train block-MSE falls while held-out block / gen_gap reveal whether
      // it generalizes or just memorizes. Panel hides until a run emits block_val.
      id: "chart-memgen", panel: "memgen-panel",
      requires: { src: "eval", key: "block_val" },
      series: [
        { src: "train", key: "block", label: "train block", color: COL.block, axis: "y", type: "line" },
        { src: "eval", key: "block_val", label: "held-out block", color: OI.sky, axis: "y", type: "scatter", r: 4 },
        { src: "eval", key: "gen_gap", label: "gen gap", color: OI.purple, axis: "y1", type: "line" },
      ],
    },
  ];

  // union of fields to fetch in one request
  const TRAIN_FIELDS = ["loss", "tok_per_sec", "gnorm", "lr", "lm_ce", "block", "smt_mem", "dmt_mem",
    "rosa_inj_rms", "engram_inj_rms", "rosa_e_gap", "wnorm_rms", "stable_rank",
    "ce_loss", "grounded_ce_loss", "grounding_contrastive_loss", "grounding_retrieval_accuracy",
    "deep_vision_inj_rms"];
  const EVAL_FIELDS = ["loss", "ppl", "top1", "top5",
    "h1_top1", "h2_top1", "h3_top1", "h4_top1",
    "h1_top5", "h2_top5", "h3_top5", "h4_top5", "h1_ppl", "h4_ppl",
    "block_val", "gen_gap"];

  // ---- timeline marker helpers (shared by chart overlays + event list) ----
  function markColor(e) {
    if (e.type === "checkpoint") return OI.gray;
    if (e.type === "control") return OI.blue;
    if (e.type === "action") return OI.purple;
    return e.severity === "critical" ? OI.vermillion : OI.yellow; // alert
  }
  function cssHex(n) { return "#" + ((n >>> 0) & 0xffffff).toString(16).padStart(6, "0"); }
  function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function wrap(str, n) {
    const words = String(str).split(/\s+/); let line = "", out = [];
    for (const w of words) { if ((line + " " + w).trim().length > n) { out.push(line.trim()); line = w; } else line += " " + w; }
    if (line.trim()) out.push(line.trim());
    return out.join("\n");
  }

  // ---- controller ----
  const charts = {};
  let customChart = null;
  const OICYCLE = [OI.orange, OI.sky, OI.green, OI.vermillion, OI.blue, OI.purple, OI.yellow, OI.gray];
  function loadCustom() { try { return new Set(JSON.parse(localStorage.getItem("tb_custom") || "[]")); } catch (e) { return new Set(); } }
  function saveCustom(set) { try { localStorage.setItem("tb_custom", JSON.stringify([...set])); } catch (e) {} }
  const selectedMetrics = loadCustom();

  function renderCatalog(cat) {
    const host = document.getElementById("metric-catalog");
    if (!host) return;
    host.innerHTML = "";
    const avail = new Set();
    for (const src of ["train", "eval"]) for (const k of (cat[src] || [])) avail.add(src + ":" + k);
    for (const m of [...selectedMetrics]) if (!avail.has(m)) selectedMetrics.delete(m);
    for (const src of ["train", "eval"]) {
      const keys = cat[src] || []; if (!keys.length) continue;
      const grp = document.createElement("div"); grp.className = "mc-group";
      const lab = document.createElement("span"); lab.className = "mc-label"; lab.textContent = src;
      grp.appendChild(lab);
      for (const k of keys) {
        const id = src + ":" + k;
        const chip = document.createElement("button");
        chip.className = "mc-chip" + (selectedMetrics.has(id) ? " active" : "");
        chip.textContent = k; chip.dataset.metric = id;
        chip.addEventListener("click", () => {
          if (selectedMetrics.has(id)) selectedMetrics.delete(id); else selectedMetrics.add(id);
          chip.classList.toggle("active");
          saveCustom(selectedMetrics); applyCustom();
        });
        grp.appendChild(chip);
      }
      host.appendChild(grp);
    }
  }

  let customGen = 0, customRun = "", customTrainStep = -1, customEvalStep = -1;
  let customAppendRequest = 0, customAppendApplied = 0;
  async function applyCustom() {
    const box = document.getElementById("custom-chartbox");
    if (!customChart) return;
    const sel = [...selectedMetrics];
    const gen = ++customGen;
    if (!sel.length || !curRun) {
      customRun = "";
      if (box) box.style.display = "none";
      return;
    }
    if (box) box.style.display = "";            // show as soon as there's a selection
    const run = curRun;
    const trainF = sel.filter(m => m.startsWith("train:")).map(m => m.slice(6));
    const evalF = sel.filter(m => m.startsWith("eval:")).map(m => m.slice(5));
    const url = `/api/series/${encodeURIComponent(run)}?train=${trainF.join(",")}&eval=${evalF.join(",")}`;
    const data = await fetchJSON(url).catch(() => null);
    if (!data || gen !== customGen || run !== curRun) return;
    let ci = 0; const series = [];
    for (const m of sel) {
      const src = m.startsWith("train:") ? "train" : "eval";
      const key = m.slice(src.length + 1);
      series.push({ src, key, label: key, color: OICYCLE[ci++ % OICYCLE.length], axis: "y", type: "line", smooth: src === "train" ? 0.1 : 0 });
    }
    customChart.spec.series = series;
    customChart.hidden = new Set();
    if (customChart._resize) customChart._resize();
    customChart.setData(data);
    customRun = run;
    customTrainStep = seriesTip(data.train, -1);
    customEvalStep = seriesTip(data.eval, -1);
  }

  async function appendCustom(run) {
    const sel = [...selectedMetrics];
    if (!sel.length || !customChart || customRun !== run) return;
    const gen = customGen;
    const request = ++customAppendRequest;
    const trainF = sel.filter(m => m.startsWith("train:")).map(m => m.slice(6));
    const evalF = sel.filter(m => m.startsWith("eval:")).map(m => m.slice(5));
    const url = `/api/series/${encodeURIComponent(run)}?train=${trainF.join(",")}&eval=${evalF.join(",")}`
      + `&train_since=${customTrainStep - 1}&eval_since=${customEvalStep - 1}`;
    const data = await fetchJSON(url).catch(() => null);
    if (!data || gen !== customGen || run !== customRun || request < customAppendApplied) return;
    customChart.append(data);
    // Commit cursors only after the chart accepts the rows. Several main-series
    // ticks can have custom fetches in flight at once, so they are monotonic too.
    customTrainStep = Math.max(customTrainStep, seriesTip(data.train, customTrainStep));
    customEvalStep = Math.max(customEvalStep, seriesTip(data.eval, customEvalStep));
    customAppendApplied = request;
  }

  function persistCompare(name) { try { localStorage.setItem("tb_compare", name || ""); } catch (e) {} }
  function loadCompare() { try { return localStorage.getItem("tb_compare") || ""; } catch (e) { return ""; } }
  let compareGen = 0;
  async function setCompareRun(name) {
    const gen = ++compareGen;
    persistCompare(name);
    if (!name || name === curRun) { for (const id in charts) charts[id].setCompare(null); return; }
    const url = `/api/series/${encodeURIComponent(name)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}`;
    const data = await fetchJSON(url).catch(() => null);
    if (!data || gen !== compareGen || name !== loadCompare()) return;
    for (const id in charts) charts[id].setCompare(data, name);
  }
  function populateCompareOptions() {
    const sel = document.getElementById("compare-run");
    if (!sel) return;
    const runs = [...document.querySelectorAll('#run-list .run-item[data-run]')].map(e => e.getAttribute("data-run"));
    const keep = sel.value;
    sel.innerHTML = ['<option value="">none</option>']
      .concat(runs.filter(r => r !== curRun).map(r => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`)).join("");
    if (keep && keep !== curRun && runs.includes(keep)) sel.value = keep;
  }
  function wireCompare() {
    const sel = document.getElementById("compare-run");
    if (sel) sel.addEventListener("change", () => setCompareRun(sel.value));
  }

  async function loadCatalog(run) {
    const gen = loadGen;
    const cat = await fetchJSON(`/api/metrics/${encodeURIComponent(run)}`).catch(() => null);
    if (!cat || run !== curRun || gen !== loadGen) return;
    renderCatalog(cat);
    applyCustom();
  }
  function redrawAll() { for (const id in charts) charts[id].drawStatic(); if (customChart) customChart.drawStatic(); }
  function focusStep(step) {
    if (tailN) { tailN = 0; saveTail(); updateTailChips(); }   // focusing an event unpins the tail
    for (const id in charts) {
      const c = charts[id];
      let lo = c._stepToX(Math.max(0, step - 400)), hi = c._stepToX(step + 400);
      if (hi <= lo) { lo -= 1; hi = lo + 2; }
      c.view = { min: lo, max: hi };
      c.drawStatic();
    }
  }
  function renderEventList(events) {
    const host = document.getElementById("event-list");
    const panel = document.getElementById("events-panel");
    if (!host) return;
    if (!events || !events.length) { if (panel) panel.style.display = "none"; host.innerHTML = ""; return; }
    if (panel) panel.style.display = "";
    host.innerHTML = "";
    for (const e of events) {
      const row = document.createElement("div");
      row.className = "evrow";
      const c = cssHex(markColor(e)), sev = e.severity ? " \u00b7 " + e.severity : "";
      row.innerHTML = '<span class="evdot" style="background:' + c + '"></span>'
        + '<span class="evstep">' + (e.step || 0).toLocaleString() + '</span>'
        + '<span class="evtype" style="color:' + c + '">' + escapeHtml(e.type + sev) + '</span>'
        + '<span class="evlabel">' + escapeHtml(e.label || "") + '</span>'
        + '<span class="evdetail">' + escapeHtml(e.detail || "") + '</span>';
      row.addEventListener("click", () => focusStep(e.step || 0));
      host.appendChild(row);
    }
  }
  function wireSmoothControls() {
    const cb = document.getElementById("smooth-on"), sl = document.getElementById("smooth-alpha");
    if (cb) { cb.checked = SMOOTH.on; cb.addEventListener("change", () => { SMOOTH.on = cb.checked; saveSmooth(); redrawAll(); }); }
    if (sl) { sl.value = String(SMOOTH.alpha); sl.addEventListener("input", () => { SMOOTH.alpha = parseFloat(sl.value); saveSmooth(); redrawAll(); }); }
    const ly = document.getElementById("log-y");
    if (ly) { ly.checked = LOGY.on; ly.addEventListener("change", () => { LOGY.on = ly.checked; saveLogy(); redrawAll(); }); }
    const tx = document.getElementById("time-x");
    if (tx) {
      tx.checked = XM.time;
      tx.addEventListener("change", () => {
        XM.time = tx.checked; saveXmode();
        // views are stored in x-units, so a mode flip invalidates them
        for (const id in charts) charts[id].view = null;
        if (customChart) customChart.view = null;
        if (tailN) applyTail(); else redrawAll();
      });
    }
  }

  // ---- tail window chips: keep the view glued to the last N steps ----
  let tailN = (() => { try { return parseInt(localStorage.getItem("tb_tail") || "0", 10) || 0; } catch (e) { return 0; } })();
  function saveTail() { try { localStorage.setItem("tb_tail", String(tailN)); } catch (e) {} }
  function updateTailChips() {
    document.querySelectorAll("[data-tail]").forEach(btn => {
      const n = parseInt(btn.dataset.tail, 10) || 0;
      btn.classList.toggle("active", tailN > 0 && n === tailN);
    });
  }
  function applyTail() {
    if (!tailN || !lastStep) { redrawAll(); return; }
    const c = charts["chart-loss"];
    if (!c) return;
    const lo = c._stepToX(Math.max(0, lastStep - tailN));
    let hi = c._stepToX(lastStep);
    if (hi <= lo) hi = lo + 1;
    onChartZoom({ min: lo, max: hi }, true);
  }
  function setTail(n) {
    tailN = n; saveTail(); updateTailChips();
    if (!n) { onChartReset(); return; }
    applyTail();
  }
  function wireTailChips() {
    document.querySelectorAll("[data-tail]").forEach(btn =>
      btn.addEventListener("click", () => setTail(parseInt(btn.dataset.tail, 10) || 0)));
    updateTailChips();
  }

  let curRun = null, lastStep = 0, lastTrainStep = -1, lastEvalStep = -1;
  let lastGeneration = -1;
  let lastVersion = 0, pendingVersion = 0, loading = false, loadGen = 0;
  let runLoaded = false;
  let seriesRetryMs = 500, seriesRetryTimer = null;
  let shownEvalSampleStep = -1;
  let timelineGen = 0;
  let zoomTimer = null, zoomGen = 0;
  async function fetchJSON(url, timeoutMs = 8000) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const response = await fetch(url, { signal: ac.signal });
      if (!response.ok) throw new Error(`request failed (${response.status})`);
      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  }
  function scheduleSeriesRetry(run, callback, immediate = false) {
    clearTimeout(seriesRetryTimer);
    const delay = immediate ? 0 : seriesRetryMs;
    if (!immediate) seriesRetryMs = Math.min(seriesRetryMs * 2, 8000);
    seriesRetryTimer = setTimeout(() => {
      seriesRetryTimer = null;
      if (run === curRun) callback();
    }, delay);
  }
  function syncView(view) {
    for (const id in charts) {
      if (id === "chart-custom") continue;          // custom chart manages its own data
      charts[id].view = view ? { min: view.min, max: view.max } : null;
      charts[id].drawStatic();
    }
  }
  function onChartZoom(view, fromTail) {
    if (!fromTail && tailN) { tailN = 0; saveTail(); updateTailChips(); } // manual zoom unpins the tail
    syncView(view);                                  // instant zoom on the decimated data
    const gen = ++zoomGen;
    clearTimeout(zoomTimer);
    zoomTimer = setTimeout(async () => {
      if (!curRun) return;
      const run = curRun, runGen = loadGen;
      // view is in the active x-unit; the API window is steps — map back if needed
      const primary = charts["chart-loss"];
      let from, to;
      if (XM.time && primary) { from = primary._xToStep(view.min); to = primary._xToStep(view.max); }
      else { from = Math.floor(view.min); to = Math.ceil(view.max); }
      from = Math.max(0, from);
      if (to - from < 2) return;
      const url = `/api/series/${encodeURIComponent(run)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}&from=${from}&to=${to}`;
      const data = await fetchJSON(url).catch(() => null);
      if (!data || gen !== zoomGen || run !== curRun || runGen !== loadGen) return;
      for (const id in charts) { if (id !== "chart-custom") charts[id].setWindow(data); }
    }, 260);
  }
  function onChartReset() {
    zoomGen++;
    if (tailN) { tailN = 0; saveTail(); updateTailChips(); }
    for (const id in charts) { if (id !== "chart-custom") charts[id].restoreFull(); }
  }

  function distribute(chart, data) {
    // give each chart the full payload; it reads only the cols it needs
    return data;
  }

  function seriesTip(series, fallback) {
    return series && series.step && series.step.length
      ? series.step[series.step.length - 1] : fallback;
  }

  function showNewestEvalSample(run, series) {
    if (!series || !series.step || !series.step.length || !series.cols || !series.cols.ppl) return;
    for (let i = series.step.length - 1; i >= 0; i--) {
      const ppl = series.cols.ppl[i];
      if (ppl == null || !isFinite(ppl)) continue;
      const step = series.step[i];
      if (step === shownEvalSampleStep) return;
      shownEvalSampleStep = step;
      if (window.trainboard && window.trainboard.openEvalSamples) {
        window.trainboard.openEvalSamples(run, step, ppl);
      }
      return;
    }
  }

  async function loadRun(run, version) {
    const gen = ++loadGen;
    ++timelineGen; // invalidate timeline responses launched by the old view
    const switching = run !== curRun;
    curRun = run;
    runLoaded = false;
    customGen++; customRun = ""; // invalidate custom appends on rewinds too
    if (switching) {
      clearTimeout(seriesRetryTimer); seriesRetryTimer = null;
      lastStep = 0; lastTrainStep = -1; lastEvalStep = -1;
      lastGeneration = -1;
      lastVersion = 0; pendingVersion = 0; seriesRetryMs = 500;
      compareGen++;
    }
    shownEvalSampleStep = -1;
    if (window.trainboard && window.trainboard.resetEvalSamples) {
      window.trainboard.resetEvalSamples(run);
    }
    pendingVersion = version || 0;
    loading = true;
    const url = `/api/series/${encodeURIComponent(run)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}`;
    const [data, tl] = await Promise.all([
      fetchJSON(url).catch(() => null),
      fetchJSON(`/api/timeline/${encodeURIComponent(run)}`).catch(() => ({ events: [] })),
    ]);
    if (gen !== loadGen) return;
    loading = false;
    if (!data) {
      scheduleSeriesRetry(run, () => loadRun(run, pendingVersion));
      return;
    }
    seriesRetryMs = 500;
    try {
    lastTrainStep = seriesTip(data.train, -1);
    lastEvalStep = seriesTip(data.eval, -1);
    lastGeneration = Number(data.generation || 0);
    lastStep = Math.max(lastTrainStep, lastEvalStep, 0);
    lastVersion = version || pendingVersion;
    showNewestEvalSample(run, data.eval);
    const note = document.getElementById("decim-note");
    if (note) note.style.display = data.decimated ? "" : "none";
    const events = (tl && tl.events) || [];
    for (const spec of CHARTS) {
      const ch = charts[spec.id];
      if (!ch) continue;
      ch.timeline = events;
      togglePanel(spec, data);
      ch.setData(data);
    }
    if (tailN) applyTail();                          // sticky tail follows across run switches
    renderEventList(events);
    saveRun(run);
    loadCatalog(run);
    populateCompareOptions();
    const cmp = loadCompare();
    const cmpSel = document.getElementById("compare-run");
    if (cmp && cmp !== run) { if (cmpSel) cmpSel.value = cmp; setCompareRun(cmp); }
    else { if (cmpSel) cmpSel.value = ""; for (const id in charts) charts[id].setCompare(null); }
    runLoaded = true;
    } catch (e) {
      console.warn("[trainboard] loadRun render failed; retrying full load:", e);
      scheduleSeriesRetry(run, () => loadRun(run, pendingVersion));
      return;
    }
    if (pendingVersion > lastVersion) {
      scheduleSeriesRetry(run, () => appendRun(run, pendingVersion), true);
    }
  }

  async function appendRun(run, version) {
    pendingVersion = Math.max(pendingVersion, version || 0);
    if (version <= lastVersion || loading || run !== curRun || !runLoaded) return;
    clearTimeout(seriesRetryTimer); seriesRetryTimer = null;
    loading = true;
    const gen = loadGen;
    let succeeded = false;
    let rewound = false;
    // Crash-isolate the whole live-append tick: a throw here (or a hung fetch)
    // must never leave `loading` stuck true, or EVERY later tick returns early
    // and the chart freezes until a manual reload (observed on a NaN/diverged
    // run). finally always clears the flag; catch keeps one bad point from
    // killing the live loop and logs it so the real trigger is visible.
    try {
      // Overlap each cursor by one integer step. A trainer can append a
      // corrected row at the current tip; the chart's upsert path replaces it.
      const url = `/api/series/${encodeURIComponent(run)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}&train_since=${lastTrainStep - 1}&eval_since=${lastEvalStep - 1}`;
      // Bound the request: a wedged/deadlocked query must not hang the await
      // forever (which would strand `loading` true and freeze all live ticks).
      const data = await fetchJSON(url).catch(() => null);
      if (!data) return;
      // A run switch/full reload can begin while this request is in flight.  Do
      // not let the obsolete append alter cursors, charts, or the eval card;
      // its finally block must not release the newer load's global lock either.
      if (run !== curRun || gen !== loadGen) return;
      succeeded = true;
      seriesRetryMs = 500;
      const generation = Number(data.generation || 0);
      if ((lastGeneration >= 0 && generation !== lastGeneration) ||
          Number(data.max_train_step || 0) < lastTrainStep ||
          Number(data.max_eval_step || 0) < lastEvalStep) {
        // The ingester reset after checkpoint recovery. Append-only charts
        // cannot delete abandoned future points, so replace them with one
        // authoritative full load instead of requiring a manual run switch.
        rewound = true;
        return;
      }
      lastTrainStep = seriesTip(data.train, lastTrainStep);
      lastEvalStep = seriesTip(data.eval, lastEvalStep);
      lastGeneration = generation;
      showNewestEvalSample(run, data.eval);
      lastStep = Math.max(lastTrainStep, lastEvalStep, 0);
      lastVersion = version;
      for (const spec of CHARTS) {
        const ch = charts[spec.id];
        if (ch) ch.append(data);
      }
      appendCustom(run).catch((e) => {
        console.warn("[trainboard] custom append failed; rebuilding:", e);
        if (run === curRun && customRun === run) applyCustom();
      });
      // pinned tail slides with the live tip (appended rows are full resolution)
      if (tailN && charts["chart-loss"]) {
        const c = charts["chart-loss"];
        const lo = c._stepToX(Math.max(0, lastStep - tailN));
        let hi = c._stepToX(lastStep);
        if (hi <= lo) hi = lo + 1;
        syncView({ min: lo, max: hi });
      }
      const timelineRequest = ++timelineGen;
      fetchJSON(`/api/timeline/${encodeURIComponent(run)}`).then(tl => {
        if (run !== curRun || gen !== loadGen || timelineRequest !== timelineGen) return;
        const events = (tl && tl.events) || [];
        for (const id in charts) charts[id].setTimeline(events);
        renderEventList(events);
      }).catch(() => {});
    } catch (e) {
      console.warn("[trainboard] appendRun failed (live tick skipped):", e);
      // Cursors/charts may already have been mutated before a Pixi render
      // throws. Treat the tick like a rewind and rebuild every chart from one
      // authoritative full payload; merely clearing `loading` would commit a
      // partial client transaction and skip this version forever.
      rewound = true;
    } finally {
      if (run !== curRun || gen !== loadGen) return;
      loading = false;
      if (run === curRun && rewound) {
        runLoaded = false;
        scheduleSeriesRetry(run, () => loadRun(run, pendingVersion), true);
      } else if (run === curRun && pendingVersion > lastVersion) {
        scheduleSeriesRetry(
          run, () => appendRun(run, pendingVersion), succeeded);
      }
    }
  }

  function togglePanel(spec, data) {
    if (!spec.panel) return;
    const el = document.getElementById(spec.panel);
    if (!el) return;
    let has = true;
    if (spec.requires) {
      const s = data[spec.requires.src];
      has = !!(s && s.cols[spec.requires.key] && s.cols[spec.requires.key].some(v => v != null));
    }
    el.style.display = has ? "" : "none";
  }

  function restoreRun() {
    const want = desiredRun();
    if (!want) return;
    let tries = 0;
    const sel = (window.CSS && CSS.escape) ? CSS.escape(want) : want;
    const tick = () => {
      if (curRun === want) return;
      const el = document.querySelector('#run-list .run-item[data-run="' + sel + '"]');
      if (el) { el.dispatchEvent(new MouseEvent("click", { bubbles: true })); return; }
      if (++tries < 40) setTimeout(tick, 150);
    };
    setTimeout(tick, 250);
  }

  async function boot() {
    // render-on-demand: no chart animates, so the global tickers must not spin
    // a 60fps rAF loop on the training box (per-app tickers are already off)
    try { PIXI.Ticker.shared.autoStart = false; PIXI.Ticker.shared.stop(); } catch (e) {}
    try { PIXI.Ticker.system.autoStart = false; PIXI.Ticker.system.stop(); } catch (e) {}
    for (const spec of CHARTS) {
      const ch = new Chart(spec.id, spec);
      await ch.init();
      ch.onView = onChartZoom;
      ch.onReset = onChartReset;
      charts[spec.id] = ch;
    }
    customChart = new Chart("chart-custom", { id: "chart-custom", normalize: true, series: [] });
    await customChart.init();
    wireSmoothControls();
    wireTailChips();
    wireCompare();
    restoreRun();
    // React to (run, version) changes published by Datastar via #active-run.
    if (window.trainboard && window.trainboard.watchActiveRun) {
      window.trainboard.watchActiveRun((run, version) => {
        if (!run) return;
        if (run !== curRun) loadRun(run, version);
        else if (version < lastVersion) loadRun(run, version);
        else appendRun(run, version);
      });
    }
  }

  window.addEventListener("DOMContentLoaded", boot);
})();
