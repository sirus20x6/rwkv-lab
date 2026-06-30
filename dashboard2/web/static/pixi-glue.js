// trainboard Pixi chart layer.
//
// Each Chart owns a PIXI.Application on a <canvas>, holds its own data, and
// redraws on the GPU. The ChartController fetches /api/series for the selected
// run (one union request), distributes columns to charts, and on each
// runVersion bump fetches ?since=<lastStep> and APPENDS only new points — no
// full re-render of the existing curve. Interactions: wheel = zoom-x,
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
  const AXIS = 0x7a8594, GRID = 0x222a33, TXT = 0xaab1b9, INK = 0x0f1113;
  const PAD = { l: 52, r: 52, t: 14, b: 26 };

  // user smoothing preference + per-chart hidden-series set, persisted to localStorage
  const SMOOTH = (() => { try { return Object.assign({ on: true, alpha: 0.10 }, JSON.parse(localStorage.getItem("tb_smooth") || "{}")); } catch (e) { return { on: true, alpha: 0.10 }; } })();
  function saveSmooth() { try { localStorage.setItem("tb_smooth", JSON.stringify(SMOOTH)); } catch (e) {} }
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
    const step = [], cols = {};
    for (const k in (s.cols || {})) cols[k] = [];
    let last = -Infinity;
    for (const i of idx) {
      const st = s.step[i];
      if (!(st > last)) continue;            // skip <= previous (dup / non-increasing)
      last = st; step.push(st);
      for (const k in cols) cols[k].push(s.cols[k][i]);
    }
    return { step, cols };
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
      });
      this.gStatic = new PIXI.Graphics();
      this.labels = new PIXI.Container();
      this.gOverlay = new PIXI.Graphics();
      this.tip = new PIXI.Container();
      this.app.stage.addChild(this.gStatic, this.labels, this.gOverlay, this.tip);

      this._wire();
      const ro = new ResizeObserver(() => this._resize());
      ro.observe(this.canvas.parentElement);
      this.ready = true;
    }

    _resize() {
      const box = this.canvas.parentElement;
      if (!box) return;
      this.app.renderer.resize(box.clientWidth, box.clientHeight);
      this.drawStatic();
    }

    setData(d) {
      this._fullData = d;
      this.data = { train: monotonicSeries(d.train), eval: monotonicSeries(d.eval), baseline: d.baseline || null };
      this.view = null;
      this.drawStatic();
    }

    // setWindow swaps in a full-resolution slice for the current zoom (keeps view).
    setWindow(d) {
      const base = this.data ? this.data.baseline : null;
      this.data = { train: monotonicSeries(d.train), eval: monotonicSeries(d.eval), baseline: (d.baseline || base) || null };
      this.drawStatic();
    }

    restoreFull() {
      if (!this._fullData) { this.view = null; this.drawStatic(); return; }
      const d = this._fullData;
      this.data = { train: monotonicSeries(d.train), eval: monotonicSeries(d.eval), baseline: d.baseline || null };
      this.view = null;
      this.drawStatic();
    }

    _emitView() { if (this.onView && this.view) this.onView(this.view); }

    // Append incremental rows, keeping the step axis strictly increasing: skip any
    // row whose step is <= the current tip (duplicate restart row / overlapping fetch).
    append(d) {
      for (const src of ["train", "eval"]) {
        const inc = d[src];
        if (!inc || !inc.step || !inc.step.length) continue;
        const cur = this.data[src];
        if (!cur) { this.data[src] = monotonicSeries(inc); continue; }
        let maxStep = cur.step.length ? cur.step[cur.step.length - 1] : -Infinity;
        for (let i = 0; i < inc.step.length; i++) {
          const st = inc.step[i];
          if (!(st > maxStep)) continue;
          maxStep = st;
          cur.step.push(st);
          for (const k in inc.cols) {
            if (!cur.cols[k]) cur.cols[k] = [];
            cur.cols[k].push(inc.cols[k][i]);
          }
        }
      }
      this.drawStatic();
    }

    _xDomain() {
      if (this.view) return [this.view.min, this.view.max];
      let lo = Infinity, hi = -Infinity;
      for (const src of ["train", "eval"]) {
        const s = this.data[src];
        if (s && s.step.length) { lo = Math.min(lo, s.step[0]); hi = Math.max(hi, s.step[s.step.length - 1]); }
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
        const col = s.cols[sp.key], steps = s.step;
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
      const log = this.spec.series.some(s => s.axis === axis && s.log);
      const [lo, hi] = dom;
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
      const W = this.app.screen.width, H = this.app.screen.height;
      const g = this.gStatic; g.clear(); clearKids(this.labels);
      clearKids(this.tip); this.gOverlay.clear();

      const [xmin, xmax] = this._xDomain();
      const xpx = step => PAD.l + (W - PAD.l - PAD.r) * (step - xmin) / (xmax - xmin || 1);
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
          this._label(fmtNum(val), PAD.l - 6, y, "right", 0x99a3af);
        }
      }
      if (sy1) {
        for (let i = 0; i <= ticks; i++) {
          const t = i / ticks;
          const val = sy1.log ? Math.pow(10, Math.log10(sy1.lo) + t * (Math.log10(sy1.hi) - Math.log10(sy1.lo))) : sy1.lo + t * (sy1.hi - sy1.lo);
          this._label(fmtNum(val), W - PAD.r + 6, sy1.px(val), "left", 0x99a3af);
        }
      }
      // x ticks
      for (let i = 0; i <= 4; i++) {
        const step = xmin + (i / 4) * (xmax - xmin);
        const x = xpx(step);
        this._label(Math.round(step).toLocaleString(), x, H - PAD.b + 4, "center", TXT, true);
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
        if (norm) { scale = this._normScale(s.cols[sp.key], xmin, xmax, s.step); this._normScales[sp.key] = scale; }
        if (!scale) continue;
        if (sp.type === "scatter") this._drawScatter(s, sp, scale, xpx);
        else this._drawLine(s, sp, scale, xpx);
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
      // legend
      this._legend();
    }

    setTimeline(evs) { this.timeline = Array.isArray(evs) ? evs : []; this.drawStatic(); }

    setCompare(d, label) {
      this.compare = d ? { train: monotonicSeries(d.train), eval: monotonicSeries(d.eval) } : null;
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
        const sm = emaCol(col, 0.12);                 // clean trend so B reads as a faint underlay
        let pen = false;
        for (let i = 0; i < s.step.length; i++) {
          const st = s.step[i];
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
        const t = new PIXI.Text({ text: "dim = " + this.compareLabel, style: { fill: 0x9aa4b0, fontSize: 13.5, fontFamily: "monospace" } });
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
        const st = e.step || 0;
        if (st < xmin || st > xmax) continue;
        const x = xpx(st), c = markColor(e);
        for (let yy = PAD.t; yy < H - PAD.b; yy += 9) {
          g.moveTo(x, yy).lineTo(x, Math.min(yy + 4, H - PAD.b)).stroke({ width: 1, color: c, alpha: 0.3 });
        }
        g.moveTo(x - 4, PAD.t - 1).lineTo(x + 4, PAD.t - 1).lineTo(x, PAD.t + 5).fill({ color: c, alpha: 0.95 });
        this._markHits.push({ x, e, color: c });
      }
    }

    _drawLine(s, sp, scale, xpx) {
      const g = this.gStatic, col = s.cols[sp.key], steps = s.step;
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

    _drawScatter(s, sp, scale, xpx) {
      const g = this.gStatic, col = s.cols[sp.key], steps = s.step;
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
      const col = s.cols[sp.key], steps = s.step, g = this.gStatic;
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
        const t = new PIXI.Text({ text: txt, style: { fill: sp.color, fontSize: 16.2, fontFamily: "monospace", fontWeight: "600" } });
        const bw = t.width + 14, bh = 22;
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
        const t = new PIXI.Text({ text: sp.label, style: { fill: sp.color, fontSize: 15.6, fontFamily: "monospace" } });
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

    _label(text, x, y, align, color, below) {
      const t = new PIXI.Text({ text, style: { fill: color, fontSize: 15.0, fontFamily: "monospace" } });
      if (align === "right") t.x = x - t.width;
      else if (align === "center") t.x = x - t.width / 2;
      else t.x = x;
      t.y = below ? y : y - 6;
      this.labels.addChild(t);
    }

    _wire() {
      const cv = this.canvas;
      let dragging = false, dragX = 0, dragDom = null;
      cv.addEventListener("wheel", (e) => {
        e.preventDefault();
        const [xmin, xmax] = this._xDomain();
        const rect = cv.getBoundingClientRect();
        const W = this.app.screen.width;
        const at = xmin + (xmax - xmin) * ((e.clientX - rect.left) - PAD.l) / (W - PAD.l - PAD.r);
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
        dragging = true; dragX = e.clientX; dragDom = this._xDomain();
      });
      window.addEventListener("pointerup", () => { const was = dragging; dragging = false; if (was) this._emitView(); });
      cv.addEventListener("pointermove", (e) => {
        if (dragging && dragDom) {
          const W = this.app.screen.width, rect = cv.getBoundingClientRect();
          const dpx = (e.clientX - dragX) / (W - PAD.l - PAD.r);
          const span = dragDom[1] - dragDom[0];
          this.view = { min: dragDom[0] - dpx * span, max: dragDom[1] - dpx * span };
          this.drawStatic();
        } else {
          const rect = cv.getBoundingClientRect();
          this._hover(e.clientX - rect.left, e.clientY - rect.top);
        }
      });
      cv.addEventListener("pointerleave", () => { this.gOverlay.clear(); clearKids(this.tip); });
      cv.addEventListener("dblclick", () => { this.view = null; this.drawStatic(); if (this.onReset) this.onReset(); });
      cv.title = "scroll: zoom-x · drag: pan · dbl-click: reset";
    }

    _hover(mouseX, mouseY) {
      if (!this.scales) return;
      const { xmin, xmax, xpx } = this.scales;
      const W = this.app.screen.width, H = this.app.screen.height;
      if (mouseX < PAD.l || mouseX > W - PAD.r) { this.gOverlay.clear(); clearKids(this.tip); return; }
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
          const box = new PIXI.Text({ text: txt, style: { fill: TXT, fontSize: 14, fontFamily: "monospace", lineHeight: 16 } });
          const bx = Math.min(best.x + 10, W - box.width - 8), by = PAD.t + 6;
          const bg = new PIXI.Graphics();
          bg.roundRect(bx - 5, by - 4, box.width + 10, box.height + 8, 4).fill({ color: INK, alpha: 0.96 }).stroke({ width: 1, color: best.color, alpha: 0.7 });
          box.x = bx; box.y = by; this.tip.addChild(bg, box); return;
        }
      }
      let step = xmin + (xmax - xmin) * (mouseX - PAD.l) / (W - PAD.l - PAD.r);
      // snap to the nearest marked (eval) point so a sparse value is easy to land on
      const _snap = this.spec.series.find(sp => sp.labels || sp.points);
      if (_snap) { const ss = this.data[_snap.src]; if (ss && ss.step && ss.step.length) step = ss.step[nearestIdx(ss.step, step)]; }
      this.gOverlay.clear(); clearKids(this.tip);
      this.gOverlay.moveTo(mouseX, PAD.t).lineTo(mouseX, H - PAD.b).stroke({ width: 1, color: 0x4a5663, alpha: 0.8 });

      const lines = [`step ${Math.round(step).toLocaleString()}`];
      for (const sp of this.spec.series) {
        if (sp.horizontal || this.hidden.has(sp.label)) continue;
        const s = this.data[sp.src];
        if (!s || !s.cols[sp.key] || !s.step.length) continue;
        const i = nearestIdx(s.step, step);
        const v = s.cols[sp.key][i];
        if (v == null || !isFinite(v)) continue;
        const scale = this.scales.norm ? (this._normScales && this._normScales[sp.key]) : (sp.axis === "y1" ? this.scales.sy1 : this.scales.sy);
        if (scale) { this.gOverlay.circle(xpx(s.step[i]), scale.px(v), 7).stroke({ width: 2.5, color: sp.color }); }
        lines.push(`${sp.label}: ${fmtNum(v)}`);
      }
      const box = new PIXI.Text({ text: lines.join("\n"), style: { fill: TXT, fontSize: 15.6, fontFamily: "monospace", lineHeight: 17 } });
      const bx = Math.min(mouseX + 10, W - box.width - 8), by = PAD.t + 6;
      const bg = new PIXI.Graphics();
      bg.roundRect(bx - 5, by - 4, box.width + 10, box.height + 8, 4).fill({ color: INK, alpha: 0.9 }).stroke({ width: 1, color: 0x2a323b });
      box.x = bx; box.y = by;
      this.tip.addChild(bg, box);
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
        { src: "eval", key: "ppl", label: "eval ppl", color: COL.ppl, axis: "y1", type: "line", points: true, r: 6.5, width: 2.6, labels: true, maxLabels: 24, labelFmt: v => v.toFixed(2) },
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
    "rosa_inj_rms", "engram_inj_rms", "rosa_e_gap", "wnorm_rms", "stable_rank"];
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

  let customGen = 0;
  async function applyCustom() {
    const box = document.getElementById("custom-chartbox");
    if (!customChart) return;
    const sel = [...selectedMetrics];
    if (!sel.length || !curRun) { if (box) box.style.display = "none"; return; }
    if (box) box.style.display = "";            // show as soon as there's a selection
    const gen = ++customGen, run = curRun;
    const trainF = sel.filter(m => m.startsWith("train:")).map(m => m.slice(6));
    const evalF = sel.filter(m => m.startsWith("eval:")).map(m => m.slice(5));
    const url = `/api/series/${encodeURIComponent(run)}?train=${trainF.join(",")}&eval=${evalF.join(",")}`;
    const data = await fetch(url).then(r => r.json()).catch(() => null);
    if (!data || gen !== customGen) return;     // stale (newer selection or run switch)
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
  }

  function persistCompare(name) { try { localStorage.setItem("tb_compare", name || ""); } catch (e) {} }
  function loadCompare() { try { return localStorage.getItem("tb_compare") || ""; } catch (e) { return ""; } }
  async function setCompareRun(name) {
    persistCompare(name);
    if (!name || name === curRun) { for (const id in charts) charts[id].setCompare(null); return; }
    const url = `/api/series/${encodeURIComponent(name)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}`;
    const data = await fetch(url).then(r => r.json()).catch(() => null);
    if (!data) return;
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
    const cat = await fetch(`/api/metrics/${encodeURIComponent(run)}`).then(r => r.json()).catch(() => null);
    if (!cat) return;
    renderCatalog(cat);
    applyCustom();
  }
  function redrawAll() { for (const id in charts) charts[id].drawStatic(); if (customChart) customChart.drawStatic(); }
  function focusStep(step) {
    for (const id in charts) { charts[id].view = { min: Math.max(0, step - 400), max: step + 400 }; charts[id].drawStatic(); }
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
  }
  let curRun = null, lastStep = 0, loading = false;
  let zoomTimer = null, zoomGen = 0;
  function syncView(view) {
    for (const id in charts) {
      if (id === "chart-custom") continue;          // custom chart manages its own data
      charts[id].view = view ? { min: view.min, max: view.max } : null;
      charts[id].drawStatic();
    }
  }
  function onChartZoom(view) {
    syncView(view);                                  // instant zoom on the decimated data
    const gen = ++zoomGen;
    clearTimeout(zoomTimer);
    zoomTimer = setTimeout(async () => {
      if (!curRun) return;
      const from = Math.max(0, Math.floor(view.min)), to = Math.ceil(view.max);
      if (to - from < 2) return;
      const url = `/api/series/${encodeURIComponent(curRun)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}&from=${from}&to=${to}`;
      const data = await fetch(url).then(r => r.json()).catch(() => null);
      if (!data || gen !== zoomGen) return;          // stale (newer zoom or run switch)
      for (const id in charts) { if (id !== "chart-custom") charts[id].setWindow(data); }
    }, 260);
  }
  function onChartReset() { zoomGen++; for (const id in charts) { if (id !== "chart-custom") charts[id].restoreFull(); } }

  function distribute(chart, data) {
    // give each chart the full payload; it reads only the cols it needs
    return data;
  }

  async function loadRun(run) {
    loading = true;
    const url = `/api/series/${encodeURIComponent(run)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}`;
    const [data, tl] = await Promise.all([
      fetch(url).then(r => r.json()).catch(() => null),
      fetch(`/api/timeline/${encodeURIComponent(run)}`).then(r => r.json()).catch(() => ({ events: [] })),
    ]);
    loading = false;
    if (!data) return;
    curRun = run; lastStep = data.max_step || 0;
    const events = (tl && tl.events) || [];
    for (const spec of CHARTS) {
      const ch = charts[spec.id];
      if (!ch) continue;
      ch.timeline = events;
      togglePanel(spec, data);
      ch.setData(data);
    }
    renderEventList(events);
    saveRun(run);
    loadCatalog(run);
    populateCompareOptions();
    const cmp = loadCompare();
    const cmpSel = document.getElementById("compare-run");
    if (cmp && cmp !== run) { if (cmpSel) cmpSel.value = cmp; setCompareRun(cmp); }
    else { if (cmpSel) cmpSel.value = ""; for (const id in charts) charts[id].setCompare(null); }
  }

  async function appendRun(run, version) {
    if (version <= lastStep || loading) return;
    loading = true;
    const url = `/api/series/${encodeURIComponent(run)}?train=${TRAIN_FIELDS.join(",")}&eval=${EVAL_FIELDS.join(",")}&since=${lastStep}`;
    const data = await fetch(url).then(r => r.json()).catch(() => null);
    loading = false;
    if (!data) return;
    lastStep = Math.max(lastStep, data.max_step || 0);
    for (const spec of CHARTS) {
      const ch = charts[spec.id];
      if (ch) ch.append(data);
    }
    fetch(`/api/timeline/${encodeURIComponent(run)}`).then(r => r.json()).then(tl => {
      const events = (tl && tl.events) || [];
      for (const id in charts) charts[id].setTimeline(events);
      renderEventList(events);
    }).catch(() => {});
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
    wireCompare();
    restoreRun();
    // React to (run, version) changes published by Datastar via #active-run.
    if (window.trainboard && window.trainboard.watchActiveRun) {
      window.trainboard.watchActiveRun((run, version) => {
        if (!run) return;
        if (run !== curRun) loadRun(run);
        else appendRun(run, version);
      });
    }
  }

  window.addEventListener("DOMContentLoaded", boot);
})();
