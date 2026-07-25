#!/usr/bin/env python3
"""Local side-by-side pHash cutoff review UI."""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import sqlite3
import threading
import urllib.parse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path("/workspace/downloads/cache/moe-mla/local_private_web_image_dedup.sqlite")
DEFAULT_POLICY = ROOT / "curated_vision/local_private_web_dedupe_policy.json"


def bands(value: str, count: int = 8) -> list[tuple[int, int]]:
    number, bits = int(value, 16), len(value) * 4
    width, mask = bits // count, (1 << (bits // count)) - 1
    return [(band, (number >> (band * width)) & mask) for band in range(count)]


def sample_pairs(db_path: Path, scan_limit: int, per_distance: int,
                 seed: int) -> tuple[dict[int, list[dict]], int]:
    """Build bounded LSH samples without changing the production database."""
    rng = random.Random(seed)
    samples: dict[int, list[dict]] = defaultdict(list)
    seen = defaultdict(int)
    buckets: dict[tuple[int, int], list[tuple[str, int, int]]] = defaultdict(list)
    uri = f"file:{urllib.parse.quote(str(db_path))}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as db:
        rows = db.execute("""
            SELECT path,phash,colorhash FROM files
            WHERE excluded_reason IS NULL AND decode_error IS NULL
              AND duplicate_of IS NULL AND phash IS NOT NULL
            ORDER BY rowid LIMIT ?
        """, (scan_limit,))
        scanned = 0
        for path, phash, colorhash in rows:
            scanned += 1
            number, color = int(phash, 16), int(colorhash, 16)
            candidates: dict[str, tuple[int, int]] = {}
            pieces = bands(phash)
            for piece in pieces:
                for other, other_hash, other_color in buckets.get(piece, ()):
                    candidates[other] = (other_hash, other_color)
            for other, (other_hash, other_color) in candidates.items():
                distance = (number ^ other_hash).bit_count()
                color_distance = (color ^ other_color).bit_count()
                if distance > 7 or color_distance > 4:
                    continue
                seen[distance] += 1
                item = {"left": other, "right": path, "distance": distance,
                        "color_distance": color_distance}
                current = samples[distance]
                if len(current) < per_distance:
                    current.append(item)
                else:
                    replace = rng.randrange(seen[distance])
                    if replace < per_distance:
                        current[replace] = item
            for piece in pieces:
                bucket = buckets[piece]
                if len(bucket) < 4:
                    bucket.append((path, number, color))
                elif rng.random() < 0.002:
                    bucket[rng.randrange(len(bucket))] = (path, number, color)
    return dict(samples), scanned


def page(samples: dict[int, list[dict]], scanned: int,
         policy: dict | None) -> str:
    payload = json.dumps(samples).replace("</", "<\\/")
    selected = int((policy or {}).get("phash_distance", 4))
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>dedupe cutoff review</title><style>
body{{margin:0;background:#101318;color:#e8edf3;font:15px system-ui}}
header{{position:sticky;top:0;z-index:2;background:#171c24;padding:14px 22px;border-bottom:1px solid #303844}}
h1{{font-size:19px;margin:0 0 7px}} .sub{{color:#9ba8b7}}
.controls{{display:flex;gap:14px;align-items:center;margin-top:10px;flex-wrap:wrap}}
input[type=range]{{width:280px}} button{{background:#58a6ff;color:#08111c;border:0;border-radius:6px;padding:8px 14px;font-weight:700}}
#status{{color:#7ee787}} main{{padding:18px 22px}} section{{margin-bottom:30px}}
.distance{{font-size:17px;margin:0 0 9px}} .accepted{{color:#7ee787}} .rejected{{color:#ff9b9b}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}}
.pair{{background:#171c24;border:1px solid #303844;border-radius:8px;padding:8px}}
.images{{display:grid;grid-template-columns:1fr 1fr;gap:7px;height:320px}}
img{{width:100%;height:100%;object-fit:contain;background:#090b0f}}
.meta{{padding-top:7px;color:#9ba8b7;font:12px ui-monospace,monospace}}
</style></head><body><header><h1>local image dedupe cutoff</h1>
<div class=sub>Reviewing candidates from {scanned:,} hashed keepers. Production also requires color-hash distance ≤ 4.</div>
<div class=controls><label>mark pHash distances ≤ <b id=value>{selected}</b> as duplicates</label>
<input id=cutoff type=range min=0 max=7 value={selected} step=1>
<label>minimum side <input id=minside type=number min=1 value={(policy or {}).get('min_side',256)} style="width:75px"></label>
<button id=save>Save cutoff</button><span id=status>{'saved' if policy else 'not saved — clustering is paused'}</span></div></header>
<main id=main></main><script>
const samples={payload}, cutoff=document.getElementById('cutoff'), value=document.getElementById('value');
const esc=s=>s.replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function render(){{value.textContent=cutoff.value;let out='';for(let d=0;d<=7;d++){{const yes=d<=+cutoff.value;
 out+=`<section><h2 class="distance ${{yes?'accepted':'rejected'}}">distance ${{d}} · ${{yes?'DEDUPLICATE':'KEEP SEPARATE'}} · ${{(samples[d]||[]).length}} samples</h2><div class=grid>`;
 for(const [i,p] of (samples[d]||[]).entries()) out+=`<div class=pair><div class=images><img loading=lazy src="/image?d=${{d}}&i=${{i}}&side=left"><img loading=lazy src="/image?d=${{d}}&i=${{i}}&side=right"></div><div class=meta>pHash ${{p.distance}} · color ${{p.color_distance}}<br>${{esc(p.left)}}<br>${{esc(p.right)}}</div></div>`;
 out+='</div></section>';}}document.getElementById('main').innerHTML=out;}}
cutoff.oninput=render;document.getElementById('save').onclick=async()=>{{const r=await fetch('/policy',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{phash_distance:+cutoff.value,min_side:+document.getElementById('minside').value}})}});document.getElementById('status').textContent=r.ok?'saved — clustering will use this policy':'save failed';}};render();
</script></body></html>"""


class App:
    def __init__(self, samples: dict[int, list[dict]], scanned: int, policy: Path):
        self.samples, self.scanned, self.policy = samples, scanned, policy
        self.lock = threading.Lock()

    def handler(self):
        app = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/":
                    saved = json.loads(app.policy.read_text()) if app.policy.is_file() else None
                    data = page(app.samples, app.scanned, saved).encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
                if parsed.path == "/image":
                    try:
                        q = urllib.parse.parse_qs(parsed.query)
                        item = app.samples[int(q["d"][0])][int(q["i"][0])]
                        path = Path(item[q["side"][0]])
                        if q["side"][0] not in ("left", "right") or not path.is_file(): raise ValueError
                        data = path.read_bytes()
                    except (KeyError, IndexError, ValueError, OSError):
                        self.send_error(404); return
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "private, max-age=3600")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
                self.send_error(404)
            def do_POST(self):
                if self.path != "/policy": self.send_error(404); return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    value = json.loads(self.rfile.read(size))
                    distance, min_side = int(value["phash_distance"]), int(value["min_side"])
                    if not 0 <= distance <= 7 or min_side < 1: raise ValueError
                    payload = {"schema": 1, "phash_distance": distance,
                               "colorhash_distance": 4, "min_side": min_side,
                               "selected_by": "human_review"}
                    with app.lock:
                        temporary = app.policy.with_suffix(".tmp")
                        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                        os.replace(temporary, app.policy)
                except (KeyError, ValueError, json.JSONDecodeError):
                    self.send_error(400); return
                self.send_response(204); self.end_headers()
            def log_message(self, fmt, *args):
                print({"client": self.client_address[0], "message": fmt % args}, flush=True)
        return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9126)
    parser.add_argument("--scan-limit", type=int, default=250_000)
    parser.add_argument("--per-distance", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    samples, scanned = sample_pairs(args.db, args.scan_limit, args.per_distance, args.seed)
    print({"kind": "dedupe_review_ready", "scanned": scanned,
           "samples": {key: len(value) for key, value in samples.items()},
           "url": f"http://{args.host}:{args.port}"}, flush=True)
    app = App(samples, scanned, args.policy)
    ThreadingHTTPServer((args.host, args.port), app.handler()).serve_forever()


if __name__ == "__main__":
    main()
