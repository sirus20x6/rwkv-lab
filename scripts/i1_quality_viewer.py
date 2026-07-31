#!/usr/bin/env python3
"""Fast local review UI for the scored I1 dataset.

Deletion is recoverable: the image is moved below ``review_trash/images``, an
append-only audit record is fsynced, and a background worker atomically removes
the row from metadata.jsonl and train.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sqlite3
import threading
import time
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>I1 quality review</title>
<style>
:root { color-scheme:dark; --bg:#0d0f13; --panel:#171a21; --line:#292e39;
  --text:#eceff4; --muted:#9ca6b7; --accent:#70a5ff; --danger:#ff627d }
* { box-sizing:border-box } html,body { width:100%; height:100%; overflow:hidden }
body { margin:0; background:var(--bg); color:var(--text); font:13px/1.35 system-ui,sans-serif;
  display:grid; grid-template-rows:auto minmax(0,1fr) auto }
header { z-index:5; padding:7px 12px; background:#0d0f13f2;
  border-bottom:1px solid var(--line) }.top { display:flex; gap:9px; align-items:center;
  white-space:nowrap } h1 { font-size:16px; margin:0 auto 0 0 }
label { color:var(--muted) } select,input,button { color:var(--text); background:var(--panel);
  border:1px solid var(--line); border-radius:6px; padding:5px 7px }
button { cursor:pointer } button:hover { border-color:var(--accent) }
#status { color:var(--muted); font-size:11px; margin-left:4px }
main { min-height:0; padding:9px; display:grid; gap:9px; overflow:hidden }
.card { min-width:0; min-height:0; border:1px solid var(--line);
  border-radius:9px; overflow:hidden; display:grid;
  grid-template-rows:minmax(0,1fr) 146px; background:var(--panel) }
.media { min-height:0; position:relative; background:#08090c; touch-action:none;
  user-select:none; cursor:zoom-in }.media img { width:100%; height:100%; object-fit:contain;
  display:block; pointer-events:none }
.score { position:absolute; top:7px; right:7px; padding:4px 7px; border-radius:18px;
  background:#080a0de8; font-weight:700; font-variant-numeric:tabular-nums }
.hold-tip { position:absolute; left:7px; bottom:7px; padding:3px 6px; border-radius:5px;
  color:#c9d1de; background:#080a0dba; font-size:10px; opacity:.76 }
.body { min-height:0; padding:7px; display:grid; grid-template-rows:23px minmax(0,1fr) 27px;
  gap:5px }.meta { color:var(--muted); font-size:11px; display:flex; gap:5px;
  align-items:center; overflow:hidden; white-space:nowrap }.tag { border:1px solid var(--line);
  border-radius:15px; padding:2px 6px; overflow:hidden; text-overflow:ellipsis }
.caption-box { min-height:0; display:grid; grid-template-columns:27px minmax(0,1fr) 27px;
  border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#11141a }
.caption-arrow { border:0; border-radius:0; padding:0; font-size:18px; background:#1a1e27 }
.caption-center { min-width:0; min-height:0; padding:5px 7px; display:grid;
  grid-template-rows:17px minmax(0,1fr) }.caption-head { color:var(--muted); font-size:10px;
  font-weight:700; text-transform:uppercase; letter-spacing:.04em; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap }.caption-text { white-space:pre-wrap;
  overflow:auto; overflow-wrap:anywhere; font-size:12px; scrollbar-width:thin }
.selected { color:#9ee6ae }.actions { display:flex; align-items:center; gap:7px }
.danger { margin-left:auto; color:#fff; background:#6d2030; border-color:#9e3148;
  padding:3px 8px }.danger:hover { background:#8b2940; border-color:var(--danger) }
.caption-page { color:var(--muted); font-variant-numeric:tabular-nums }
footer { min-height:36px; padding:4px 10px 6px; display:flex; align-items:center;
  justify-content:center; gap:9px; border-top:1px solid var(--line) }
.empty { grid-column:1/-1; display:grid; place-items:center; color:var(--muted) }
@media(max-width:700px){h1{display:none}.top{gap:4px}header{padding:5px}
  label{font-size:10px}.card{grid-template-rows:minmax(0,1fr) 140px}}
</style>
</head>
<body>
<header id="header"><div class="top">
 <h1>I1 quality review</h1>
 <label>Sort <select id="sort"><option value="score_desc">quality high → low</option>
  <option value="score_asc">quality low → high</option>
  <option value="native">dataset order</option></select></label>
 <label>Min <input id="min" type="number" min="0" max="1" step=".01" value="0"
  style="width:62px"></label><span id="status">Loading…</span>
</div></header>
<main id="grid"></main>
<footer id="footer"><button id="prev">← Previous</button><span id="page"></span>
 <button id="next">Next →</button></footer>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;',
  '>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let offset=0,total=0,loading=false,pageLimit=1,columns=1,rows=1,resizeTimer;
const items=new Map(), captionPositions=new Map();
function fitGrid(){
 const grid=$('grid'), gap=9, width=grid.clientWidth-18, height=grid.clientHeight-18;
 const minWidth=innerWidth>=2500?420:390, minHeight=420;
 columns=Math.max(1,Math.floor((width+gap)/(minWidth+gap)));
 rows=Math.max(1,Math.floor((height+gap)/(minHeight+gap))-1);
 pageLimit=Math.min(96,columns*rows);
 grid.style.gridTemplateColumns=`repeat(${columns},minmax(0,1fr))`;
 grid.style.gridTemplateRows=`repeat(${rows},minmax(0,1fr))`;
}
function query(){
 const p=new URLSearchParams({offset,limit:pageLimit,
  sort:$('sort').value,min_score:$('min').value||0});
 return p.toString();
}
function captionEntries(x){
 return [{name:`training · ${x.caption_variant}`,text:x.caption,selected:true,training:true},
  ...x.captions.map(c=>({name:c.name,text:c.text,selected:c.selected,training:false}))];
}
function captionAt(x,index){
 const entries=captionEntries(x), position=((index%entries.length)+entries.length)%entries.length;
 return {entry:entries[position],position,total:entries.length};
}
async function load(){
 if(loading)return; fitGrid(); loading=true; $('status').textContent='Loading…';
 try{
  const r=await fetch('/api/items?'+query()); if(!r.ok)throw Error(await r.text());
  const d=await r.json(); total=d.total; items.clear(); captionPositions.clear();
  d.items.forEach(x=>items.set(x.key,x));
  $('grid').innerHTML=d.items.length?d.items.map(card).join(''):
    '<div class="empty">No images match these filters.</div>';
  const first=total?offset+1:0,last=Math.min(offset+pageLimit,total);
  $('page').textContent=`${first.toLocaleString()}–${last.toLocaleString()} of ${total.toLocaleString()} · ${columns}×${rows}`;
  $('prev').disabled=offset===0; $('next').disabled=offset+pageLimit>=total;
  $('status').textContent=`${d.active.toLocaleString()} active · ${d.deleted.toLocaleString()} deleted`;
  pollStatus();
 }catch(e){$('status').textContent='Error: '+e.message}finally{loading=false}
}
function card(x){
 const current=captionAt(x,0),c=current.entry;
 return `<article class="card" id="card-${esc(x.key)}">
  <div class="media" onclick="window.open('/image/${encodeURIComponent(x.key)}','_blank')">
   <img loading="lazy" src="/thumb/${encodeURIComponent(x.key)}" alt="">
   <span class="score">${x.score.toFixed(3)} · ${esc(x.label)}</span>
   <span class="hold-tip">click: original</span></div>
  <div class="body"><div class="meta">
   <span class="tag">${x.width}×${x.height}</span><span class="tag">${esc(x.subset)}</span>
  </div><div class="caption-box">
   <button class="caption-arrow" onclick="cycleCaption('${esc(x.key)}',-1)">‹</button>
   <div class="caption-center"><div class="caption-head ${c.selected?'selected':''}">${esc(c.name)}</div>
    <div class="caption-text">${esc(c.text)}</div></div>
   <button class="caption-arrow" onclick="cycleCaption('${esc(x.key)}',1)">›</button>
  </div><div class="actions"><span class="caption-page">1/${current.total}</span>
   <button class="danger" onclick="removeItem('${esc(x.key)}')">Delete</button>
  </div></div></article>`;
}
function cycleCaption(key,direction){
 const x=items.get(key); if(!x)return;
 const next=(captionPositions.get(key)||0)+direction; captionPositions.set(key,next);
 const current=captionAt(x,next),card=document.getElementById('card-'+key);
 const head=card.querySelector('.caption-head'),text=card.querySelector('.caption-text');
 head.textContent=current.entry.name; head.classList.toggle('selected',current.entry.selected);
 text.textContent=current.entry.text;
 card.querySelector('.caption-page').textContent=`${current.position+1}/${current.total}`;
}
async function removeItem(key){
 const card=document.getElementById('card-'+key),button=card.querySelector('.danger');
 button.disabled=true;
 const r=await fetch('/api/delete/'+encodeURIComponent(key),{method:'POST'});
 if(!r.ok){alert(await r.text());button.disabled=false;return}
 card.remove(); total--; items.delete(key); pollStatus(); setTimeout(load,90);
}
async function pollStatus(){
 try{const d=await (await fetch('/api/status')).json();
  if(d.compacting)$('status').textContent=`Deletions queued/syncing · ${d.message||''}`;
  else $('status').textContent=`${d.active.toLocaleString()} active · ${d.deleted.toLocaleString()} deleted · manifests synced`;
 }catch{}
}
for(const id of ['sort','min'])$(id).addEventListener('change',()=>{offset=0;load()});
$('prev').onclick=()=>{offset=Math.max(0,offset-pageLimit);load()};
$('next').onclick=()=>{offset+=pageLimit;load()};
addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{offset=0;load()},180)});
fitGrid();load();setInterval(pollStatus,3000);
</script></body></html>"""


@dataclass(frozen=True)
class Item:
    key: str
    image: Path
    i1_key: str
    caption: str
    caption_variant: str
    score: float
    label: str
    confidence: float
    width: int
    height: int
    subset: str

    def payload(self, captions: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "key": self.key,
            "caption": self.caption,
            "caption_variant": self.caption_variant,
            "captions": captions,
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
            "subset": self.subset,
        }


def caption_order(name: str) -> tuple[int, int, str]:
    if name.startswith("caption") and name.removeprefix("caption").isdigit():
        return 0, int(name.removeprefix("caption")), name
    qwen_order = {
        "qwen2vl_2b": 0,
        "qwen2.5vl_3b": 1,
        "qwen3vl_2b": 2,
        "qwen3vl_4b": 3,
    }
    if name in qwen_order:
        return 1, qwen_order[name], name
    if name == "short":
        return 2, 0, name
    if name == "no_center_crop":
        return 3, 0, name
    return 4, 0, name


class CaptionStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise RuntimeError(
                f"caption cache is missing: {self.path}; run "
                "scripts/build_i1_quality_caption_cache.py"
            )
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            schema = connection.execute(
                "SELECT value FROM cache_metadata WHERE name='schema_version'"
            ).fetchone()
            if schema != ("1",):
                raise RuntimeError(f"unsupported caption cache schema: {schema}")
        finally:
            connection.close()
        self.local = threading.local()

    def connection(self) -> sqlite3.Connection:
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
            )
            self.local.connection = connection
        return connection

    def variants(self, item: Item) -> list[dict[str, Any]]:
        rows = self.connection().execute(
            """
            SELECT name, text
            FROM caption_variants
            WHERE subset=? AND i1_key=?
            """,
            (item.subset, item.i1_key),
        ).fetchall()
        rows.sort(key=lambda row: caption_order(str(row[0])))
        return [
            {
                "name": str(name),
                "text": str(text),
                "selected": name == item.caption_variant,
            }
            for name, text in rows
        ]


def image_key(row: dict[str, Any]) -> str:
    value = row.get("image_sha256") or row.get("image_id")
    if isinstance(value, str) and value:
        return value
    image = row.get("image")
    if not isinstance(image, str) or not image:
        raise RuntimeError("row has no image identity")
    return hashlib.sha256(image.encode()).hexdigest()


def write_filtered_manifest(
    source: Path,
    deleted: set[str],
) -> tuple[Path, int, int]:
    temporary = source.with_name(
        f".{source.name}.quality-viewer.{os.getpid()}.tmp"
    )
    kept = removed = 0
    try:
        with (
            source.open("r", encoding="utf-8") as input_file,
            temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as output,
        ):
            for line_no, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{source}:{line_no}: malformed JSON") from exc
                key = image_key(row)
                if key in deleted:
                    removed += 1
                    continue
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, kept, removed


class Dataset:
    def __init__(self, root: Path, caption_cache: Path | None = None) -> None:
        self.root = root.resolve()
        self.images_root = (self.root / "images").resolve()
        self.trash_root = (self.root / "review_trash" / "images").resolve()
        self.work = self.root / "work"
        self.thumb_root = self.work / "quality_viewer_thumbs"
        self.ledger = self.work / "quality_viewer_deletions.jsonl"
        self.metadata = self.root / "metadata.jsonl"
        self.train = self.root / "train.jsonl"
        self.caption_store = CaptionStore(
            caption_cache
            if caption_cache is not None
            else self.work / "quality_viewer_captions.sqlite3"
        )
        self.lock = threading.RLock()
        self.items: dict[str, Item] = {}
        self.deleted = self._load_deleted()
        self.generation = 0
        self.compacted_generation = 0
        self.compacting = False
        self.message = ""
        self.wake = threading.Event()
        self.manifests_need_sync = False
        self._load_items()
        self.order_cache: dict[str, tuple[str, ...]] = {
            "native": tuple(self.items)
        }
        self.worker = threading.Thread(target=self._compaction_worker, daemon=True)
        self.worker.start()
        if self.manifests_need_sync:
            self.generation = 1
            self.wake.set()

    def _load_deleted(self) -> set[str]:
        deleted: set[str] = set()
        if not self.ledger.exists():
            return deleted
        with self.ledger.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if any(rest.strip() for rest in handle):
                        raise RuntimeError(
                            f"{self.ledger}:{line_no}: malformed non-final line"
                        )
                    break
                if row.get("action") == "delete" and isinstance(row.get("key"), str):
                    deleted.add(row["key"])
        return deleted

    def _load_items(self) -> None:
        with self.metadata.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                key = image_key(row)
                if key in self.deleted:
                    self.manifests_need_sync = True
                    continue
                image = Path(str(row["image"])).resolve()
                try:
                    image.relative_to(self.images_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"{self.metadata}:{line_no}: image outside dataset root"
                    ) from exc
                score = row.get("deepghs_anime_aesthetic_score")
                if not isinstance(score, (int, float)):
                    raise RuntimeError(
                        f"{self.metadata}:{line_no}: missing aesthetic score"
                    )
                caption = row.get("caption") or row.get("text") or ""
                self.items[key] = Item(
                    key=key,
                    image=image,
                    i1_key=str(row.get("i1_key", "")),
                    caption=str(caption),
                    caption_variant=str(row.get("caption_variant", "")),
                    score=float(score),
                    label=str(row.get("deepghs_anime_aesthetic_label", "")),
                    confidence=float(
                        row.get("deepghs_anime_aesthetic_confidence", 0.0)
                    ),
                    width=int(row.get("width", 0)),
                    height=int(row.get("height", 0)),
                    subset=str(row.get("i1_subset") or row.get("source") or ""),
                )

    def captions(self, item: Item) -> list[dict[str, Any]]:
        captions = self.caption_store.variants(item)
        if not captions:
            raise RuntimeError(
                f"caption cache has no entry for {item.subset}:{item.i1_key}"
            )
        return captions

    def query(
        self,
        *,
        offset: int,
        limit: int,
        sort: str,
        min_score: float,
    ) -> tuple[list[Item], int]:
        with self.lock:
            order = self.order_cache.get(sort)
            if order is None:
                sort_key = (
                    (lambda item: (-item.score, item.key))
                    if sort == "score_desc"
                    else (lambda item: (item.score, item.key))
                )
                order = tuple(
                    item.key
                    for item in sorted(
                        self.items.values(),
                        key=sort_key,
                    )
                )
                self.order_cache[sort] = order
            selected: list[Item] = []
            matched = 0
            stop = offset + limit
            for key in order:
                item = self.items.get(key)
                if item is None:
                    continue
                if item.score < min_score:
                    continue
                if offset <= matched < stop:
                    selected.append(item)
                matched += 1
            return selected, matched

    def get(self, key: str) -> Item | None:
        with self.lock:
            return self.items.get(key)

    def delete(self, key: str) -> dict[str, Any]:
        with self.lock:
            item = self.items.get(key)
            if item is None:
                raise KeyError(key)
            relative = item.image.relative_to(self.images_root)
            trash = self.trash_root / relative
            trash.parent.mkdir(parents=True, exist_ok=True)
            if trash.exists():
                raise FileExistsError(trash)
            os.replace(item.image, trash)
            record = {
                "schema_version": 1,
                "action": "delete",
                "created_unix": time.time(),
                "key": key,
                "original_image": str(item.image),
                "trash_image": str(trash),
                "caption": item.caption,
                "score": item.score,
            }
            try:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                with self.ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                os.replace(trash, item.image)
                raise
            self.deleted.add(key)
            del self.items[key]
            self.generation += 1
            self.wake.set()
            return record

    def thumbnail(self, item: Item) -> Path:
        target = self.thumb_root / item.key[:2] / f"{item.key}.webp"
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{item.key}.{threading.get_ident()}.tmp.webp"
        )
        with Image.open(item.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            image.save(temporary, "WEBP", quality=84, method=4)
        os.replace(temporary, target)
        return target

    def _ensure_backups(self) -> None:
        backup_root = self.work / "pre_quality_viewer_manifests"
        for source in (self.metadata, self.train):
            backup = backup_root / source.name
            if backup.exists():
                continue
            backup.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, backup)
            except OSError:
                shutil.copy2(source, backup)

    def _compaction_worker(self) -> None:
        while True:
            self.wake.wait()
            self.wake.clear()
            # Human review produces bursts. Wait for a quiet interval so a
            # million-row manifest is not rewritten after every click.
            while self.wake.wait(timeout=5.0):
                self.wake.clear()
            with self.lock:
                target_generation = self.generation
                deleted = set(self.deleted)
                self.compacting = True
                self.message = f"{len(deleted):,} deletion(s)"
            try:
                self._ensure_backups()
                written: list[tuple[Path, Path, int, int]] = []
                for source in (self.metadata, self.train):
                    temporary, kept, removed = write_filtered_manifest(
                        source, deleted
                    )
                    written.append((source, temporary, kept, removed))
                results = [
                    (kept, removed) for _, _, kept, removed in written
                ]
                if results[0] != results[1]:
                    raise RuntimeError(
                        "metadata/train compaction row counts do not match"
                    )
                for source, temporary, _, _ in written:
                    os.replace(temporary, source)
                metadata_result = results[0]
                with self.lock:
                    self.compacted_generation = target_generation
                    self.message = (
                        f"{metadata_result[0]:,} rows active; "
                        f"{metadata_result[1]:,} removed this pass"
                    )
            except Exception as exc:
                for _, temporary, _, _ in locals().get("written", []):
                    temporary.unlink(missing_ok=True)
                with self.lock:
                    self.message = f"compaction failed: {exc}"
            finally:
                with self.lock:
                    self.compacting = False
                    if self.compacted_generation < self.generation:
                        self.wake.set()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active": len(self.items),
                "deleted": len(self.deleted),
                "compacting": self.compacting
                or self.compacted_generation < self.generation,
                "message": self.message,
            }


class Handler(BaseHTTPRequestHandler):
    dataset: Dataset

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, *, cache: bool = False) -> None:
        try:
            stat = path.stat()
            handle = path.open("rb")
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        with handle:
            shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            data = PAGE.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/status":
            self.send_json(self.dataset.status())
            return
        if parsed.path == "/api/items":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(96, max(1, int(query.get("limit", ["48"])[0])))
                min_score = min(
                    1.0, max(0.0, float(query.get("min_score", ["0"])[0]))
                )
                sort = query.get("sort", ["score_desc"])[0]
                if sort not in {"score_desc", "score_asc", "native"}:
                    raise ValueError("invalid sort")
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            items, total = self.dataset.query(
                offset=offset,
                limit=limit,
                sort=sort,
                min_score=min_score,
            )
            status = self.dataset.status()
            self.send_json(
                {
                    "items": [
                        item.payload(self.dataset.captions(item)) for item in items
                    ],
                    "total": total,
                    "active": status["active"],
                    "deleted": status["deleted"],
                }
            )
            return
        for prefix, thumbnail in (("/thumb/", True), ("/image/", False)):
            if parsed.path.startswith(prefix):
                key = urllib.parse.unquote(parsed.path[len(prefix) :])
                item = self.dataset.get(key)
                if item is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    path = self.dataset.thumbnail(item) if thumbnail else item.image
                    self.send_file(path, cache=thumbnail)
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/api/delete/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        key = urllib.parse.unquote(parsed.path.removeprefix("/api/delete/"))
        try:
            record = self.dataset.delete(key)
        except KeyError:
            self.send_json({"error": "unknown or already deleted key"}, HTTPStatus.NOT_FOUND)
            return
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_json({"ok": True, "deletion": record})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/thearray/git/datasets/i1"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--caption-cache",
        type=Path,
        help=(
            "SQLite cache made by build_i1_quality_caption_cache.py "
            "(defaults to ROOT/work/quality_viewer_captions.sqlite3)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Dataset(args.root, args.caption_cache)
    Handler.dataset = dataset
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"I1 quality viewer: http://127.0.0.1:{args.port} "
        f"({len(dataset.items):,} active images)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
