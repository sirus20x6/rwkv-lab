#!/usr/bin/env python3
"""Resumably recaption the Civitai and Joy slices with local Qwen3.6-VL.

The existing description is supplied as an untrusted visual hint.  Results are
appended one record at a time so an interrupted multi-hour run resumes without
redoing completed images.  vLLM performs the actual dynamic batching; this
client keeps several independent requests in flight.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "curated_vision/captioning_first_civitai_joy_train.jsonl",
    ROOT / "curated_vision/captioning_first_civitai_joy_eval.jsonl",
)
DEFAULT_OUTPUT = Path(
    "/thearray/downloads/cache/moe-mla/qwen36_civitai_joy_recaption/recaptions.jsonl")
DEFAULT_ENDPOINT = "http://127.0.0.1:9080/v1/chat/completions"
DEFAULT_MODEL = "qwen3.6-35b-a3b-heretic"
SOURCE_NAMES = frozenset(("captioning_civitai", "captioning_joy"))
PROMPT_VERSION = "qwen36_grounded_natural_same_language_prefix_first_v2"


SYSTEM_INSTRUCTION = """Study the supplied image carefully, then write a detailed, vivid, natural description of what is visibly present. Include the subjects, their appearance, actions, expressions and relationships; the setting, composition, lighting, viewpoint, and visual style when relevant. Preserve useful fine detail, but do not pad the answer with generic praise, symbolism, quality tags, or image-generation language.

Use the same primary natural language as the supplied existing description. Do not translate it into English when it is written in another language. Write fluidly in that language, with an engaging human tone; do not sound overly clinical, sterile, detached, or dry. Describe adult or sensitive visible content plainly and naturally when present, without euphemizing, moralizing, or adding safety commentary.

The existing description is an UNTRUSTED hint, not ground truth. It may be incomplete, tag-like, awkward, or wrong. The pixels are authoritative: independently verify every hinted object, count, attribute, action, identity, and spatial relationship, and omit anything the image does not support. Do not invent names or facts that cannot be seen.

Return only the finished image description, with no heading, analysis, preamble, labels, or commentary."""


def caption_instruction(existing: str) -> str:
    """Build the per-image hint; the constant rules live in the system prefix."""
    return f"""Here is the existing description / prompt hint for the image that follows:

<existing_description>
{existing}
</existing_description>

Now inspect the following image and produce the requested description."""


def caption_messages(existing: str, image_url: str) -> list[dict]:
    """Put reusable text first, then the unique hint, then the unique image."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption_instruction(existing)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def stable_job_id(source: str, image: Path, existing: str) -> str:
    # Prompt-versioning the identity prevents a prior prompt's completed row
    # from silently satisfying a newer captioning policy.
    material = f"{PROMPT_VERSION}\0{source}\0{image.resolve()}\0{existing}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class Job:
    job_id: str
    split: str
    source: str
    image: Path
    existing_caption: str
    input_row: int


def iter_jobs(inputs: Iterable[Path], sources: frozenset[str] = SOURCE_NAMES) -> Iterator[Job]:
    seen: set[str] = set()
    for manifest in inputs:
        split = manifest.stem.rsplit("_", 1)[-1]
        with manifest.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, 1):
                row = json.loads(line)
                source = str(row.get("stage1_source", ""))
                if source not in sources:
                    continue
                existing = str(row.get("text", "")).strip()
                image = Path(str(row.get("image", ""))).expanduser()
                if not image.is_absolute():
                    image = ROOT / image
                job_id = stable_job_id(source, image, existing)
                if job_id in seen:
                    continue
                seen.add(job_id)
                yield Job(job_id, split, source, image, existing, row_number)


def completed_job_ids(output: Path) -> set[str]:
    completed: set[str] = set()
    if not output.exists():
        return completed
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"corrupt resume journal {output}:{line_number}: {error}") from error
            if row.get("status") == "ok" and row.get("caption"):
                completed.add(str(row["job_id"]))
    return completed


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class ThreadSessions:
    def __init__(self) -> None:
        self.local = threading.local()

    def get(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            self.local.session = session
        return session


def caption_job(
    job: Job,
    *,
    endpoint: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int,
    sessions: ThreadSessions,
) -> dict:
    started = time.perf_counter()
    base_record = {
        "schema": 1,
        "job_id": job.job_id,
        "split": job.split,
        "source": job.source,
        "image": str(job.image),
        "existing_caption": job.existing_caption,
        "input_row": job.input_row,
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    if not job.image.is_file():
        return {**base_record, "status": "error", "error": "image_not_found"}
    payload = {
        "model": model,
        "messages": caption_messages(job.existing_caption, image_data_url(job.image)),
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "seed": 20260718,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    error = "unknown_error"
    for attempt in range(1, retries + 1):
        try:
            response = sessions.get().post(endpoint, json=payload, timeout=timeout)
            if response.status_code != 200:
                error = f"http_{response.status_code}: {response.text[:1000]}"
                if response.status_code < 500 and response.status_code != 429:
                    break
            else:
                body = response.json()
                choice = body["choices"][0]
                caption = str(choice["message"].get("content") or "").strip()
                if not caption:
                    error = "empty_caption"
                else:
                    return {
                        **base_record,
                        "status": "ok",
                        "caption": caption,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": body.get("usage", {}),
                        "attempts": attempt,
                        "latency_seconds": time.perf_counter() - started,
                    }
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 20))
    return {
        **base_record,
        "status": "error",
        "error": error,
        "attempts": retries,
        "latency_seconds": time.perf_counter() - started,
    }


def server_preflight(endpoint: str, expected_model: str, timeout: float) -> None:
    models_endpoint = endpoint.rsplit("/chat/completions", 1)[0] + "/models"
    try:
        response = requests.get(models_endpoint, timeout=min(timeout, 30))
        response.raise_for_status()
        model_ids = {str(row["id"]) for row in response.json().get("data", [])}
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise SystemExit(f"vLLM preflight failed at {models_endpoint}: {exc}") from exc
    if expected_model not in model_ids:
        raise SystemExit(
            f"vLLM is serving {sorted(model_ids)}, not requested model {expected_model!r}")


def append_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sources", nargs="+", default=sorted(SOURCE_NAMES))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats-every", type=int, default=25)
    args = parser.parse_args()

    inputs = tuple(args.inputs or DEFAULT_INPUTS)
    for manifest in inputs:
        if not manifest.is_file():
            raise SystemExit(f"missing input manifest: {manifest}")
    if args.concurrency < 1 or args.max_tokens < 1 or args.retries < 1:
        raise SystemExit("concurrency, max-tokens, and retries must be positive")

    completed = set() if args.force else completed_job_ids(args.output)
    pending = [job for job in iter_jobs(inputs, frozenset(args.sources))
               if job.job_id not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    total_selected = len(completed) + len(pending)
    print(json.dumps({
        "phase": "inventory",
        "selected": total_selected,
        "already_completed": len(completed),
        "pending": len(pending),
        "sources": args.sources,
        "output": str(args.output),
        "prompt_version": PROMPT_VERSION,
    }), flush=True)
    if args.dry_run or not pending:
        return

    server_preflight(args.endpoint, args.model, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sessions = ThreadSessions()
    worker_args = dict(
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        sessions=sessions,
    )
    successes = failures = completion_tokens = 0
    started = time.perf_counter()
    pending_iter = iter(pending)
    in_flight: dict[Future, Job] = {}
    window = max(args.concurrency * 2, args.concurrency)
    with args.output.open("a", encoding="utf-8") as output_handle, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while len(in_flight) < window:
            try:
                job = next(pending_iter)
            except StopIteration:
                break
            in_flight[pool.submit(caption_job, job, **worker_args)] = job
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                job = in_flight.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # preserve the queue on unexpected worker bugs
                    record = {
                        "schema": 1,
                        "job_id": job.job_id,
                        "split": job.split,
                        "source": job.source,
                        "image": str(job.image),
                        "existing_caption": job.existing_caption,
                        "model": args.model,
                        "prompt_version": PROMPT_VERSION,
                        "status": "error",
                        "error": f"worker_exception: {type(exc).__name__}: {exc}",
                    }
                append_record(output_handle, record)
                if record["status"] == "ok":
                    successes += 1
                    completion_tokens += int(record.get("usage", {}).get("completion_tokens", 0))
                else:
                    failures += 1
                try:
                    next_job = next(pending_iter)
                except StopIteration:
                    continue
                in_flight[pool.submit(caption_job, next_job, **worker_args)] = next_job
            done_count = successes + failures
            if done_count % args.stats_every < len(done):
                os.fsync(output_handle.fileno())
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "phase": "captioning",
                    "completed_this_run": done_count,
                    "successes": successes,
                    "failures": failures,
                    "remaining": len(pending) - done_count,
                    "captions_per_second": successes / elapsed,
                    "completion_tokens_per_second": completion_tokens / elapsed,
                    "elapsed_seconds": elapsed,
                }), flush=True)

        os.fsync(output_handle.fileno())
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "phase": "complete",
        "successes": successes,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "captions_per_second": successes / elapsed if elapsed else 0,
        "output": str(args.output),
    }), flush=True)
    if failures:
        print("failed records remain retryable on the next invocation", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
