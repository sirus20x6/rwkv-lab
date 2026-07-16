#!/usr/bin/env python3
"""Compare grounded Qwen3-VL captions with and without generation hints.

The paired quality pass uses the eleven audited Midjourney-v6 images.  A
separate concurrency sweep uses disjoint images so vLLM's multimodal cache
cannot make later batch sizes look artificially faster.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pynvml
import requests


ROOT = Path(__file__).resolve().parents[1]
MODEL = "huihui-qwen3-vl-30b-a3b-instruct-abliterated"
IMAGE_DIR = ROOT / "datasets/i1_matched_images/midjourneyv6"
ITEMS = [
    ("1199_1.jpg", "A row of racer horses and people on motor bike parked in the background"),
    ("1789_3.jpg", "A person on it"),
    ("1557_0.jpg", "Orange grown in between scissors and a coffee are on a softball game"),
    ("604_3.jpg", "A bacon and sauce"),
    ("423_2.jpg", "A view from a roof"),
    ("98_1.jpg", "A man standing near cones and fences"),
    ("441_2.jpg", "A woman is standing on its surfaces"),
    ("1102_3.jpg", "A kitchen with an elephant in a green vase sitting on top of it"),
    ("1587_3.jpg", "A guy is seen in this photo"),
    ("1758_1.jpg", "A laptop computer on a horse in front of net"),
    ("228_0.jpg", "The woman is holding a beer sit on a table A man holds a multi-colored kite"),
]

BASE_INSTRUCTION = """Inspect the image carefully and write one detailed, natural caption describing only what is visibly present. Cover the primary subjects, their actions and spatial relationships, the setting, composition, lighting, and visual style when relevant. Aim for roughly 100 to 180 words, but prefer accuracy over length. Do not invent names, locations, text, objects, actions, or symbolism. If a detail is genuinely ambiguous, describe its visible appearance rather than guessing. Return only the caption, with no heading or commentary."""


def hinted_instruction(prompt: str) -> str:
    return BASE_INSTRUCTION + f"""

The source dataset also provides this original generation prompt as an UNTRUSTED weak hint:
{prompt}

The generated image may differ substantially from that prompt. Treat the pixels as authoritative. Evaluate every prompt detail independently: a garbled prompt may contain a correct object noun but incorrect actions or relationships. Use a hinted detail only when a corresponding visible feature supports it. Verify every object, action, count, and spatial relationship against the image before producing the final caption."""


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@dataclass(frozen=True)
class Job:
    image: Path
    condition: str
    prompt: str | None = None


class GpuSampler:
    def __init__(self, interval: float = 0.2) -> None:
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append({
                    "time": time.time(),
                    "gpu_util_pct": float(util.gpu),
                    "memory_used_gib": memory.used / 2**30,
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000,
                })
            except pynvml.NVMLError:
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            return {"samples": 0}
        return {
            "samples": len(self.samples),
            "mean_gpu_util_pct": statistics.fmean(x["gpu_util_pct"] for x in self.samples),
            "max_gpu_util_pct": max(x["gpu_util_pct"] for x in self.samples),
            "mean_memory_used_gib": statistics.fmean(x["memory_used_gib"] for x in self.samples),
            "max_memory_used_gib": max(x["memory_used_gib"] for x in self.samples),
            "mean_power_w": statistics.fmean(x["power_w"] for x in self.samples),
            "max_power_w": max(x["power_w"] for x in self.samples),
        }


def caption_job(job: Job, encoded: dict[Path, str], endpoint: str,
                max_tokens: int) -> dict:
    instruction = BASE_INSTRUCTION if job.condition == "image_only" else hinted_instruction(job.prompt or "")
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encoded[job.image]}},
                {"type": "text", "text": instruction},
            ],
        }],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 20260715,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=900)
    latency = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(
            f"{job.image.name}/{job.condition}: HTTP {response.status_code}: {response.text[:1000]}")
    body = response.json()
    choice = body["choices"][0]
    return {
        "image": str(job.image),
        "basename": job.image.name,
        "condition": job.condition,
        "source_prompt": job.prompt,
        "caption": choice["message"]["content"].strip(),
        "finish_reason": choice.get("finish_reason"),
        "latency_seconds": latency,
        "usage": body.get("usage", {}),
    }


def run_jobs(jobs: list[Job], encoded: dict[Path, str], endpoint: str,
             concurrency: int, max_tokens: int) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    with GpuSampler() as sampler:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(caption_job, job, encoded, endpoint, max_tokens)
                       for job in jobs]
            rows = [future.result() for future in as_completed(futures)]
    wall = time.perf_counter() - started
    total_completion = sum(int(row["usage"].get("completion_tokens", 0)) for row in rows)
    total_prompt = sum(int(row["usage"].get("prompt_tokens", 0)) for row in rows)
    summary = {
        "concurrency": concurrency,
        "requests": len(rows),
        "wall_seconds": wall,
        "captions_per_second": len(rows) / wall,
        "seconds_per_caption": wall / len(rows),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "completion_tokens_per_second": total_completion / wall,
        "mean_request_latency_seconds": statistics.fmean(
            row["latency_seconds"] for row in rows),
        "p95_request_latency_seconds": sorted(
            row["latency_seconds"] for row in rows)[max(0, round(0.95 * len(rows)) - 1)],
        "gpu": sampler.summary(),
    }
    return sorted(rows, key=lambda row: (row["basename"], row["condition"])), summary


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9080/v1/chat/completions")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/qwen_prompt_hint_audit_20260715")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--quality-concurrency", type=int, default=4)
    parser.add_argument("--benchmark-requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    quality_jobs: list[Job] = []
    for basename, prompt in ITEMS:
        image = IMAGE_DIR / basename
        if not image.is_file():
            raise SystemExit(f"missing quality image: {image}")
        quality_jobs += [Job(image, "image_only"), Job(image, "prompt_hint", prompt)]

    excluded = {job.image.resolve() for job in quality_jobs}
    candidates = [path for path in sorted(IMAGE_DIR.glob("*"))
                  if path.is_file() and path.resolve() not in excluded]
    needed = args.benchmark_requests * len(args.concurrency)
    random.Random(20260715).shuffle(candidates)
    if len(candidates) < needed:
        raise SystemExit(f"need {needed} disjoint benchmark images, found {len(candidates)}")
    benchmark_sets = {
        concurrency: candidates[i * args.benchmark_requests:(i + 1) * args.benchmark_requests]
        for i, concurrency in enumerate(args.concurrency)
    }

    all_images = {job.image for job in quality_jobs}
    all_images.update(path for paths in benchmark_sets.values() for path in paths)
    print(f"Preloading {len(all_images)} image payloads", flush=True)
    encoded = {path: data_url(path) for path in all_images}

    # This warmup image is excluded from every measured benchmark set.
    warmup_image = next(path for path in candidates[needed:] if path not in all_images)
    encoded[warmup_image] = data_url(warmup_image)
    print("Running one unmeasured end-to-end warmup caption", flush=True)
    warmup = caption_job(Job(warmup_image, "image_only"), encoded,
                         args.endpoint, min(args.max_tokens, 128))
    write_json(args.output / "warmup.json", warmup)

    print(f"Running 22-caption paired quality pass at concurrency {args.quality_concurrency}",
          flush=True)
    paired_rows, paired_summary = run_jobs(
        quality_jobs, encoded, args.endpoint, args.quality_concurrency, args.max_tokens)
    write_json(args.output / "paired_results.json", paired_rows)
    write_json(args.output / "paired_timing.json", paired_summary)
    for row in paired_rows:
        directory = args.output / row["condition"]
        directory.mkdir(exist_ok=True)
        (directory / f"{Path(row['basename']).stem}.txt").write_text(row["caption"] + "\n")

    benchmark = []
    for concurrency in args.concurrency:
        print(f"Benchmarking {args.benchmark_requests} unique images at concurrency {concurrency}",
              flush=True)
        jobs = [Job(path, "image_only") for path in benchmark_sets[concurrency]]
        rows, summary = run_jobs(
            jobs, encoded, args.endpoint, concurrency, args.max_tokens)
        benchmark.append({"summary": summary, "results": rows})
        write_json(args.output / f"benchmark_c{concurrency}.json", benchmark[-1])
        print(json.dumps(summary, indent=2), flush=True)
    write_json(args.output / "benchmark_all.json", benchmark)
    write_json(args.output / "summary.json", {
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "paired": paired_summary,
        "benchmark": [entry["summary"] for entry in benchmark],
        "benchmark_policy": "disjoint images across concurrency levels; one excluded warmup image",
    })


if __name__ == "__main__":
    main()
