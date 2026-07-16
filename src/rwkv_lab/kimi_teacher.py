"""Build and execute a resumable Kimi K2.6 image-caption teacher queue.

The queue builder deliberately uses the existing caption only for selection.
The caption request contains the image, not the old caption, so errors in the
old teacher cannot leak into Kimi's answer.  Every API response is retained as
an atomic JSON receipt, including token log-probabilities, before a derived
training manifest is written.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "curated_vision/vision_eight_hour.jsonl"
DEFAULT_EVAL = ROOT / "curated_vision/vision_eight_hour_eval.jsonl"
DEFAULT_QUEUE = ROOT / "curated_vision/kimi_k26_teacher_queue.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets/kimi_k26_teacher"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k2.6"
PROVIDER = "decart"
DEFAULT_VISION_PROVIDERS = (
    # Decart currently advertises the multimodal model on OpenRouter but its
    # endpoint rejects image input. Keep it out until that upstream mismatch is
    # fixed. These endpoints advertise logprobs + top_logprobs for Kimi K2.6.
    "inceptron", "parasail", "digitalocean", "streamlake", "wandb",
    "cloudflare", "fireworks",
)
INPUT_USD_PER_TOKEN = 0.66e-6
OUTPUT_USD_PER_TOKEN = 3.41e-6

# This is intentionally not a short-caption prompt. Kimi should stop at EOS
# after exhausting useful visual evidence.
CAPTION_PROMPT = """Write one exhaustive, grounded caption of this image for training a vision-language model. Describe all useful visible details: subjects, objects, attributes, actions, spatial relationships, setting, composition, lighting, visual style, and clearly legible text. Use coherent natural prose. Be precise and use cautious wording where the image is ambiguous. Do not add a heading, list, preamble, prompt language, aesthetic quality tags, or facts that cannot be inferred from the pixels. Do not target an arbitrary word count; continue until the useful visible information is covered, then stop."""

WORD_RE = re.compile(r"\b[\w]+(?:['-][\w]+)*\b", re.UNICODE)
INTRO_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:this is (?:a |an )?(?:detailed )?description of "
    r"(?:the |this )?image|the image (?:shows|depicts|captures|presents)|"
    r"this image (?:shows|depicts|captures|presents))",
    re.IGNORECASE,
)
GENERATIONISM_RE = re.compile(
    r"\b(?:masterpiece|best quality|amazing quality|ultra[- ]?detailed|"
    r"trending on artstation|award[- ]winning|8k|4k)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    """Commit directory-entry changes needed to recover after host power loss."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def resolve_image(path: str, root: Path = ROOT) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def caption_selection_score(text: str) -> int:
    """Favor descriptive length while discounting obvious non-information.

    The penalties are deliberately conservative: this remains essentially a
    longest-first ranking, not a stylistic classifier. Multiple image-intro
    phrases are a useful signal that two paraphrased captions were concatenated.
    """
    words = word_count(text)
    repeated_intros = max(0, len(INTRO_RE.findall(text)) - 1)
    generationisms = len(GENERATIONISM_RE.findall(text))
    return words - 16 * repeated_intros - 18 * generationisms


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield value


def _queue_row(row: dict, *, split: str, source: str) -> dict:
    text = str(row.get("text", "")).strip()
    image = str(row.get("image", ""))
    identity = hashlib.sha256(f"{split}\0{source}\0{image}".encode()).hexdigest()[:24]
    return {
        "id": identity,
        "split": split,
        "source": source,
        "image": image,
        "old_caption": text,
        "old_caption_words": word_count(text),
        "selection_score": caption_selection_score(text),
    }


def build_queue(
    train_manifest: Path,
    eval_manifest: Path,
    output: Path,
    *,
    train_limit: int,
    eval_limit: int,
    require_images: bool = True,
) -> dict:
    """Write clean evaluation rows first, then longest useful Pexels rows."""
    evaluation = []
    allowed_eval = {"eval_i1_pexels", "eval_i1_midjourneyv6"}
    for row in iter_jsonl(eval_manifest):
        source = str(row.get("stage1_source", ""))
        if source not in allowed_eval:
            continue
        item = _queue_row(row, split="eval", source=source)
        if item["old_caption_words"] and (
            not require_images or resolve_image(item["image"]).is_file()
        ):
            evaluation.append(item)

    training = []
    for row in iter_jsonl(train_manifest):
        source = str(row.get("stage1_source", ""))
        if source != "eight_hour_i1_pexels":
            continue
        item = _queue_row(row, split="train", source=source)
        if item["old_caption_words"] and (
            not require_images or resolve_image(item["image"]).is_file()
        ):
            training.append(item)

    # Stable tie-breaks make a queue receipt reproducible.
    rank = lambda item: (-item["selection_score"], -item["old_caption_words"], item["id"])
    evaluation.sort(key=rank)
    training.sort(key=rank)

    def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
        # Multi-caption images share one queue identity; scheduling both rows
        # would bill the same image twice and overwrite one receipt with the
        # other. Rows are already ranked, so the first per id is the best.
        unique: dict[str, dict] = {}
        for item in rows:
            unique.setdefault(item["id"], item)
        return list(unique.values()), len(rows) - len(unique)

    evaluation, eval_duplicates = deduplicate(evaluation)
    training, train_duplicates = deduplicate(training)
    dropped_duplicates = eval_duplicates + train_duplicates
    if dropped_duplicates:
        print(f"dropped {dropped_duplicates} duplicate queue rows "
              f"(eval={eval_duplicates}, train={train_duplicates})", flush=True)
    if eval_limit:
        evaluation = evaluation[:eval_limit]
    if train_limit:
        training = training[:train_limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for queue_priority, item in enumerate([*evaluation, *training], 1):
            item["queue_priority"] = queue_priority
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    fsync_directory(output.parent)
    return {
        "output": str(output),
        "eval": len(evaluation),
        "train": len(training),
        "total": len(evaluation) + len(training),
        "dropped_duplicates": dropped_duplicates,
        "top_train_words": training[0]["old_caption_words"] if training else 0,
        "bottom_train_words": training[-1]["old_caption_words"] if training else 0,
    }


def image_data_url(path: Path, max_side: int = 0) -> tuple[str, dict]:
    """Encode a full image, optionally downscaled without cropping."""
    original_bytes = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with Image.open(path) as image:
        original_size = image.size
        if max_side > 0 and max(image.size) > max_side:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=95, optimize=True)
            payload = buffer.getvalue()
            mime = "image/jpeg"
            submitted_size = image.size
        else:
            payload = original_bytes
            submitted_size = image.size
    digest = hashlib.sha256(original_bytes).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}", {
        "sha256": digest,
        "original_size": list(original_size),
        "submitted_size": list(submitted_size),
        "submitted_bytes": len(payload),
        "mime": mime,
    }


def make_payload(data_url: str, *, max_completion_tokens: int,
                 top_logprobs: int, providers: tuple[str, ...] = (PROVIDER,),
                 allow_fallbacks: bool = False) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }],
        "provider": {
            "only": list(providers),
            "order": list(providers),
            "allow_fallbacks": allow_fallbacks,
            "require_parameters": True,
        },
        "temperature": 0,
        "seed": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "reasoning": {"effort": "none", "exclude": True},
        "stream": False,
    }
    # A very high value is a runaway guard, not a desired caption length.
    # A zero value omits the parameter entirely when explicitly requested.
    if max_completion_tokens > 0:
        payload["max_tokens"] = max_completion_tokens
    return payload


class ApiFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def post_completion(payload: dict, api_key: str, *, timeout: int) -> dict:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sirus20x6/rwkv-lab",
            "X-Title": "RWKV Kimi vision teacher",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read(4096).decode("utf-8", "replace")
        raise ApiFailure(
            f"OpenRouter HTTP {error.code}: {body}",
            retryable=error.code == 429 or 500 <= error.code < 600,
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ApiFailure(f"OpenRouter transport failure: {error}", retryable=True) from error
    if not isinstance(result, dict):
        raise ApiFailure("OpenRouter returned a non-object response", retryable=True)
    error_body = result.get("error")
    if error_body:
        # OpenRouter embeds provider errors (including 429s) inside 200
        # bodies. Classify them with the same rules as the HTTPError path.
        code = None
        message = f"OpenRouter error: {error_body}"
        if isinstance(error_body, dict):
            try:
                code = int(error_body.get("code"))
            except (TypeError, ValueError):
                code = None
            metadata = error_body.get("metadata")
            if metadata:
                message = f"OpenRouter error {code}: {error_body.get('message')} metadata={metadata}"
        retryable = code is not None and (code == 429 or 500 <= code < 600)
        raise ApiFailure(message, retryable=retryable)
    return result


def extract_caption(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response has no assistant content") from error
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ).strip()
    raise ValueError("assistant content has an unsupported shape")


def logprob_summary(response: dict) -> dict:
    try:
        content = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        content = []
    chosen = [float(token["logprob"]) for token in content
              if isinstance(token, dict) and isinstance(token.get("logprob"), (int, float))]
    top_counts = [len(token.get("top_logprobs", [])) for token in content
                  if isinstance(token, dict)]
    if not chosen:
        return {"tokens": 0, "top_alternatives": 0}
    return {
        "tokens": len(chosen),
        "sequence_logprob": sum(chosen),
        "mean_logprob": sum(chosen) / len(chosen),
        "mean_chosen_probability": sum(math.exp(value) for value in chosen) / len(chosen),
        "min_logprob": min(chosen),
        "top_alternatives": sum(top_counts),
    }


def response_cost(response: dict) -> float:
    usage = response.get("usage") or {}
    explicit = usage.get("cost")
    try:
        explicit_cost = float(explicit)
    except (TypeError, ValueError):
        explicit_cost = -1.0
    if explicit_cost >= 0:
        return explicit_cost
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt * INPUT_USD_PER_TOKEN + completion * OUTPUT_USD_PER_TOKEN


@dataclass(frozen=True)
class CaptionConfig:
    output: Path
    max_completion_tokens: int
    top_logprobs: int
    max_image_side: int
    timeout: int
    retries: int
    providers: tuple[str, ...]
    allow_fallbacks: bool


def caption_one(item: dict, *, config: CaptionConfig, api_key: str) -> dict:
    image_path = resolve_image(item["image"])
    data_url, image_metadata = image_data_url(image_path, config.max_image_side)
    payload = make_payload(
        data_url,
        max_completion_tokens=config.max_completion_tokens,
        top_logprobs=config.top_logprobs,
        providers=config.providers,
        allow_fallbacks=config.allow_fallbacks,
    )
    response = None
    failure = None
    for attempt in range(config.retries + 1):
        try:
            response = post_completion(payload, api_key, timeout=config.timeout)
            break
        except ApiFailure as error:
            failure = error
            if not error.retryable or attempt == config.retries:
                break
            time.sleep(min(30.0, (2 ** attempt) + random.random()))

    request_metadata = {
        "model": MODEL,
        "providers": list(config.providers),
        "allow_fallbacks": config.allow_fallbacks,
        "prompt": CAPTION_PROMPT,
        "temperature": 0,
        "seed": 0,
        "top_logprobs": config.top_logprobs,
        "max_completion_tokens": config.max_completion_tokens or None,
        "image": image_metadata,
    }
    receipt = {
        "schema": 1,
        "created_at": utc_now(),
        "queue": item,
        "request": request_metadata,
    }
    if response is None:
        receipt.update({"accepted": False, "error": str(failure), "cost_usd": 0.0})
    else:
        parse_error = None
        try:
            caption = extract_caption(response)
        except ValueError as error:
            # The request may already have been billed. Preserve and account
            # for the response even when its normalized shape is unexpected.
            caption = ""
            parse_error = str(error)
        choice = (response.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        response_provider = str(response.get("provider") or "").lower()
        wrong_provider = bool(
            response_provider
            and not any(provider in response_provider for provider in config.providers)
        )
        accepted = bool(caption) and finish_reason == "stop" and not wrong_provider
        receipt.update({
            "accepted": accepted,
            "caption": caption,
            "finish_reason": finish_reason,
            "provider": response.get("provider"),
            "cost_usd": response_cost(response),
            "logprob_summary": logprob_summary(response),
            # Preserve the normalized response verbatim. This includes every
            # chosen-token logprob, its bytes, and the top alternatives.
            "response": response,
        })
        if not accepted:
            reasons = []
            if parse_error:
                reasons.append(parse_error)
            if not caption:
                reasons.append("empty caption")
            if finish_reason != "stop":
                reasons.append(f"non-EOS finish reason {finish_reason!r}")
            if wrong_provider:
                reasons.append(f"unexpected provider {response.get('provider')!r}")
            receipt["error"] = "; ".join(reasons) + "; retained but excluded from training"
    atomic_json(config.output / "raw" / f"{item['id']}.json", receipt)
    return receipt


def existing_receipts(output: Path) -> dict[str, dict]:
    receipts = {}
    for path in sorted((output / "raw").glob("*.json")):
        try:
            receipt = json.loads(path.read_text())
            identity = receipt["queue"]["id"]
            receipts[str(identity)] = receipt
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            # A corrupt receipt is never considered complete and will be
            # replaced atomically by a resumed request.
            continue
    return receipts


def write_derived_manifests(output: Path, receipts: Iterable[dict]) -> dict:
    rows = []
    spent = 0.0
    for receipt in receipts:
        spent += float(receipt.get("cost_usd") or 0.0)
        if not receipt.get("accepted"):
            continue
        queue = receipt["queue"]
        rows.append({
            "image": queue["image"],
            "text": receipt["caption"],
            "task": "caption",
            "stage1_source": "kimi_k26_teacher",
            "split": queue["split"],
            "teacher": {
                "model": MODEL,
                "provider": receipt.get("provider") or PROVIDER,
                "receipt": f"raw/{queue['id']}.json",
                "request_id": receipt.get("response", {}).get("id"),
                "finish_reason": receipt.get("finish_reason"),
                "cost_usd": receipt.get("cost_usd"),
                **receipt.get("logprob_summary", {}),
            },
        })
    rows.sort(key=lambda row: (row["split"] != "eval", row["teacher"]["receipt"]))
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "eval"):
        target = output / f"{split}.jsonl"
        selected = [row for row in rows if row["split"] == split]
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
        counts[split] = len(selected)
    summary = {"updated_at": utc_now(), "spent_usd": spent, **counts}
    atomic_json(output / "summary.json", summary)
    return summary


def execute_queue(args: argparse.Namespace) -> dict:
    if not args.execute:
        raise SystemExit("refusing to spend credits without --execute")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")
    if not 0 <= args.top_logprobs <= 20:
        raise SystemExit("--top-logprobs must be between 0 and 20")

    items = list(iter_jsonl(args.queue))
    receipts = existing_receipts(args.output)
    spent = sum(float(row.get("cost_usd") or 0.0) for row in receipts.values())
    # Zero-cost transport/provider rejections are safe to retry after routing
    # is repaired. Billed non-EOS responses remain complete artifacts so a
    # resume cannot silently pay for the same image twice.
    complete_ids = {
        identity for identity, receipt in receipts.items()
        if receipt.get("accepted") or float(receipt.get("cost_usd") or 0.0) > 0
    }
    pending = [item for item in items if item["id"] not in complete_ids]
    if args.max_items:
        pending = pending[:args.max_items]

    config = CaptionConfig(
        output=args.output,
        max_completion_tokens=args.max_completion_tokens,
        top_logprobs=args.top_logprobs,
        max_image_side=args.max_image_side,
        timeout=args.timeout,
        retries=args.retries,
        providers=tuple(args.providers),
        allow_fallbacks=not args.no_provider_fallbacks,
    )
    # Reserve enough for a full safety-limit completion plus a large visual
    # prompt for every in-flight request. This prevents concurrency overshoot.
    completion_reserve = (args.max_completion_tokens or 4096) * OUTPUT_USD_PER_TOKEN
    per_request_reserve = completion_reserve + args.input_reserve_usd
    iterator = iter(pending)
    futures = {}
    attempted = 0
    accepted_this_run = 0
    consecutive_failures = 0
    stop_scheduling = False
    requeued: list[dict] = []
    retried: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="kimi-teacher") as pool:
        while True:
            while not stop_scheduling and len(futures) < args.workers \
                    and spent + (len(futures) + 1) * per_request_reserve <= args.budget_usd:
                if requeued:
                    item = requeued.pop(0)
                else:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                future = pool.submit(caption_one, item, config=config, api_key=api_key)
                futures[future] = item
                attempted += 1
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                try:
                    receipt = future.result()
                except Exception as error:
                    # A local exception means no receipt was written; the item
                    # would otherwise silently vanish from this run. Re-queue
                    # it exactly once before giving up with the old message.
                    if item["id"] not in retried:
                        retried.add(item["id"])
                        requeued.append(item)
                        print(f"re-queueing {item['id']} after local failure: {error}",
                              flush=True)
                    else:
                        print(f"failed {item['id']}: {error}", flush=True)
                    consecutive_failures += 1
                    stop_scheduling = consecutive_failures >= args.max_consecutive_failures
                    continue
                receipts[item["id"]] = receipt
                spent += float(receipt.get("cost_usd") or 0.0)
                if receipt.get("accepted"):
                    accepted_this_run += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    stop_scheduling = consecutive_failures >= args.max_consecutive_failures
                print(
                    f"{len(receipts)}/{len(items)} {item['split']} {item['id']} "
                    f"accepted={receipt.get('accepted')} spent=${spent:.4f}",
                    flush=True,
                )
    summary = write_derived_manifests(args.output, receipts.values())
    if attempted and accepted_this_run == 0:
        raise SystemExit(
            f"stopped after {attempted} attempted requests with no accepted captions"
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="build a longest-first clean-image queue")
    select.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    select.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL)
    select.add_argument("--output", type=Path, default=DEFAULT_QUEUE)
    select.add_argument("--train-limit", type=int, default=5000)
    select.add_argument("--eval-limit", type=int, default=256)
    select.add_argument("--allow-missing-images", action="store_true")

    caption = subparsers.add_parser("caption", help="execute or resume the queue")
    caption.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    caption.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    caption.add_argument("--budget-usd", type=float, default=3.80)
    caption.add_argument("--workers", type=int, default=4)
    caption.add_argument(
        "--providers", nargs="+", default=list(DEFAULT_VISION_PROVIDERS),
        help="ordered OpenRouter provider allowlist",
    )
    caption.add_argument("--no-provider-fallbacks", action="store_true")
    caption.add_argument("--max-consecutive-failures", type=int, default=4)
    caption.add_argument("--max-items", type=int, default=0)
    caption.add_argument("--top-logprobs", type=int, default=20)
    caption.add_argument(
        "--max-completion-tokens", type=int, default=2048,
        help="runaway safety ceiling; length-finished answers are rejected (0 omits it)",
    )
    caption.add_argument(
        "--max-image-side", type=int, default=0,
        help="optional full-frame resize; zero sends original pixels",
    )
    caption.add_argument("--input-reserve-usd", type=float, default=0.002)
    caption.add_argument("--timeout", type=int, default=180)
    caption.add_argument("--retries", type=int, default=4)
    caption.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    caption.add_argument(
        "--execute", action="store_true",
        help="required acknowledgement that API requests spend credits",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "select":
        summary = build_queue(
            args.train_manifest,
            args.eval_manifest,
            args.output,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            require_images=not args.allow_missing_images,
        )
    else:
        summary = execute_queue(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
