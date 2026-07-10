"""Typed, inspectable datasets for SFT, preference learning, feedback, and RLVR.

The important contract is not a particular chat syntax.  It is that rendering preserves
message roles and produces an explicit token-level loss mask.  That prevents the common
instruction-tuning bug where user/system text is silently trained as a target.  The schema is
inspired by the dataset/template layers exposed by LLaMA-Factory and Axolotl, while remaining
small and architecture-neutral:

* LLaMA-Factory paper / data pipeline: https://arxiv.org/abs/2403.13372
* Axolotl dataset formats: https://docs.axolotl.ai/dataset-formats/

JSONL schemas (``rwkv-lab.posttrain.v1``):

* ``pretrain``: ``{id, split, kind, text}``
* ``sft``: ``{id, split, kind, messages:[{role,content}, ...]}``
* ``preference``: ``{..., messages:[prompt turns...], chosen, rejected}``
* ``feedback``: ``{..., messages:[prompt turns...], response, label:true|false}``
* ``rlvr``: the existing ``{prompt, verifier}`` record is accepted without conversion.

All loaders retain metadata, source hashes, and deterministic template hashes for registry and
export receipts.  Generated code is data only; nothing in this module executes it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Protocol, Sequence


SCHEMA = "rwkv-lab.posttrain.v1"
IGNORE_INDEX = -100
ROLES = frozenset({"system", "user", "assistant", "tool"})
KINDS = frozenset({"pretrain", "sft", "preference", "feedback", "rlvr"})
SPLITS = frozenset({"train", "eval", "test"})


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: str = ""

    @classmethod
    def from_value(cls, value: Any, *, where: str) -> "Message":
        if not isinstance(value, dict):
            raise ValueError(f"{where}: message must be an object")
        role = str(value.get("role") or value.get("from") or "").strip().lower()
        role = {"human": "user", "gpt": "assistant", "function": "tool"}.get(role, role)
        content = value.get("content", value.get("value", ""))
        if role not in ROLES:
            raise ValueError(f"{where}: unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{where}: message content must be non-empty text")
        return cls(role, content, str(value.get("name") or ""))


@dataclass(frozen=True)
class PostTrainingExample:
    id: str
    kind: str
    split: str = "train"
    messages: tuple[Message, ...] = ()
    text: str = ""
    chosen: str = ""
    rejected: str = ""
    response: str = ""
    label: bool | None = None
    verifier: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, line: int = 0) -> "PostTrainingExample":
        if not isinstance(value, dict):
            raise ValueError(f"line {line}: example must be an object")
        kind = str(value.get("kind") or _infer_kind(value)).strip().lower()
        split = str(value.get("split") or "train").strip().lower()
        eid = str(value.get("id") or f"line-{line}")
        if kind not in KINDS:
            raise ValueError(f"line {line}: unsupported kind {kind!r}")
        if split not in SPLITS:
            raise ValueError(f"line {line}: split must be one of {sorted(SPLITS)}")
        raw_messages = value.get("messages", value.get("conversations", ()))
        if not raw_messages and value.get("prompt") is not None and kind != "pretrain":
            raw_messages = [{"role": "user", "content": str(value["prompt"])}]
        if kind == "sft" and (value.get("response") is not None or value.get("answer") is not None):
            raw_messages = list(raw_messages or ())
            if not any(str(message.get("role", message.get("from", ""))).lower() in ("assistant", "gpt")
                       for message in raw_messages if isinstance(message, dict)):
                raw_messages.append({"role": "assistant",
                                     "content": _response_text(value.get("response", value.get("answer")))})
        if raw_messages and not isinstance(raw_messages, list):
            raise ValueError(f"line {line}: messages must be an array")
        messages = tuple(Message.from_value(m, where=f"line {line} message {i}")
                         for i, m in enumerate(raw_messages or ()))
        example = cls(
            id=eid, kind=kind, split=split, messages=messages,
            text=str(value.get("text") or ""), chosen=_response_text(value.get("chosen")),
            rejected=_response_text(value.get("rejected")),
            response=_response_text(value.get("response", value.get("answer"))),
            label=_bool_or_none(value.get("label")),
            verifier=dict(value.get("verifier") or {}) or None,
            metadata=dict(value.get("metadata") or {}),
        )
        example.validate(line=line)
        return example

    def validate(self, *, line: int = 0) -> None:
        where = f"line {line}" if line else self.id
        if self.kind == "pretrain" and not self.text.strip():
            raise ValueError(f"{where}: pretrain example needs text")
        if self.kind == "sft":
            if not self.messages or not any(m.role == "assistant" for m in self.messages):
                raise ValueError(f"{where}: SFT example needs at least one assistant message")
        if self.kind == "preference":
            if not self.messages or not self.chosen.strip() or not self.rejected.strip():
                raise ValueError(f"{where}: preference example needs prompt messages, chosen, and rejected")
            if self.chosen.strip() == self.rejected.strip():
                raise ValueError(f"{where}: chosen and rejected responses must differ")
        if self.kind == "feedback":
            if not self.messages or not self.response.strip() or self.label is None:
                raise ValueError(f"{where}: feedback example needs prompt messages, response, and boolean label")
        if self.kind == "rlvr":
            if not self.messages or not self.verifier or not self.verifier.get("kind"):
                raise ValueError(f"{where}: RLVR example needs a prompt and verifier.kind")


def _infer_kind(value: dict[str, Any]) -> str:
    if "verifier" in value:
        return "rlvr"
    if "chosen" in value or "rejected" in value:
        return "preference"
    if "label" in value and ("response" in value or "answer" in value):
        return "feedback"
    if "messages" in value or "conversations" in value or "prompt" in value:
        return "sft"
    return "pretrain"


def _response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value.get("value", "")))
    raise ValueError("response values must be text or a message object")


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError("feedback label must be boolean")


@dataclass(frozen=True)
class Segment:
    text: str
    train: bool
    role: str = ""


@dataclass(frozen=True)
class RenderedVariant:
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)


@dataclass(frozen=True)
class RenderedExample:
    id: str
    kind: str
    split: str
    variants: dict[str, RenderedVariant]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChatTemplate:
    name: str = "rwkv-role-v1"
    system_prefix: str = "System: "
    user_prefix: str = "User: "
    assistant_prefix: str = "Assistant: "
    tool_prefix: str = "Tool: "
    turn_separator: str = "\n\n"
    eos: str = "\x00"

    def prefix(self, role: str) -> str:
        return getattr(self, f"{role}_prefix")

    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def prompt_segments(self, messages: Sequence[Message]) -> list[Segment]:
        out: list[Segment] = []
        for i, message in enumerate(messages):
            if i:
                out.append(Segment(self.turn_separator, False, message.role))
            train = message.role == "assistant"
            out.append(Segment(self.prefix(message.role), train, message.role))
            out.append(Segment(message.content, train, message.role))
            if train:
                out.append(Segment(self.eos, True, message.role))
        return out

    def response_segments(self, messages: Sequence[Message], response: str, *, train: bool) -> list[Segment]:
        out = self.prompt_segments(messages)
        if out:
            out.append(Segment(self.turn_separator, False))
        out.extend((Segment(self.assistant_prefix, train, "assistant"),
                    Segment(response, train, "assistant"),
                    Segment(self.eos, train, "assistant")))
        return out


DEFAULT_TEMPLATE = ChatTemplate()
TEMPLATES: dict[str, ChatTemplate] = {DEFAULT_TEMPLATE.name: DEFAULT_TEMPLATE}


def register_template(template: ChatTemplate, *, replace: bool = False) -> None:
    if template.name in TEMPLATES and not replace:
        raise ValueError(f"chat template {template.name!r} is already registered")
    TEMPLATES[template.name] = template


def get_template(name: str = DEFAULT_TEMPLATE.name) -> ChatTemplate:
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        raise ValueError(f"unknown chat template {name!r}; available={sorted(TEMPLATES)}") from exc


def load_template(path: str | Path) -> ChatTemplate:
    value = json.loads(Path(path).read_text())
    allowed = set(ChatTemplate.__dataclass_fields__)
    unknown = set(value) - allowed - {"sha256"}
    if unknown:
        raise ValueError(f"unknown chat-template fields: {sorted(unknown)}")
    value.pop("sha256", None)
    template = ChatTemplate(**value)
    register_template(template, replace=True)
    return template


def render(example: PostTrainingExample, template: ChatTemplate = DEFAULT_TEMPLATE) -> RenderedExample:
    variants: dict[str, RenderedVariant]
    if example.kind == "pretrain":
        variants = {"text": RenderedVariant((Segment(example.text + template.eos, True, "text"),))}
    elif example.kind == "sft":
        variants = {"sft": RenderedVariant(tuple(template.prompt_segments(example.messages)))}
    elif example.kind == "preference":
        variants = {
            "chosen": RenderedVariant(tuple(template.response_segments(example.messages, example.chosen,
                                                                          train=True))),
            "rejected": RenderedVariant(tuple(template.response_segments(example.messages, example.rejected,
                                                                            train=True))),
        }
    elif example.kind == "feedback":
        variants = {"response": RenderedVariant(tuple(template.response_segments(
            example.messages, example.response, train=True)))}
    else:  # RLVR prompts are inputs; rollouts supply the trainable response later.
        variants = {"prompt": RenderedVariant(tuple(template.prompt_segments(example.messages) + [
            Segment(template.turn_separator + template.assistant_prefix, False, "assistant")]))}
    return RenderedExample(example.id, example.kind, example.split, variants,
                           {**example.metadata, "template": template.name,
                            "template_sha256": template.fingerprint(),
                            **({"label": example.label} if example.label is not None else {}),
                            **({"verifier": example.verifier} if example.verifier else {})})


@dataclass(frozen=True)
class TokenizedVariant:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    roles: tuple[str, ...]
    truncated: int = 0


@dataclass(frozen=True)
class TokenizedExample:
    id: str
    kind: str
    split: str
    variants: dict[str, TokenizedVariant]
    metadata: dict[str, Any]


def tokenize(rendered: RenderedExample, tokenizer: Tokenizer, *, max_length: int = 0,
             truncate: str = "left") -> TokenizedExample:
    if truncate not in ("left", "right", "error"):
        raise ValueError("truncate must be left, right, or error")
    variants = {}
    for name, variant in rendered.variants.items():
        ids: list[int] = []
        labels: list[int] = []
        roles: list[str] = []
        # Coalesce formatting fragments so tokenization is faithful within every role/mask span;
        # the only forced tokenizer boundary is where the supervision mask itself changes.
        segments: list[Segment] = []
        for segment in variant.segments:
            if segments and (segments[-1].train, segments[-1].role) == (segment.train, segment.role):
                previous = segments[-1]
                segments[-1] = Segment(previous.text + segment.text, previous.train, previous.role)
            else:
                segments.append(segment)
        for segment in segments:
            segment_ids = list(tokenizer.encode(segment.text))
            ids.extend(segment_ids)
            labels.extend(segment_ids if segment.train else [IGNORE_INDEX] * len(segment_ids))
            roles.extend([segment.role] * len(segment_ids))
        removed = max(0, len(ids) - int(max_length)) if max_length else 0
        if removed:
            if truncate == "error":
                raise ValueError(f"{rendered.id}/{name}: {len(ids)} tokens exceeds max_length={max_length}")
            sl = slice(removed, None) if truncate == "left" else slice(None, max_length)
            ids, labels, roles = ids[sl], labels[sl], roles[sl]
        if not ids:
            raise ValueError(f"{rendered.id}/{name}: rendering produced no tokens")
        if rendered.kind in ("sft", "preference", "feedback") and all(x == IGNORE_INDEX for x in labels):
            raise ValueError(f"{rendered.id}/{name}: truncation removed every trainable token")
        variants[name] = TokenizedVariant(tuple(ids), tuple(labels), tuple(roles), removed)
    return TokenizedExample(rendered.id, rendered.kind, rendered.split, variants, rendered.metadata)


def load_jsonl(path: str | Path) -> tuple[list[PostTrainingExample], str]:
    raw = Path(path).read_bytes()
    rows = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            rows.append(PostTrainingExample.from_dict(json.loads(line), line=line_no))
    if not rows:
        raise ValueError("post-training dataset is empty")
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("post-training example ids must be unique")
    return rows, hashlib.sha256(raw).hexdigest()


def dataset_manifest(path: str | Path, *, template: ChatTemplate = DEFAULT_TEMPLATE,
                     examples: Iterable[PostTrainingExample] | None = None) -> dict[str, Any]:
    p = Path(path).resolve()
    rows, digest = load_jsonl(p) if examples is None else (list(examples), _sha256_file(p))
    kinds: dict[str, int] = {}
    splits: dict[str, int] = {}
    contents: dict[str, list[PostTrainingExample]] = {}
    for row in rows:
        kinds[row.kind] = kinds.get(row.kind, 0) + 1
        splits[row.split] = splits.get(row.split, 0) + 1
        contents.setdefault(_content_fingerprint(row), []).append(row)
    duplicate_groups = [group for group in contents.values() if len(group) > 1]
    split_overlaps = [group for group in duplicate_groups if len({row.split for row in group}) > 1]
    return {"schema": SCHEMA, "path": str(p), "sha256": digest, "bytes": p.stat().st_size,
            "examples": len(rows), "kinds": kinds, "splits": splits,
            "duplicates": sum(len(group) - 1 for group in duplicate_groups),
            "split_overlaps": len(split_overlaps),
            "split_overlap_ids": [[row.id for row in group] for group in split_overlaps[:20]],
            "template": template.name, "template_sha256": template.fingerprint()}


def inspect(path: str | Path, *, limit: int = 3,
            template: ChatTemplate = DEFAULT_TEMPLATE) -> dict[str, Any]:
    rows, _ = load_jsonl(path)
    result = dataset_manifest(path, template=template, examples=rows)
    previews = []
    for row in rows[:max(0, limit)]:
        rendered = render(row, template)
        previews.append({"id": row.id, "kind": row.kind, "split": row.split,
                         "variants": {name: {"text": value.text,
                                             "train_chars": sum(len(s.text) for s in value.segments if s.train)}
                                      for name, value in rendered.variants.items()}})
    result["previews"] = previews
    return result


def version_dataset(paths: Sequence[str | Path], output_root: str | Path, *,
                    template: ChatTemplate = DEFAULT_TEMPLATE) -> dict[str, Any]:
    """Validate, merge, and materialize one immutable content-addressed dataset version."""
    if not paths:
        raise ValueError("at least one source dataset is required")
    rows: list[PostTrainingExample] = []
    sources = []
    for value in paths:
        source_rows, digest = load_jsonl(value)
        rows.extend(source_rows)
        source = Path(value).resolve()
        sources.append({"path": str(source), "sha256": digest, "bytes": source.stat().st_size})
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("merged dataset ids must be globally unique")
    contents: dict[str, list[PostTrainingExample]] = {}
    for row in rows:
        contents.setdefault(_content_fingerprint(row), []).append(row)
    duplicate = [group for group in contents.values() if len(group) > 1]
    if duplicate:
        overlaps = [group for group in duplicate if len({row.split for row in group}) > 1]
        if overlaps:
            raise ValueError("merged dataset has identical content across train/eval/test splits")
        raise ValueError("merged dataset has duplicate training content")
    lines = [json.dumps(_example_dict(row), ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")) for row in rows]
    payload = ("\n".join(lines) + "\n").encode()
    digest = hashlib.sha256(payload + template.fingerprint().encode()).hexdigest()
    root = Path(output_root).resolve()
    destination = root / digest[:16]
    if destination.is_dir():
        return json.loads((destination / "manifest.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=digest[:16] + ".tmp-", dir=root))
    try:
        dataset = temporary / "dataset.jsonl"
        dataset.write_bytes(payload)
        manifest = dataset_manifest(dataset, template=template, examples=rows)
        manifest.update({"version": digest[:16], "sources": sources,
                         "path": str(destination / "dataset.jsonl"),
                         "dataset": str(destination / "dataset.jsonl")})
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                              ensure_ascii=False,
                                                              sort_keys=True) + "\n")
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_fingerprint(row: PostTrainingExample) -> str:
    payload = {"kind": row.kind, "messages": [message.__dict__ for message in row.messages],
               "text": row.text, "chosen": row.chosen, "rejected": row.rejected,
               "response": row.response, "label": row.label, "verifier": row.verifier}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _example_dict(row: PostTrainingExample) -> dict[str, Any]:
    value: dict[str, Any] = {"schema": SCHEMA, "id": row.id, "kind": row.kind,
                             "split": row.split, "metadata": row.metadata}
    if row.messages:
        value["messages"] = [message.__dict__ for message in row.messages]
    for name in ("text", "chosen", "rejected", "response"):
        content = getattr(row, name)
        if content:
            value[name] = content
    if row.label is not None:
        value["label"] = row.label
    if row.verifier:
        value["verifier"] = row.verifier
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preview rwkv-lab post-training JSONL")
    parser.add_argument("path", nargs="+")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--template", default="", help="custom ChatTemplate JSON")
    parser.add_argument("--version-root", default="",
                        help="validate/merge all paths into a content-addressed version here")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    template = load_template(args.template) if args.template else DEFAULT_TEMPLATE
    if args.version_root:
        result = version_dataset(args.path, args.version_root, template=template)
    else:
        if len(args.path) != 1:
            parser.error("multiple paths require --version-root")
        result = inspect(args.path[0], limit=args.limit, template=template)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
