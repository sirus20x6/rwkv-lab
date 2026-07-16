#!/usr/bin/env python3
"""Local, source-blinded reviewer for ``quality_review_pack``.

Run from the repository root:
    python scripts/review_quality_pack.py

Navigation: Left/Right arrows (or Ctrl+Left/Ctrl+Right while editing notes).
Scores and notes are saved immediately to ``review_scores.csv`` in the pack.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageOps, ImageTk


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("grounding", "detail", "format", "safety_or_label", "usability", "notes")


def load_items(pack: Path) -> list[dict]:
    with (pack / "review_manifest.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scores(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def save_scores(path: Path, items: list[dict], scores: dict[str, dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", *FIELDS))
        writer.writeheader()
        for item in items:
            row = {"id": item["id"], **{field: scores.get(item["id"], {}).get(field, "") for field in FIELDS}}
            writer.writerow(row)


class ReviewApp(tk.Tk):
    def __init__(self, pack: Path):
        super().__init__()
        self.pack = pack
        self.items = load_items(pack)
        self.score_path = pack / "review_scores.csv"
        self.scores = load_scores(self.score_path)
        self.index = 0
        self.loading = False
        self.photo: ImageTk.PhotoImage | None = None
        self.title("Blinded Visual Quality Review")
        self.geometry("1440x900")
        self.minsize(900, 620)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.bind_all("<Right>", self.next_item)
        self.bind_all("<Left>", self.previous_item)
        self.bind_all("<Control-Right>", self.next_item)
        self.bind_all("<Control-Left>", self.previous_item)
        self.bind_all("<Key-1>", lambda _event: self.set_score("grounding", "1"))
        self.bind_all("<Key-2>", lambda _event: self.set_score("grounding", "2"))
        self.bind_all("<Key-3>", lambda _event: self.set_score("grounding", "3"))
        self.bind_all("<Key-4>", lambda _event: self.set_score("grounding", "4"))
        self.bind_all("<Key-5>", lambda _event: self.set_score("grounding", "5"))
        self.bind("<Configure>", self._resize_image)
        self.show_item()

    def _build(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(left, anchor="center", text="Image unavailable")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_status = ttk.Label(left, anchor="center")
        self.image_status.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.heading = ttk.Label(right, font=("TkDefaultFont", 14, "bold"))
        self.heading.grid(row=0, column=0, sticky="w")
        self.reference = tk.Text(right, wrap="word", height=16, state="disabled", padx=8, pady=8)
        self.reference.grid(row=1, column=0, sticky="nsew", pady=(8, 12))

        scores = ttk.LabelFrame(right, text="Scores (1 = poor, 5 = excellent)", padding=8)
        scores.grid(row=2, column=0, sticky="ew")
        self.score_vars: dict[str, tk.StringVar] = {}
        for row, field in enumerate(FIELDS[:-1]):
            ttk.Label(scores, text=field.replace("_", " ").title()).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            variable = tk.StringVar()
            variable.trace_add("write", lambda *_args: self.persist_current())
            self.score_vars[field] = variable
            ttk.Combobox(scores, textvariable=variable, values=("", "1", "2", "3", "4", "5"), width=5,
                         state="readonly").grid(row=row, column=1, sticky="w", pady=3)

        ttk.Label(right, text="Notes").grid(row=3, column=0, sticky="w", pady=(12, 2))
        self.notes = tk.Text(right, wrap="word", height=8, padx=8, pady=8)
        self.notes.grid(row=4, column=0, sticky="ew")
        self.notes.bind("<FocusOut>", lambda _event: self.persist_current())

        controls = ttk.Frame(right)
        controls.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        controls.columnconfigure(1, weight=1)
        ttk.Button(controls, text="← Previous", command=self.previous_item).grid(row=0, column=0)
        self.position = ttk.Label(controls, anchor="center")
        self.position.grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text="Next →", command=self.next_item).grid(row=0, column=2)
        ttk.Label(right, text="Keys: ←/→ navigate · Ctrl+←/→ while editing notes · 1–5 sets grounding").grid(
            row=6, column=0, sticky="w", pady=(8, 0))

    def current(self) -> dict:
        return self.items[self.index]

    def persist_current(self) -> None:
        if self.loading or not self.items:
            return
        item_id = self.current()["id"]
        self.scores[item_id] = {field: variable.get() for field, variable in self.score_vars.items()}
        self.scores[item_id]["notes"] = self.notes.get("1.0", "end-1c")
        save_scores(self.score_path, self.items, self.scores)

    def show_item(self) -> None:
        item = self.current()
        saved = self.scores.get(item["id"], {})
        self.loading = True
        source = item.get("source_dataset", "blinded source")
        self.heading.configure(text=f"{item['id']}  ·  {source}  ·  {item['task'].replace('_', ' ')}")
        self.reference.configure(state="normal")
        self.reference.delete("1.0", "end")
        self.reference.insert("1.0", item["reference"])
        self.reference.configure(state="disabled")
        for field, variable in self.score_vars.items():
            variable.set(saved.get(field, ""))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", saved.get("notes", ""))
        self.loading = False
        self.position.configure(text=f"{self.index + 1} / {len(self.items)}")
        self._display_image()

    def _display_image(self) -> None:
        relative = self.current().get("image")
        path = self.pack / relative if relative else None
        if not path or not path.exists():
            self.photo = None
            self.image_label.configure(image="", text="Image unavailable from dataset-provided URL")
            self.image_status.configure(text="")
            return
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            width = max(240, self.image_label.winfo_width() - 20)
            height = max(240, self.image_label.winfo_height() - 20)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo, text="")
            self.image_status.configure(text=f"{path.name}  ·  {image.width} × {image.height}")
        except Exception as error:
            self.photo = None
            self.image_label.configure(image="", text=f"Could not render image: {error}")

    def _resize_image(self, _event=None) -> None:
        if self.items:
            self.after_idle(self._display_image)

    def next_item(self, _event=None) -> str:
        self.persist_current()
        self.index = min(self.index + 1, len(self.items) - 1)
        self.show_item()
        return "break"

    def previous_item(self, _event=None) -> str:
        self.persist_current()
        self.index = max(self.index - 1, 0)
        self.show_item()
        return "break"

    def set_score(self, field: str, value: str) -> str:
        if self.focus_get() is not self.notes:
            self.score_vars[field].set(value)
        return "break"

    def close(self) -> None:
        self.persist_current()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=ROOT / "quality_review_pack")
    args = parser.parse_args()
    pack = args.pack.resolve()
    if not (pack / "review_manifest.jsonl").exists():
        raise SystemExit(f"not a review pack: {pack}")
    ReviewApp(pack).mainloop()


if __name__ == "__main__":
    main()
