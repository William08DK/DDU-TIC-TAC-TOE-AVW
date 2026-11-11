"""Klargør Make Sense-eksporten uden at ændre originalerne."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "Kryds-og-bolle-billeder"
LABELS = ROOT / "labels_my-project-name_2026-11-11-10-08-19"
OUTPUT = ROOT / "train-demo/egne-brikker"


def assign_splits(groups):
    rng = random.Random(42)
    split, group_split = {}, {}
    for signature, batches in sorted(groups.items()):
        keys = sorted(batches)
        if len(keys) < 2:
            raise ValueError(f"For få adskilte billedserier for klasserne {signature}.")
        rng.shuffle(keys)
        validation = set(keys[:max(1, min(len(keys) - 1, round(len(keys) * .2)))])
        for group, stems in batches.items():
            subset = group_split.setdefault(group, "val" if group in validation else "train")
            for stem in stems:
                split[stem] = subset
    return split


def prepare():
    images = {}
    for path in IMAGES.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if path.stem in images:
                raise ValueError(f"Flere billeder med samme navn: {path.stem}")
            images[path.stem] = path
    labels = {p.stem: p for p in LABELS.rglob("*.txt")}
    if not images or set(images) != set(labels):
        raise ValueError("Billeder og labels skal matche én til én. Tjek filnavnene.")
    if OUTPUT.exists():
        raise FileExistsError(f"Datasættet findes allerede: {OUTPUT}. Intet er overskrevet.")
    groups = defaultdict(lambda: defaultdict(list))
    records = {}
    for stem in sorted(images):
        text = labels[stem].read_text(encoding="utf-8-sig")
        classes = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if len(line.split()) != 5:
                raise ValueError(f"Ugyldig YOLO-linje i {stem}: {line}")
            c, x, y, w, h = map(float, line.split())
            if (c not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                    or x - w / 2 < -.002 or y - h / 2 < -.002
                    or x + w / 2 > 1.002 or y + h / 2 > 1.002):
                raise ValueError(f"Ugyldig klasse eller koordinat i {stem}: {line}")
            classes.append(int(c))
        # Hold nabofotos samlet i blokke på 20 for at mindske lækage fra billedserier.
        number = re.search(r"(\d+)$", stem)
        group = f"{stem[:number.start()]}{int(number.group()) // 20}" if number else stem
        groups[tuple(sorted(set(classes)))][group].append(stem)
        records[stem] = {"classes": classes, "group": group, "label": text}
    split = assign_splits(groups)
    OUTPUT.mkdir(parents=True)
    for subset in ("train", "val"):
        (OUTPUT / "images" / subset).mkdir(parents=True)
        (OUTPUT / "labels" / subset).mkdir(parents=True)
    for stem, subset in sorted(split.items()):
        with Image.open(images[stem]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            image.save(OUTPUT / "images" / subset / f"{stem}.jpg", quality=95)
        (OUTPUT / "labels" / subset / f"{stem}.txt").write_text(records[stem]["label"], encoding="utf-8")
    (OUTPUT / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: kryds\n  1: bolle\n", encoding="utf-8")
    manifest = {"seed": 42, "group_size": 20, "names": {0: "kryds", 1: "bolle"},
                "files": {stem: {"split": subset, "group": records[stem]["group"],
                                  "classes": records[stem]["classes"]}
                          for stem, subset in sorted(split.items())}}
    (OUTPUT / "split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for subset in ("train", "val"):
        members = [stem for stem in split if split[stem] == subset]
        counts = Counter(c for stem in members for c in records[stem]["classes"])
        print(f"{subset}: {len(members)} billeder, brikker: {dict(counts)}", flush=True)
    print(f"Klar: {OUTPUT / 'data.yaml'}", flush=True)


if __name__ == "__main__":
    prepare()
