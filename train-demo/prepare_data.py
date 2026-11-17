"""Klargør Make Sense-eksporten uden at ændre originalerne."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "Kryds-der-boller"
LABELS = ROOT / "Kryds-der-boller-data"
OUTPUT = ROOT / "train-demo/nye-brikker"


def photo_group(stem):
    """Hold kamerafotos fra samme 20 sekunder og navngivne kopier samlet."""
    timestamp = re.fullmatch(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_Pro(?: \(\d+\))?", stem)
    if timestamp:
        date, hour, minute, second = timestamp.groups()
        seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
        return f"WIN_{date}_{seconds // 20}"
    number = re.search(r"(\d+)$", stem)
    return f"{stem[:number.start()]}{int(number.group()) // 20}" if number else stem


def assign_splits(groups):
    rng = random.Random(42)
    split, group_split = {}, {}
    for signature, batches in sorted(groups.items()):
        keys = sorted(batches)
        if len(keys) < 2:
            raise ValueError(f"For få adskilte billedserier for klasserne {signature}.")
        rng.shuffle(keys)
        # Vælg hele grupper tæt på 20 % af billederne, ikke 20 % af grupperne.
        possible = {0: ()}
        for key in keys:
            for count, chosen in list(possible.items()):
                possible.setdefault(count + len(batches[key]), chosen + (key,))
        total = sum(len(stems) for stems in batches.values())
        count = min((n for n in possible if 0 < n < total), key=lambda n: abs(n - total * .2))
        validation = set(possible[count])
        for group, stems in batches.items():
            subset = group_split.setdefault(group, "val" if group in validation else "train")
            for stem in stems:
                split[stem] = subset
    return split


def prepare(images_dir=IMAGES, labels_dir=LABELS, output=OUTPUT):
    images = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if path.stem in images:
                raise ValueError(f"Flere billeder med samme navn: {path.stem}")
            images[path.stem] = path
    labels = {}
    for path in labels_dir.rglob("*.txt"):
        if path.stem in ("classes", "labels"):
            continue
        if path.stem in labels:
            raise ValueError(f"Flere label-filer med samme navn: {path.stem}")
        labels[path.stem] = path
    if not images or set(images) != set(labels):
        raise ValueError("Billeder og labels skal matche én til én. Tjek filnavnene.")
    if output.exists():
        raise FileExistsError(f"Datasættet findes allerede: {output}. Intet er overskrevet.")
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
        group = photo_group(stem)
        groups[tuple(sorted(set(classes)))][group].append(stem)
        records[stem] = {"classes": classes, "group": group, "label": text}
    split = assign_splits(groups)
    output.mkdir(parents=True)
    for subset in ("train", "val"):
        (output / "images" / subset).mkdir(parents=True)
        (output / "labels" / subset).mkdir(parents=True)
    for stem, subset in sorted(split.items()):
        with Image.open(images[stem]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            image.save(output / "images" / subset / f"{stem}.jpg", quality=95)
        (output / "labels" / subset / f"{stem}.txt").write_text(records[stem]["label"], encoding="utf-8")
    (output / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: kryds\n  1: bolle\n", encoding="utf-8")
    manifest = {"seed": 42, "grouping": "camera_20_seconds_or_numbered_20_photos",
                "source_images": str(images_dir.resolve().relative_to(ROOT)),
                "source_labels": str(labels_dir.resolve().relative_to(ROOT)),
                "names": {0: "kryds", 1: "bolle"},
                "files": {stem: {"split": subset, "group": records[stem]["group"],
                                  "classes": records[stem]["classes"]}
                          for stem, subset in sorted(split.items())}}
    (output / "split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for subset in ("train", "val"):
        members = [stem for stem in split if split[stem] == subset]
        counts = Counter(c for stem in members for c in records[stem]["classes"])
        print(f"{subset}: {len(members)} billeder, brikker: {dict(counts)}", flush=True)
    print(f"Klar: {output / 'data.yaml'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.images, args.labels, args.output)
