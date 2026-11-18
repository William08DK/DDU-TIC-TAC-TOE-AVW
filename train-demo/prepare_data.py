"""Klargør Make Sense-eksporten uden at ændre originalerne."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageOps

# Originalbilleder og Make Sense-eksporten holdes adskilt fra træningskopierne.
# ROOT går to niveauer op fra denne fil og ender i projektets hovedmappe.
ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "Kryds-der-boller"
LABELS = ROOT / "Kryds-der-boller-data"
OUTPUT = ROOT / "train-demo/nye-brikker"


def photo_group(stem):
    """Hold kamerafotos fra samme 20 sekunder og navngivne kopier samlet."""
    # Windows-kameraets filnavn indeholder dato og klokkeslæt. Udtrykket genkender
    # også kopier som "..._Pro (2)", så kopien ikke havner i en anden datagruppe.
    timestamp = re.fullmatch(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_Pro(?: \(\d+\))?", stem)
    if timestamp:
        date, hour, minute, second = timestamp.groups()
        seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
        # Heltalsdivision med 20 giver samme gruppenummer inden for samme tidsblok.
        return f"WIN_{date}_{seconds // 20}"
    # Ældre telefonbilleder kan hedde IMG_7739. Her samles 20 filnumre ad gangen.
    number = re.search(r"(\d+)$", stem)
    return f"{stem[:number.start()]}{int(number.group()) // 20}" if number else stem


def assign_splits(groups):
    # train er billeder, modellen lærer af. val er billeder, der bruges til kontrol.
    # Tallet 42 gør den tilfældige rækkefølge reproducerbar ved samme input.
    rng = random.Random(42)
    split, group_split = {}, {}
    for signature, batches in sorted(groups.items()):
        # signature fortæller, hvilke klasser en gruppe indeholder, fx (0, 1).
        keys = sorted(batches)
        if len(keys) < 2:
            # Med kun én gruppe kan vi ikke både træne og kontrollere på adskilte grupper.
            raise ValueError(f"For få adskilte billedserier for klasserne {signature}.")
        rng.shuffle(keys)
        # Vælg hele grupper tæt på 20 % af billederne, ikke 20 % af grupperne.
        possible = {0: ()}
        for key in keys:
            # Husk de antal billeder, vi kan vælge ved at tage hele grupper.
            # Fx kan grupper på 5 og 8 billeder give 0, 5, 8 eller 13 billeder.
            for count, chosen in list(possible.items()):
                possible.setdefault(count + len(batches[key]), chosen + (key,))
        total = sum(len(stems) for stems in batches.values())
        count = min((n for n in possible if 0 < n < total), key=lambda n: abs(n - total * .2))
        # Vælg den kombination, hvis billedantal ligger nærmest 20 % af totalen.
        validation = set(possible[count])
        for group, stems in batches.items():
            subset = group_split.setdefault(group, "val" if group in validation else "train")
            # setdefault genbruger en tidligere beslutning, hvis gruppen optræder igen.
            # Dermed kan samme billedserie ikke placeres i begge dele af datasættet.
            for stem in stems:
                split[stem] = subset
    return split


def prepare(images_dir=IMAGES, labels_dir=LABELS, output=OUTPUT):
    # Brug filnavnet uden endelse som nøgle: billede.jpg skal passe til billede.txt.
    images = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if path.stem in images:
                raise ValueError(f"Flere billeder med samme navn: {path.stem}")
            images[path.stem] = path
    labels = {}
    for path in labels_dir.rglob("*.txt"):
        if path.stem in ("classes", "labels"):
            # Disse filer kan beskrive klassenavne; de er ikke markeringer til et billede.
            continue
        if path.stem in labels:
            raise ValueError(f"Flere label-filer med samme navn: {path.stem}")
        labels[path.stem] = path
    if not images or set(images) != set(labels):
        # Stop, hvis noget mangler, frem for at træne med ufuldstændige data.
        raise ValueError("Billeder og labels skal matche én til én. Tjek filnavnene.")
    if output.exists():
        # En tidligere klargøring må ikke blive overskrevet ved et uheld.
        raise FileExistsError(f"Datasættet findes allerede: {output}. Intet er overskrevet.")
    # defaultdict opretter automatisk en ny tom gruppe, når et navn bruges første gang.
    groups = defaultdict(lambda: defaultdict(list))
    records = {}
    for stem in sorted(images):
        # utf-8-sig kan også læse eksporter med et usynligt UTF-8-mærke i starten.
        text = labels[stem].read_text(encoding="utf-8-sig")
        classes = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if len(line.split()) != 5:
                raise ValueError(f"Ugyldig YOLO-linje i {stem}: {line}")
            # Hver YOLO-linje: klasse, midtpunkt x/y, bredde og højde.
            # Koordinaterne er andele af billedet, ikke pixels. Derfor er de mellem 0 og 1.
            c, x, y, w, h = map(float, line.split())
            # Kontrollér også, at kassens kanter ligger inden for billedet.
            # Den lille tolerance tillader afrunding fra eksportværktøjet.
            if (c not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                    or x - w / 2 < -.002 or y - h / 2 < -.002
                    or x + w / 2 > 1.002 or y + h / 2 > 1.002):
                raise ValueError(f"Ugyldig klasse eller koordinat i {stem}: {line}")
            classes.append(int(c))
        group = photo_group(stem)
        groups[tuple(sorted(set(classes)))][group].append(stem)
        records[stem] = {"classes": classes, "group": group, "label": text}
    # Opret først filer, når alle labels har bestået kontrollen ovenfor.
    split = assign_splits(groups)
    output.mkdir(parents=True)
    for subset in ("train", "val"):
        (output / "images" / subset).mkdir(parents=True)
        (output / "labels" / subset).mkdir(parents=True)
    for stem, subset in sorted(split.items()):
        with Image.open(images[stem]) as original:
            # Ret eventuel kamerarotation fra billedets metadata og brug almindelige RGB-farver.
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            # Gem en mindre kopi. Originalen bliver hverken flyttet eller ændret.
            image.save(output / "images" / subset / f"{stem}.jpg", quality=95)
        # De normaliserede koordinater passer stadig efter skalering med samme proportioner.
        (output / "labels" / subset / f"{stem}.txt").write_text(records[stem]["label"], encoding="utf-8")
    # Dette er den opskrift, YOLO læser, når model-train.py starter træningen.
    (output / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: kryds\n  1: bolle\n", encoding="utf-8")
    # Gem en oversigt, så vi senere kan se præcis, hvordan hvert billede blev fordelt.
    manifest = {"seed": 42, "grouping": "camera_20_seconds_or_numbered_20_photos",
                "source_images": str(images_dir.resolve().relative_to(ROOT)),
                "source_labels": str(labels_dir.resolve().relative_to(ROOT)),
                "names": {0: "kryds", 1: "bolle"},
                "files": {stem: {"split": subset, "group": records[stem]["group"],
                                  "classes": records[stem]["classes"]}
                          for stem, subset in sorted(split.items())}}
    (output / "split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for subset in ("train", "val"):
        # Tæl både billeder og de enkelte markerede brikker i hver del af datasættet.
        members = [stem for stem in split if split[stem] == subset]
        counts = Counter(c for stem in members for c in records[stem]["classes"])
        print(f"{subset}: {len(members)} billeder, brikker: {dict(counts)}", flush=True)
    print(f"Klar: {output / 'data.yaml'}", flush=True)


if __name__ == "__main__":
    # Standardmapperne kan erstattes fra terminalen med --images, --labels og --output.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.images, args.labels, args.output)
