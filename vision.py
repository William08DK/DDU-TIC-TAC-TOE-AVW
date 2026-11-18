"""Kameraorientering og debugvisning af alle modellens fund."""

import cv2
import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

# OpenCV arbejder med kamerabilleder som tabeller af farvede pixels (NumPy-arrays).
# Pillow bruges til at skrive tekst med danske bogstaver oven på billederne.

def orient_frame(frame, rotation=0, mirrored=False):
    """Rotér med uret i kvarte omgange og spejl derefter vandret."""
    # rotation er 0, 1, 2 eller 3. Minus gør, at drejningen sker med uret.
    image = np.rot90(frame, -(rotation % 4))
    if mirrored:
        # ::-1 læser kolonnerne bagfra, så venstre og højre byttes om.
        image = image[:, ::-1]
    # Rotation/spejling kan give en anderledes hukommelsesopbygning.
    # Lav en sammenhængende tabel, som billedbibliotekerne kan arbejde direkte med.
    return np.ascontiguousarray(image)


# Indlæs skrifttypen én gang og genbrug den, i stedet for at læse den for hvert billede.
@lru_cache(maxsize=1)
def debug_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        # Programmet kan stadig vise tekst, hvis Arial ikke findes på computeren.
        return ImageFont.load_default(size=18)


def predict_pieces(model, frame, confidence, imgsz=320):
    """Fælles indstillinger til spillets aflæsning og live-debug."""
    # conf bestemmer minimumssikkerheden, og imgsz bestemmer analyseopløsningen.
    # YOLO returnerer en liste med ét resultat pr. billede; vi sendte kun ét billede.
    return model(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]


def detect_debug(model, frame, confidence, x_class, o_class, imgsz=320):
    """Find vores brikker uden at binde visningen til det gamle analysebillede."""
    result = predict_pieces(model, frame, confidence, imgsz)
    height, width = frame.shape[:2]
    detections = []
    for box in result.boxes:
        # .item() henter et almindeligt tal ud af modellens talbeholder (tensor).
        name = result.names[int(box.cls.item())]
        label = f"Kryds / X ({name})" if name == x_class else (
            f"Bolle / O ({name})" if name == o_class else name)
        label = f"{label} · {float(box.conf.item()):.0%}"
        # xyxy er kassens venstre/top og højre/bund målt i pixels.
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # Gem kassens placering som andele af billedet, så den kan skaleres senere.
        detections.append((label, (x1 / width, y1 / height, x2 / width, y2 / height)))
    return detections


def draw_debug(frame, detections):
    """Tegn de seneste fund oven på det nyeste kamerabillede."""
    if not detections:
        # Ingen tegning er nødvendig; brug det nye kamerabillede med det samme.
        return frame
    # OpenCV gemmer farver i rækkefølgen blå/grøn/rød. Pillow forventer rød/grøn/blå.
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = debug_font()
    for label, bounds in detections:
        # Gør de gemte andele til pixels i det billede, vi viser lige nu.
        x1, y1, x2, y2 = (bounds[0] * image.width, bounds[1] * image.height,
                          bounds[2] * image.width, bounds[3] * image.height)
        draw.rectangle((x1, y1, x2, y2), outline="#56efb5", width=3)
        bounds = draw.textbbox((0, 0), label, font=font)
        # Mål teksten, så den mørke baggrund bliver præcis stor nok.
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(x1, image.width - width - 12))
        y = max(0, y1 - height - 12)
        # Flyt teksten ind på billedet, hvis brikken ligger tæt på kanten.
        draw.rectangle((x, y, x + width + 12, y + height + 10), fill="#142e25")
        draw.text((x + 6, y + 4 - bounds[1]), label, font=font, fill="white")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # Tilbage til OpenCVs farver.


def debug_detection(model, frame, confidence, x_class, o_class):
    # Samlet genvej til analyse og tegning, som blandt andet bruges af vores tests.
    detections = detect_debug(model, frame, confidence, x_class, o_class)
    return draw_debug(frame, detections), [label for label, _ in detections]
