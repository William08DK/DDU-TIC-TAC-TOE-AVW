"""Kameraorientering og debugvisning af alle modellens fund."""

import cv2
import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont


def orient_frame(frame, rotation=0, mirrored=False):
    """Rotér med uret i kvarte omgange og spejl derefter vandret."""
    image = np.rot90(frame, -(rotation % 4))
    if mirrored:
        image = image[:, ::-1]
    return np.ascontiguousarray(image)


@lru_cache(maxsize=1)
def debug_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default(size=18)


def detect_debug(model, frame, confidence, x_class, o_class, imgsz=320):
    """Find vores brikker uden at binde visningen til det gamle analysebillede."""
    result = model(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
    height, width = frame.shape[:2]
    detections = []
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        label = f"Kryds / X ({name})" if name == x_class else (
            f"Bolle / O ({name})" if name == o_class else name)
        label = f"{label} · {float(box.conf.item()):.0%}"
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append((label, (x1 / width, y1 / height, x2 / width, y2 / height)))
    return detections


def draw_debug(frame, detections):
    """Tegn de seneste fund oven på det nyeste kamerabillede."""
    if not detections:
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = debug_font()
    for label, bounds in detections:
        x1, y1, x2, y2 = (bounds[0] * image.width, bounds[1] * image.height,
                          bounds[2] * image.width, bounds[3] * image.height)
        draw.rectangle((x1, y1, x2, y2), outline="#56efb5", width=3)
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(x1, image.width - width - 12))
        y = max(0, y1 - height - 12)
        draw.rectangle((x, y, x + width + 12, y + height + 10), fill="#142e25")
        draw.text((x + 6, y + 4 - bounds[1]), label, font=font, fill="white")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def debug_detection(model, frame, confidence, x_class, o_class):
    detections = detect_debug(model, frame, confidence, x_class, o_class)
    return draw_debug(frame, detections), [label for label, _ in detections]
