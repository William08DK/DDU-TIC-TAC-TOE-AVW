"""Webcam -> YOLO -> 3 x 3 bræt -> computerens næste træk."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from game import Game

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "train-demo/runs/detect/train/weights/best.pt"
WINDOW = "Kryds og bolle"
SIZE = 600


def load_font():
    """Brug en TrueType-skrift med danske bogstaver."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=18)
        except OSError:
            continue
    return ImageFont.load_default(size=18)


def switch_camera(current, index):
    """Behold det gamle kamera, hvis det valgte ikke leverer billeder."""
    candidate = cv2.VideoCapture(index)
    try:
        success, frame = candidate.read() if candidate.isOpened() else (False, None)
        if not success or frame is None:
            raise ValueError(f"Kamera {index} er ikke tilgængeligt. Prøv et andet nummer.")
    except (ValueError, cv2.error):
        candidate.release()
        raise
    current.release()
    return candidate


def read_board(result, x_class, o_class):
    board = [""] * 9
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        if name == "nothing":
            continue
        x, y, _, _ = box.xywhn[0].tolist()
        if not (0 <= x < 1 and 0 <= y < 1):
            continue
        cell = int(y * 3) * 3 + int(x * 3)
        if name not in (x_class, o_class):
            raise ValueError(f"Ukendt brik: {name}. Brug kun {x_class} og {o_class}.")
        if board[cell]:
            raise ValueError("Flere brikker fundet i samme felt. Prøv igen.")
        board[cell] = "X" if name == x_class else "O"
    return board


def transform(points):
    source = np.float32(points)
    contour = source.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1000:
        raise ValueError("Marker fire hjørner i rækkefølge. K: prøv igen.")
    target = np.float32([(0, 0), (SIZE, 0), (SIZE, SIZE), (0, SIZE)])
    return cv2.getPerspectiveTransform(source, target)


def draw_board(image, game):
    for i in (1, 2):
        cv2.line(image, (i * 200, 0), (i * 200, SIZE), (220, 220, 220), 2)
        cv2.line(image, (0, i * 200), (SIZE, i * 200), (220, 220, 220), 2)
    for i, piece in enumerate(game.board):
        row, col = divmod(i, 3)
        cv2.putText(image, f"{i + 1}: {piece or '-'}", (col * 200 + 12, row * 200 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2)
    if game.pending is not None:
        row, col = divmod(game.pending, 3)
        start = (col * 200 + 6, row * 200 + 6)
        end = ((col + 1) * 200 - 6, (row + 1) * 200 - 6)
        cv2.rectangle(image, start, end, (0, 255, 0), 5)
        cv2.circle(image, (col * 200 + 100, row * 200 + 100), 55, (0, 255, 0), 5)


def main():
    parser = argparse.ArgumentParser(description="Kryds og bolle med webcam og YOLO")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--x-class", default="maltese", help="YOLO-klassen for dine X")
    parser.add_argument("--o-class", default="circle", help="YOLO-klassen for computerens O")
    parser.add_argument("--confidence", type=float, default=.5)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Modellen findes ikke: {args.model}. Se README.md om træning.")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence skal være mellem 0 og 1.")
    if args.camera < 0:
        parser.error("--camera skal være et kameraindeks fra 0 og opefter.")
    model = YOLO(str(args.model))
    names = set(model.names.values())
    if (args.x_class == args.o_class or "nothing" in (args.x_class, args.o_class)
            or not {args.x_class, args.o_class} <= names):
        parser.error(f"Vælg to forskellige brikklasser fra {sorted(names - {'nothing'})}.")

    cap = cv2.VideoCapture(args.camera)
    camera_index = args.camera
    camera_ready = False
    points = []
    matrix = None
    game = Game()
    status = "Klik: øverst venstre, øverst højre, nederst højre, nederst venstre."
    font = load_font()

    def click(event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN and camera_ready
                and len(points) < 4 and y < SIZE):
            points.append((x, y))

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, click)
        while True:
            success, frame = cap.read() if cap.isOpened() else (False, None)
            camera_ready = success and frame is not None
            if not camera_ready:
                frame = np.zeros((SIZE, 900, 3), dtype=np.uint8)
            # Fast visningsstørrelse: museklik og kalibrering bruger samme koordinater.
            frame = cv2.resize(frame, (900, SIZE))
            if camera_ready and len(points) == 4 and matrix is None:
                try:
                    matrix = transform(points)
                    status = "Bræt markeret. Læg din X eller den viste O. Mellemrum: aflæs."
                except ValueError as error:
                    status = str(error)
            board_image = (cv2.warpPerspective(frame, matrix, (SIZE, SIZE))
                           if camera_ready and matrix is not None else None)
            # Gem et rent billede til YOLO inden der tegnes tekst eller markeringer.
            if board_image is not None:
                display = board_image.copy()
                draw_board(display, game)
            else:
                display = frame.copy()
                for i, point in enumerate(points):
                    cv2.circle(display, point, 6, (0, 255, 0), -1)
                    cv2.putText(display, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                                .8, (0, 255, 0), 2)
            canvas = np.zeros((SIZE + 125, 900, 3), dtype=np.uint8)
            canvas[:SIZE, :display.shape[1]] = display
            # Pillow understøtter æ, ø og å i modsætning til OpenCVs standardskrift.
            panel = Image.new("RGB", (900, 125))
            text_draw = ImageDraw.Draw(panel)
            shown_status = status if camera_ready else "Intet kamerabillede. Vælg kamera med tasterne 0–9."
            for i, text in enumerate((shown_status, "Mellemrum: aflæs | R: nyt spil | K: kalibrer | Q/Esc: luk",
                                      f"Kamera {camera_index} | 0–9: skift kamera | X: {args.x_class} | O: {args.o_class}")):
                text_draw.text((12, 12 + i * 32), text, font=font, fill="white")
            canvas[SIZE:] = np.asarray(panel)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if ord("0") <= key <= ord("9"):
                selected = key - ord("0")
                if selected != camera_index or not camera_ready:
                    try:
                        cap = switch_camera(cap, selected)
                        camera_index = selected
                        points.clear()
                        matrix = None
                        status = "Kamera skiftet. Klik hjørner: øv. venstre, øv. højre, ned. højre, ned. venstre."
                    except (ValueError, cv2.error) as error:
                        status = str(error)
            if key in (ord("r"), ord("k")):
                game = Game()
                status = "Tøm brættet. Læg derefter din første X og tryk mellemrum."
                if key == ord("k"):
                    points.clear()
                    matrix = None
                    status = "Klik hjørner: øv. venstre, øv. højre, ned. højre, ned. venstre."
            if key == ord(" ") and board_image is not None:
                try:
                    result = model(board_image, conf=args.confidence, verbose=False)[0]
                    status = game.read(read_board(result, args.x_class, args.o_class))
                except ValueError as error:
                    status = str(error)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f"Fejl: {error}") from error
