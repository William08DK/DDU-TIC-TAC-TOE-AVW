"""Webcam -> YOLO -> 3 x 3 bræt -> computerens næste træk."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from ultralytics import YOLO

from game import Game
from ui import GameWindow
from vision import detect_debug, draw_debug, orient_frame

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "train-demo/runs/detect/train/weights/best.pt"
SIZE = 600


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


def read_board(result, x_class, o_class, board_matrix=None):
    """Find brikker i kamerabilledet og overfør deres midtpunkter til brættet."""
    board = [""] * 9
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        if name == "nothing":
            continue
        x, y, _, _ = box.xywhn[0].tolist()
        if board_matrix is not None:
            height, width = result.orig_shape
            point = np.float32([[[x * width, y * height]]])
            x, y = cv2.perspectiveTransform(point, board_matrix)[0, 0] / SIZE
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
        cv2.line(image, (i * 200, 0), (i * 200, SIZE), (220, 230, 225), 1)
        cv2.line(image, (0, i * 200), (SIZE, i * 200), (220, 230, 225), 1)
    if game.pending is not None:
        row, col = divmod(game.pending, 3)
        start = (col * 200 + 6, row * 200 + 6)
        end = ((col + 1) * 200 - 6, (row + 1) * 200 - 6)
        cv2.rectangle(image, start, end, (0, 255, 0), 5)
        cv2.circle(image, (col * 200 + 100, row * 200 + 100), 55, (0, 255, 0), 5)


def main():
    parser = argparse.ArgumentParser(description="Kryds og bolle med webcam og YOLO")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--x-class", default="maltese", help="YOLO-klassen for dine X")
    parser.add_argument("--o-class", default="circle", help="YOLO-klassen for computerens O")
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--debug-model", type=Path, default=ROOT / "yolo26n.pt",
                        help="Model til debugvisningens hverdagsting")
    parser.add_argument("--debug-imgsz", type=int, choices=(320, 480, 640), default=320,
                        help="Debugopløsning: 320 er hurtigst, 640 giver flere detaljer")
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
    status = ""
    board_image = None
    camera_image = None
    running = True
    rotation = 0
    mirrored = False
    debug_revision = 0
    debug_future = None
    debug_token = None
    last_debug_time = 0
    debug_detections = []
    debug_result_token = None
    debug_result_time = 0
    debug_info = "Indlæser model …"
    debug_models = {}
    executor = ThreadPoolExecutor(max_workers=1)

    def analyze_debug(path, image):
        # Separate modeller i én baggrundstråd holder liveanalyse adskilt fra spillet.
        if path not in debug_models:
            debug_models[path] = YOLO(str(path))
        start = monotonic()
        detections = detect_debug(debug_models[path], image, args.confidence,
                                  args.x_class, args.o_class, args.debug_imgsz)
        return detections, monotonic() - start

    def on_point(x, y):
        if camera_ready and matrix is None and len(points) < 4:
            points.append((int(x), int(y)))

    def on_action(action):
        nonlocal cap, camera_index, matrix, game, status, running
        nonlocal rotation, mirrored, debug_revision, board_image
        if action == "quit":
            running = False
            window.root.destroy()
        elif action in ("rotate", "mirror"):
            if action == "rotate":
                rotation = (rotation + 1) % 4
            else:
                mirrored = not mirrored
            # Ny orientering ændrer felternes placering: start sikkert fra et tomt bræt.
            points.clear()
            matrix = None
            board_image = None
            game = Game()
            status = "Visning ændret. Tøm brættet, og markér de fire hjørner igen."
            window.rotate_button.configure(text=f"Rotér 90° · {rotation * 90}°")
            window.mirror_button.configure(text="Spejlvendt ✓" if mirrored else "Spejlvend")
            debug_revision += 1
            window.clear_debug()
            window.video.focus_set()
        elif action == "debug":
            debug_revision += 1
            if window.debug_window is None:
                window.open_debug()
            else:
                window.close_debug()
        elif action == "debug_mode":
            debug_revision += 1
            window.clear_debug("Indlæser den valgte model …")
        elif action == "camera":
            selected = int(window.camera.get().split()[-1])
            if selected != camera_index or not camera_ready:
                try:
                    cap = switch_camera(cap, selected)
                    camera_index = selected
                    points.clear()
                    matrix = None
                    status = ""
                    debug_revision += 1
                    window.clear_debug()
                except (ValueError, cv2.error) as error:
                    status = str(error)
                window.camera.set(f"Kamera {camera_index}")
            window.video.focus_set()
        elif action in ("reset", "calibrate"):
            debug_revision += 1
            window.clear_debug()
            if action == "reset":
                game = Game()
                status = "Tøm brættet. Læg derefter din første X."
            else:
                points.clear()
                matrix = None
                status = ""
            window.video.focus_set()
        elif action == "read" and camera_ready and matrix is not None and camera_image is not None:
            window.read_button.configure(text="Aflæser …", state="disabled")
            window.root.update_idletasks()
            try:
                # Genkend før perspektivopretningen ændrer brikkernes udseende.
                result = model(camera_image, imgsz=640, conf=args.confidence, verbose=False)[0]
                status = game.read(read_board(result, args.x_class, args.o_class, matrix))
            except (ValueError, RuntimeError, cv2.error) as error:
                status = str(error)
            window.video.focus_set()

    window = GameWindow(on_action, on_point, camera_index)

    def update():
        nonlocal camera_ready, matrix, board_image, camera_image, status
        nonlocal debug_future, debug_token, last_debug_time
        nonlocal debug_detections, debug_result_token, debug_result_time, debug_info
        if not running:
            return
        tick_started = monotonic()
        try:
            success, frame = cap.read() if cap.isOpened() else (False, None)
        except cv2.error:
            success, frame = False, None
        camera_ready = success and frame is not None
        if not camera_ready:
            frame = np.full((SIZE, 900, 3), (35, 50, 44), dtype=np.uint8)
        # Bevar billedets proportioner; museklik omregnes fra vinduet i ui.py.
        if camera_ready:
            frame = orient_frame(frame, rotation, mirrored)
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width * SIZE / height), SIZE))
        camera_image = frame if camera_ready else None
        if camera_ready and len(points) == 4 and matrix is None:
            try:
                matrix = transform(points)
                status = ""
            except ValueError as error:
                status = str(error)
        board_image = (cv2.warpPerspective(frame, matrix, (SIZE, SIZE))
                       if camera_ready and matrix is not None else None)
        display = board_image.copy() if board_image is not None else frame.copy()
        # YOLO ser det rene, orienterede billede, aldrig vores påtegnede spilmarkeringer.
        current_token = (debug_revision, tuple(points), matrix is not None)
        if window.debug_window is not None:
            if not camera_ready:
                window.clear_debug("Intet kamerabillede. Vælg et tilgængeligt kamera i spilvinduet.")
            if debug_future is not None and debug_future.done():
                try:
                    detections, duration = debug_future.result()
                    if camera_ready and debug_token == current_token:
                        debug_detections = detections
                        debug_result_token = current_token
                        debug_result_time = last_debug_time
                        debug_info = f"Livevideo · analyse {duration * 1000:.0f} ms"
                except Exception as error:
                    if debug_token == current_token:
                        debug_info = f"Kunne ikke analysere billedet: {error}"
                        debug_detections = []
                        debug_result_token = current_token
                        debug_result_time = 0
                debug_future = None
            if camera_ready and debug_future is None and monotonic() - last_debug_time >= .08:
                path = args.debug_model if window.debug_mode.get() == "Hverdagsting" else args.model
                if path.is_file():
                    debug_token = current_token
                    debug_future = executor.submit(analyze_debug, path,
                                                   (board_image if board_image is not None else frame).copy())
                else:
                    debug_info = f"Modellen findes ikke: {path}. Vælg Spilbrikker eller angiv --debug-model."
                    debug_result_token = current_token
                    debug_detections = []
                    debug_result_time = 0
                last_debug_time = monotonic()
            if camera_ready:
                fresh = debug_result_token == current_token and monotonic() - debug_result_time < 1
                detections = debug_detections if fresh else []
                source = board_image if board_image is not None else frame
                info = debug_info if debug_result_token == current_token else "Livevideo · afventer genkendelse …"
                if not fresh and debug_result_token == current_token:
                    info += " · afventer nye fund"
                window.show_debug(draw_debug(source, detections),
                                  [label for label, _ in detections], info)
        elif debug_future is not None and debug_future.done():
            debug_future = None
        if board_image is not None:
            draw_board(display, game)
        elif camera_ready:
            for i, point in enumerate(points):
                cv2.circle(display, point, 12, (100, 220, 170), -1)
                cv2.putText(display, str(i + 1), (point[0] - 5, point[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 50, 40), 2)
        window.show(display, game, matrix is not None, points, camera_ready, status, camera_index)
        # Medregn kamera- og tegnetid, så hver opdatering ikke også får 33 ms ventetid.
        window.root.after(max(1, 33 - int((monotonic() - tick_started) * 1000)), update)

    try:
        window.root.after(0, update)
        window.root.mainloop()
    finally:
        cap.release()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f"Fejl: {error}") from error
