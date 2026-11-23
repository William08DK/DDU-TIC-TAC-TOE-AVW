"""Webcam -> YOLO -> 3 x 3 bræt -> computerens næste træk."""

import argparse
# Baggrundstråden kan genkende brikker, mens vinduet fortsat viser livevideo.
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from ultralytics import YOLO

from game import Game
from board_detection import BoardTracker
from ui import GameWindow
from vision import detect_debug, draw_debug, orient_frame, predict_pieces

# __file__ er denne fils placering. Stierne virker derfor også, hvis terminalen
# står i en anden mappe. .pt-filen indeholder det, vores YOLO-model har lært.
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "train-demo/runs/detect/nye-brikker/weights/best.pt"
X_CLASS = "kryds"
O_CLASS = "bolle"
SIZE = 600
# Det oprettede bræt vises som 600 x 600 pixels: hvert felt er 200 x 200.


def switch_camera(current, index):
    """Behold det gamle kamera, hvis det valgte ikke leverer billeder."""
    candidate = cv2.VideoCapture(index)
    try:
        # Et kamera skal både kunne åbnes OG levere et billede for at blive valgt.
        success, frame = candidate.read() if candidate.isOpened() else (False, None)
        if not success or frame is None:
            raise ValueError(f"Kamera {index} er ikke tilgængeligt. Prøv et andet nummer.")
    except (ValueError, cv2.error):
        # Luk det fejlslagne forsøg. Det gamle kamera er endnu ikke lukket.
        candidate.release()
        raise
    current.release()  # Skift først væk fra det gamle kamera, når det nye virker.
    return candidate


def read_board(result, x_class, o_class, board_matrix=None):
    """Find brikker i kamerabilledet og overfør deres midtpunkter til brættet."""
    board = [""] * 9
    for box in result.boxes:
        # YOLO giver et klassenummer. names oversætter det til "kryds" eller "bolle".
        name = result.names[int(box.cls.item())]
        # xywhn = midtpunkt x/y og bredde/højde som andele mellem 0 og 1.
        # Vi bruger midten af brikken, så kassens kanter ikke bestemmer feltet.
        x, y, _, _ = box.xywhn[0].tolist()
        if board_matrix is not None:
            # Først omregner vi midtpunktet til kamerapixels. Derefter bruger vi
            # de fire automatisk fundne hjørner til at finde stedet på det oprettede bræt.
            height, width = result.orig_shape
            point = np.float32([[[x * width, y * height]]])
            x, y = cv2.perspectiveTransform(point, board_matrix)[0, 0] / SIZE
        if not (0 <= x < 1 and 0 <= y < 1):
            # En brik ved siden af det fundne bræt deltager ikke i spillet.
            continue
        # int(x * 3) giver kolonnen 0, 1 eller 2. Rækken beregnes på samme måde.
        cell = int(y * 3) * 3 + int(x * 3)
        if name not in (x_class, o_class):
            raise ValueError(f"Ukendt brik: {name}. Brug kun {x_class} og {o_class}.")
        if board[cell]:
            # To fund i ét felt er usikkert. Vi beder om en ny aflæsning frem for at gætte.
            row, col = divmod(cell, 3)
            raise ValueError(f"Flere fund i række {row + 1}, kolonne {col + 1}. Hold brættet frit og stille.")
        board[cell] = "X" if name == x_class else "O"
    return board


def transform(points):
    # Hjørnerne skal omkranse et reelt område og ikke danne en krydset figur.
    source = np.float32(points)
    contour = source.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1000:
        raise ValueError("Brættets placering er usikker. Hold hele brættet synligt og stille.")
    target = np.float32([(0, 0), (SIZE, 0), (SIZE, SIZE), (0, SIZE)])
    # En matrix er her en beregningsopskrift, der retter et skråt bræt op til et kvadrat.
    return cv2.getPerspectiveTransform(source, target)


def read_game_frame(model, frame, matrix, confidence, imgsz):
    # Samme genkendelse som debug; bagefter omsætter vi fundene til de ni spilfelter.
    result = predict_pieces(model, frame, confidence, imgsz)
    board = read_board(result, X_CLASS, O_CLASS, matrix)
    if not any(board):
        # Det er to forskellige problemer: ingen fund eller fund uden for brættet.
        if len(result.boxes):
            raise ValueError("Brikker fundet uden for brættet. Hold hele brættet synligt, og prøv igen.")
        raise ValueError("Ingen brikker genkendt på dette billede. Fjern hænderne, og prøv igen.")
    return board


def draw_board(image, game):
    # Tegn hjælpelinjer og computerens forslag oven på VISNINGSBILLEDET.
    # Denne tegning sendes aldrig tilbage til YOLO som et rigtigt kameraobjekt.
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
    # argparse læser valg fra terminalen, fx --camera 1 eller --confidence 0.6.
    # confidence er den mindste sikkerhed, vi accepterer: 0.5 svarer til 50 %.
    parser = argparse.ArgumentParser(description="Kryds og bolle med webcam og YOLO")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--imgsz", "--debug-imgsz", dest="imgsz", type=int,
                        choices=(320, 480, 640), default=320,
                        help="Fælles analyseopløsning til spil og debug (standard: 320)")
    args = parser.parse_args()
    # Stop med en forklaring, hvis indstillingerne ikke kan bruges.
    if not args.model.is_file():
        parser.error(f"Modellen findes ikke: {args.model}. Se README.md om træning.")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence skal være mellem 0 og 1.")
    if args.camera < 0:
        parser.error("--camera skal være et kameraindeks fra 0 og opefter.")
    model = YOLO(str(args.model))  # Indlæs den trænede model fra disken.
    names = set(model.names.values())
    if names != {X_CLASS, O_CLASS}:
        parser.error("Modellen skal være trænet på vores to klasser: kryds og bolle.")

    cap = cv2.VideoCapture(args.camera)
    # Disse værdier er programmets hukommelse mellem kamerabillederne.
    camera_index = args.camera
    camera_ready = False
    tracker = BoardTracker()  # Finder rammen/gitteret og venter på en stabil placering.
    points = []  # Hjørnerne findes nu automatisk i kamerabilledet.
    matrix = None
    last_board_scan = 0
    game = Game()
    status = ""
    board_image = None  # Det oprettede kvadratiske bræt, som vises i vinduet.
    camera_image = None  # Det rene kamerabillede, som bruges til genkendelse.
    running = True
    rotation = 0
    mirrored = False
    debug_revision = 0  # Forøges ved ændringer, så gamle analyser kan kasseres.
    debug_future = None  # En igangværende analyse, hvis der er en.
    debug_token = None
    last_debug_time = 0
    debug_detections = []
    debug_result_token = None
    debug_result_time = 0
    debug_info = "Indlæser model …"
    debug_model = None
    executor = ThreadPoolExecutor(max_workers=1)  # Højst én debuganalyse ad gangen.

    def analyze_debug(image):
        # nonlocal lader denne indre funktion ændre en variabel fra main().
        nonlocal debug_model
        # Samme modelvægte som spillet; separat instans til baggrundstråden.
        if debug_model is None:
            debug_model = YOLO(str(args.model))
        start = monotonic()
        # monotonic måler forløbet tid og påvirkes ikke af, at computerens ur stilles.
        detections = detect_debug(debug_model, image, args.confidence,
                                  X_CLASS, O_CLASS, args.imgsz)
        return detections, monotonic() - start

    def on_action(action):
        # Alle knapper og tastaturgenveje ender her med et kort navn, fx "read".
        nonlocal cap, camera_index, matrix, game, status, running
        nonlocal rotation, mirrored, debug_revision, board_image, points
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
            tracker.reset()
            matrix = None
            board_image = None
            game = Game()
            status = "Visning ændret. Tøm brættet; programmet finder det automatisk."
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
        elif action == "camera":
            # Menuen viser fx "Kamera 1". Det sidste ord er det nummer, OpenCV bruger.
            selected = int(window.camera.get().split()[-1])
            if selected != camera_index or not camera_ready:
                try:
                    cap = switch_camera(cap, selected)
                    camera_index = selected
                    points.clear()
                    tracker.reset()
                    matrix = None
                    status = ""
                    debug_revision += 1
                    window.clear_debug()
                except (ValueError, cv2.error) as error:
                    status = str(error)
                window.camera.set(f"Kamera {camera_index}")
            window.video.focus_set()
        elif action in ("reset", "calibrate"):
            # Nyt spil nulstiller reglerne. Ny brætsøgning beholder spillets træk.
            debug_revision += 1
            window.clear_debug()
            if action == "reset":
                game = Game()
                status = "Tøm brættet. Læg derefter din første X."
            else:
                points.clear()
                tracker.reset()
                matrix = None
                status = ""
            window.video.focus_set()
        elif action == "read" and camera_ready and tracker.corners is not None and camera_image is not None:
            window.read_button.configure(text="Aflæser …", state="disabled")
            # Vis "Aflæser" med det samme, før genkendelsen begynder at beregne.
            window.root.update_idletasks()
            try:
                # Kontrollér placeringen igen på det billede, vi faktisk vil aflæse.
                # Det forhindrer et træk med gamle koordinater lige efter en flytning.
                found = tracker.update(camera_image)
                if found is None:
                    points = []
                    matrix = None
                    raise ValueError("Brættet er skjult eller i bevægelse. Hold det frit og stille et øjeblik.")
                # Brug den friske måling til reglerne, ikke den støjfiltrerede visningslås.
                points = tracker.measurement.tolist()
                matrix = transform(points)
                # Genkend før perspektivopretningen ændrer brikkernes udseende.
                observed = read_game_frame(model, camera_image, matrix, args.confidence, args.imgsz)
                # Genkendte brikker bliver først til et træk, når spillereglerne godkender det.
                status = game.read(observed)
            except (ValueError, RuntimeError, cv2.error) as error:
                status = str(error)
            window.video.focus_set()

    window = GameWindow(on_action, camera_index)

    def update():
        # Dette er ét gennemløb af kameravisningen. Funktionen planlægger selv det næste.
        nonlocal camera_ready, matrix, board_image, camera_image, status
        nonlocal points, last_board_scan
        nonlocal debug_future, debug_token, last_debug_time
        nonlocal debug_detections, debug_result_token, debug_result_time, debug_info
        if not running:
            return
        tick_started = monotonic()
        try:
            # read() giver både en succesværdi og det nyeste billede fra kameraet.
            success, frame = cap.read() if cap.isOpened() else (False, None)
        except cv2.error:
            success, frame = False, None
        camera_ready = success and frame is not None
        if not camera_ready:
            # Et mørkt reservebillede holder vinduet brugbart, selv når kameraet mangler.
            frame = np.full((SIZE, 900, 3), (35, 50, 44), dtype=np.uint8)
        # Brætsøgning og brikgenkendelse skal bruge samme billede og koordinater.
        if camera_ready:
            frame = orient_frame(frame, rotation, mirrored)
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width * SIZE / height), SIZE))
        camera_image = frame if camera_ready else None
        # Søg løbende, også efter første fund. Tre stabile fund kræves før automatisk zoom.
        # Begræns søgefrekvensen, så brætsøgningen ikke gør kameravisningen tung.
        if camera_ready and monotonic() - last_board_scan >= .12:
            found = tracker.update(frame)
            points = found.tolist() if found is not None else []
            matrix = transform(points) if points else None
            last_board_scan = monotonic()
        elif not camera_ready:
            tracker.mark_missing()
            points = []
            matrix = None
        # Visningslåsen må gerne overleve usikre søgninger. Det er separat fra
        # matrix/ready, og et træk kræver altid en frisk kontrol ovenfor.
        view_matrix = transform(tracker.corners) if tracker.corners is not None else None
        board_image = (cv2.warpPerspective(frame, view_matrix, (SIZE, SIZE))
                       if camera_ready and view_matrix is not None else None)
        display = board_image.copy() if board_image is not None else frame.copy()
        # En kopi gør, at hjælpelinjer ikke bliver tegnet ind i det rene kamerabillede.
        # YOLO ser det rene, orienterede billede, aldrig vores påtegnede spilmarkeringer.
        current_token = (debug_revision,)
        # Dette mærke beskriver den aktuelle opsætning. En analyse fra før et
        # kameraskift må ikke vises på den nye opsætning. Automatisk zoom ændrer
        # ikke debugbilledet, som stadig er det fulde kamerabillede.
        if window.debug_window is not None:
            if not camera_ready:
                window.clear_debug("Intet kamerabillede. Vælg et tilgængeligt kamera i spilvinduet.")
            if debug_future is not None and debug_future.done():
                # Hent kun et færdigt svar; vi venter ikke på baggrundstråden her.
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
                # Send kun et nyt billede, når den forrige analyse er færdig.
                # Dermed opbygges der ikke en kø af gamle billeder, som giver forsinkelse.
                debug_token = current_token
                debug_future = executor.submit(analyze_debug, frame.copy())
                last_debug_time = monotonic()
            if camera_ready:
                fresh = debug_result_token == current_token and monotonic() - debug_result_time < 1
                # Skjul fund fra billeder, der er mere end ét sekund gamle.
                detections = debug_detections if fresh else []
                info = debug_info if debug_result_token == current_token else "Livevideo · afventer genkendelse …"
                if not fresh and debug_result_token == current_token:
                    info += " · afventer nye fund"
                window.show_debug(draw_debug(frame, detections),
                                  [label for label, _ in detections], info)
        elif debug_future is not None and debug_future.done():
            debug_future = None
        if board_image is not None:
            draw_board(display, game)
        elif camera_ready and tracker.previous is not None:
            # Gul kant betyder: brættet er fundet, men placeringen er endnu ikke stabil.
            cv2.polylines(display, [tracker.previous.astype(np.int32)], True, (60, 190, 245), 3)
        window.show(display, game, tracker.corners is not None, camera_ready, status,
                    camera_index, tracking_ready=tracker.ready)
        # Medregn kamera- og tegnetid, så hver opdatering ikke også får 33 ms ventetid.
        window.root.after(max(1, 33 - int((monotonic() - tick_started) * 1000)), update)

    try:
        # Tkinter styrer museklik, taster og de planlagte kald til update().
        window.root.after(0, update)
        window.root.mainloop()
    finally:
        # finally kører også ved fejl, så kameraet bliver frigivet til andre programmer.
        cap.release()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    # Start kun spilvinduet, når filen køres direkte, ikke når tests importerer den.
    try:
        main()
    except (RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f"Fejl: {error}") from error
