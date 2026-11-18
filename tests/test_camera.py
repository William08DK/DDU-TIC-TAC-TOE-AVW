from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Mock er en erstatning, vi selv styrer. Tests kan derfor bruge et tænkt kamera
# og tænkte YOLO-fund uden at kræve en tilsluttet USB-enhed eller en modelanalyse.
import numpy as np

from main import read_board, read_game_frame, switch_camera, transform
from game import Game
from vision import detect_debug


def result(*detections):
    # Byg kun de egenskaber fra et YOLO-resultat, som read_board faktisk bruger.
    return SimpleNamespace(names={0: "kryds", 1: "bolle", 2: "nothing", 3: "star"},
                           boxes=[SimpleNamespace(cls=np.array(c),
                                                  xywhn=np.array([[x, y, .1, .1]]))
                                  for c, x, y in detections])


class CameraTests(unittest.TestCase):
    def test_game_uses_the_same_inference_settings_as_working_debug(self):
        # Beskyt mod fejlen, hvor debug kunne se en brik, men spillet analyserede anderledes.
        detected = result((0, 1 / 6, 1 / 6))
        detected.orig_shape = (600, 600)
        detected.boxes[0].conf = np.array(.95)
        detected.boxes[0].xyxy = np.array([[50, 50, 150, 150]])
        # Genskab en model, der finder brikken ved 320, men overser den ved 640.
        model = Mock(side_effect=lambda frame, **kw: [detected if kw['imgsz'] == 320 else result()])
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        debug = detect_debug(model, frame, .5, "kryds", "bolle", 320)
        board = read_game_frame(model, frame, np.eye(3), .5, 320)
        self.assertEqual(len(debug), 1)
        self.assertEqual(board, ["X", "", "", "", "", "", "", "", ""])
        game = Game()
        game.read(board)
        self.assertEqual(game.pending, 4)
        self.assertEqual(model.call_args_list[0].kwargs, model.call_args_list[1].kwargs)

    def test_no_detections_and_wrong_calibration_have_distinct_messages(self):
        # Fejlbeskeden skal vise, om modellen eller placeringen af brættet er problemet.
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        model = Mock(return_value=[result()])
        with self.assertRaisesRegex(ValueError, "Ingen brikker genkendt"):
            read_game_frame(model, frame, np.eye(3), .5, 320)
        outside = result((0, .9, .9))
        outside.orig_shape = (600, 600)
        model.return_value = [outside]
        matrix = transform([(100, 100), (300, 100), (300, 300), (100, 300)])
        with self.assertRaisesRegex(ValueError, "uden for"):
            read_game_frame(model, frame, matrix, .5, 320)

    def test_camera_detections_are_mapped_to_the_marked_board(self):
        # Brættet fylder kun en del af billedet. Andre figurer udenfor ignoreres.
        detected = result((0, .35, .35), (1, .55, .55), (3, .05, .05))
        detected.orig_shape = (1000, 1000)
        matrix = transform([(300, 300), (600, 300), (600, 600), (300, 600)])
        board = read_board(detected, "kryds", "bolle", matrix)
        self.assertEqual(board, ["X", "", "", "", "", "", "", "", "O"])

    def test_switch_camera_requires_working_video(self):
        # Et skift skal frigive det gamle kamera, når det nye har leveret et billede.
        current = Mock()
        candidate = Mock()
        candidate.isOpened.return_value = True
        candidate.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        with patch("main.cv2.VideoCapture", return_value=candidate) as capture:
            self.assertIs(switch_camera(current, 1), candidate)
        capture.assert_called_once_with(1)
        current.release.assert_called_once()
        candidate.release.assert_not_called()

    def test_failed_camera_switch_keeps_current_camera(self):
        # Prøv både et kamera, der ikke åbner, og et der åbner uden at levere billeder.
        for opened in (False, True):
            with self.subTest(opened=opened):
                current = Mock()
                candidate = Mock()
                candidate.isOpened.return_value = opened
                candidate.read.return_value = (False, None)
                with patch("main.cv2.VideoCapture", return_value=candidate):
                    with self.assertRaises(ValueError):
                        switch_camera(current, 1)
                current.release.assert_not_called()
                candidate.release.assert_called_once()

    def test_detection_positions(self):
        # Midtpunkter i øverste venstre og nederste højre felt skal blive felt 0 og 8.
        board = read_board(result((0, .1, .1), (1, .8, .8)),
                           "kryds", "bolle")
        self.assertEqual(board, ["X", "", "", "", "", "", "", "", "O"])

    def test_ambiguous_and_unknown_detections_rejected(self):
        # To brikker i samme felt eller en ukendt klasse må ikke godkendes stiltiende.
        for detections in (((0, .1, .1), (1, .2, .2)), ((3, .5, .5),)):
            with self.assertRaises(ValueError):
                read_board(result(*detections), "kryds", "bolle")

    def test_perspective_maps_corners(self):
        # Selv når brættet ligger skråt, skal hjørnerne ende på et 600 x 600-kvadrat.
        import cv2
        corners = [(100, 50), (800, 70), (700, 550), (120, 500)]
        matrix = transform(corners)
        mapped = cv2.perspectiveTransform(np.float32([corners]), matrix)[0]
        np.testing.assert_allclose(mapped, [(0, 0), (600, 0), (600, 600), (0, 600)], atol=.001)
        with self.assertRaises(ValueError):
            transform([(0, 0)] * 4)
