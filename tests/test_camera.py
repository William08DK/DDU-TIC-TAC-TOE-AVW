from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from main import read_board, switch_camera, transform


def result(*detections):
    return SimpleNamespace(names={0: "maltese", 1: "circle", 2: "nothing", 3: "star"},
                           boxes=[SimpleNamespace(cls=np.array(c),
                                                  xywhn=np.array([[x, y, .1, .1]]))
                                  for c, x, y in detections])


class CameraTests(unittest.TestCase):
    def test_camera_detections_are_mapped_to_the_marked_board(self):
        # Brættet fylder kun en del af billedet. Andre figurer udenfor ignoreres.
        detected = result((0, .35, .35), (1, .55, .55), (3, .05, .05))
        detected.orig_shape = (1000, 1000)
        matrix = transform([(300, 300), (600, 300), (600, 600), (300, 600)])
        board = read_board(detected, "maltese", "circle", matrix)
        self.assertEqual(board, ["X", "", "", "", "", "", "", "", "O"])

    def test_switch_camera_requires_working_video(self):
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
        board = read_board(result((0, .1, .1), (1, .8, .8), (2, .5, .5)),
                           "maltese", "circle")
        self.assertEqual(board, ["X", "", "", "", "", "", "", "", "O"])

    def test_ambiguous_and_unknown_detections_rejected(self):
        for detections in (((0, .1, .1), (1, .2, .2)), ((3, .5, .5),)):
            with self.assertRaises(ValueError):
                read_board(result(*detections), "maltese", "circle")

    def test_perspective_maps_corners(self):
        import cv2
        corners = [(100, 50), (800, 70), (700, 550), (120, 500)]
        matrix = transform(corners)
        mapped = cv2.perspectiveTransform(np.float32([corners]), matrix)[0]
        np.testing.assert_allclose(mapped, [(0, 0), (600, 0), (600, 600), (0, 600)], atol=.001)
        with self.assertRaises(ValueError):
            transform([(0, 0)] * 4)
