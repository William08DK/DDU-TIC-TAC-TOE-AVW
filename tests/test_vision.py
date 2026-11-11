from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from vision import debug_detection, detect_debug, draw_debug, orient_frame


class VisionTests(unittest.TestCase):
    def test_debug_uses_small_inference_and_draws_on_the_current_frame(self):
        box = SimpleNamespace(cls=np.array(0), conf=np.array(.9),
                              xyxy=np.array([[40, 60, 100, 120]]))
        model = Mock(return_value=[SimpleNamespace(names={0: "bolle"}, boxes=[box])])
        old_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detect_debug(model, old_frame, .5, "kryds", "bolle")
        self.assertEqual(model.call_args.kwargs["imgsz"], 320)
        current_frame = np.full((400, 400, 3), 70, dtype=np.uint8)
        annotated = draw_debug(current_frame, detections)
        np.testing.assert_array_equal(annotated[350, 350], [70, 70, 70])
        self.assertFalse(np.array_equal(annotated[120, 80], [70, 70, 70]))
        self.assertFalse(np.any(old_frame))
        np.testing.assert_array_equal(current_frame, 70)

    def test_orientation_keeps_pixels_in_expected_positions(self):
        frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(orient_frame(frame, 1), [[4, 1], [5, 2], [6, 3]])
        np.testing.assert_array_equal(orient_frame(frame, 0, True), [[3, 2, 1], [6, 5, 4]])
        np.testing.assert_array_equal(orient_frame(frame, 1, True), [[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(orient_frame(frame, 4), frame)
        self.assertTrue(orient_frame(frame, 1, True).flags.c_contiguous)

    def test_debug_labels_our_crosses_and_circles(self):
        image = np.zeros((300, 600, 3), dtype=np.uint8)
        boxes = [SimpleNamespace(cls=np.array(i), conf=np.array(.9),
                                 xyxy=np.array([[20 + i * 180, 40, 150 + i * 180, 200]]))
                 for i in range(2)]
        result = SimpleNamespace(names={0: "kryds", 1: "bolle"}, boxes=boxes)
        def model(frame, **kwargs):
            return [result]
        annotated, findings = debug_detection(model, image, .5, "kryds", "bolle")
        self.assertEqual(len(findings), 2)
        self.assertIn("Kryds / X", findings[0])
        self.assertIn("Bolle / O", findings[1])
        self.assertTrue(all("90%" in item for item in findings))
        self.assertTrue(np.any(annotated))
        self.assertFalse(np.any(image))  # Originalbilledet til spillet er urørt.
