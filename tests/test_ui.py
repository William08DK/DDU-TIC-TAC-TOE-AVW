from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ui import GameWindow


# Vi afprøver kun klik/taster her og åbner ikke et rigtigt vindue.
# __new__ laver objektet uden at køre __init__, som ellers bygger hele brugerfladen.
class WindowInteractionTests(unittest.TestCase):
    def test_corner_click_maps_resized_image_and_ignores_margins(self):
        # Et klik midt på det nedskalerede billede skal blive midten af originalbilledet.
        window = GameWindow.__new__(GameWindow)
        window.on_point = Mock()
        # Et 900 x 600-billede vises i 450 x 300 med sorte kanter.
        window.image_bounds = (20, 40, 450, 300, 900, 600)
        window.click_video(SimpleNamespace(x=245, y=190))
        window.on_point.assert_called_once_with(450, 300)
        window.click_video(SimpleNamespace(x=10, y=190))
        self.assertEqual(window.on_point.call_count, 1)

    def test_shortcuts_select_camera_and_trigger_read(self):
        # Tasten 1 skal vælge kamera 1, og mellemrum skal sende handlingen "read".
        window = GameWindow.__new__(GameWindow)
        window.camera = Mock()
        window.on_action = Mock()
        window.keypress(SimpleNamespace(widget=None, keysym="1", char="1"))
        window.camera.set.assert_called_once_with("Kamera 1")
        window.on_action.assert_called_with("camera")
        window.keypress(SimpleNamespace(widget=None, keysym="space", char=" "))
        window.on_action.assert_called_with("read")
