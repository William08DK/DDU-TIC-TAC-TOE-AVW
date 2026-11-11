import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[1] / "train-demo/prepare_data.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class PreparationTests(unittest.TestCase):
    def test_splits_keep_photo_groups_together_across_classes(self):
        groups = {
            (0,): {"series1": ["a", "b"], "series2": ["c"], "shared": ["d"]},
            (1,): {"series3": ["e"], "series4": ["f"], "shared": ["g"]},
        }
        split = preparation.assign_splits(groups)
        self.assertEqual(set(split), set("abcdefg"))
        self.assertEqual(split["a"], split["b"])
        self.assertEqual(split["d"], split["g"])
        self.assertEqual(split, preparation.assign_splits(groups))
        self.assertEqual(set(split.values()), {"train", "val"})

    def test_single_photo_group_cannot_be_split_honestly(self):
        with self.assertRaises(ValueError):
            preparation.assign_splits({(0,): {"only_series": ["a", "b"]}})
