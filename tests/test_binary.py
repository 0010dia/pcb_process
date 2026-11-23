"""이진 분류의 정상 영역 샘플링·분할·라벨·지표 계약을 검증한다."""

import unittest

import numpy as np
import pandas as pd

from src.binary_data import assert_binary_split, deduplicate, normal_roi, overlaps_defect
from src.binary_evaluate import binary_scores
from src.binary_model import build_binary_model
from src.binary_train import make_binary_dataset


class BinaryTests(unittest.TestCase):
    def test_normal_roi_avoids_expanded_annotation(self):
        pixels = np.tile(np.array([0, 255], dtype=np.uint8), (64, 32))
        boxes = np.array([[20, 20, 30, 30]])
        rng = np.random.default_rng(42)
        for _ in range(10):
            roi = normal_roi(pixels, boxes, 16, rng, guard=8)
            self.assertIsNotNone(roi)
            self.assertFalse(overlaps_defect(roi, boxes, 8))
        covered = np.array([[0, 0, 63, 63]])
        self.assertIsNone(normal_roi(pixels, covered, 16, rng))

    def test_deduplication_quarantines_cross_split_and_label_conflicts(self):
        frame = pd.DataFrame([
            dict(patch_id='a', split='train', target=0, pixel_sha256='cross'),
            dict(patch_id='b', split='val', target=0, pixel_sha256='cross'),
            dict(patch_id='c', split='train', target=0, pixel_sha256='conflict'),
            dict(patch_id='d', split='train', target=1, pixel_sha256='conflict'),
            dict(patch_id='e', split='train', target=0, pixel_sha256='repeated'),
            dict(patch_id='f', split='train', target=0, pixel_sha256='repeated'),
        ])
        included, excluded = deduplicate(frame)
        self.assertEqual(included.patch_id.tolist(), ['e'])
        self.assertEqual(len(excluded), 5)
        self.assertEqual(set(excluded.exclusion_reason), {
            'identical_pixels_across_splits', 'identical_pixels_conflicting_labels', 'repeated_pixels_within_split'})

    def test_group_and_source_are_shared_by_pairs_only_inside_split(self):
        frame = pd.DataFrame([
            dict(patch_id='a0', source_id='a', group='ga', split='train', official_split='trainval', target=0, pixel_sha256='a0'),
            dict(patch_id='a1', source_id='a', group='ga', split='train', official_split='trainval', target=1, pixel_sha256='a1'),
            dict(patch_id='b0', source_id='b', group='gb', split='val', official_split='trainval', target=0, pixel_sha256='b0'),
            dict(patch_id='b1', source_id='b', group='gb', split='val', official_split='trainval', target=1, pixel_sha256='b1'),
        ])
        assert_binary_split(frame)
        frame.loc[3, 'source_id'] = 'a'
        with self.assertRaises(ValueError):
            assert_binary_split(frame)

    def test_false_positive_and_missed_defect_are_distinguished(self):
        values = binary_scores([0, 0, 0, 1, 1], [.1, .9, .2, .8, .3])
        self.assertEqual(values['false_defect'], 1)
        self.assertEqual(values['missed_defect'], 1)
        self.assertAlmostEqual(values['normal_false_positive_rate'], 1/3)
        self.assertAlmostEqual(values['defect_recall'], .5)

    def test_test_and_validation_augmentation_are_blocked(self):
        with self.assertRaises(ValueError):
            make_binary_dataset('.', 'test')
        with self.assertRaises(ValueError):
            make_binary_dataset('.', 'val', augment=True)

    def test_binary_output_and_single_normalization(self):
        model = build_binary_model()
        inputs = np.full((1, 128, 128, 1), 255, dtype=np.float32)
        self.assertTrue(np.allclose(model.get_layer('normalize')(inputs).numpy(), 1))
        output = model(inputs, training=False).numpy()
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(np.all((0 <= output) & (output <= 1)))


if __name__ == '__main__':
    unittest.main()
