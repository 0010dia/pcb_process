"""Crop 경계와 원본/그룹 분할의 실패 조건을 작은 합성 데이터로 검증한다."""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import CropConfig, assert_separation, crop_patch, neighbor_count, square_roi


class PreprocessingTests(unittest.TestCase):
    def test_edge_crop_contains_endpoint_and_preserves_square(self):
        pixels = np.arange(100, dtype=np.uint8).reshape(10, 10)
        roi = square_roi((0, 0, 3, 5), margin=2)
        self.assertEqual(roi[2]-roi[0], roi[3]-roi[1])
        self.assertLess(roi[0], 0)
        patch, fraction = crop_patch(pixels, roi, 128)
        self.assertEqual(patch.size, (128, 128))
        self.assertEqual(patch.mode, 'L')
        self.assertGreater(fraction, 0)
        self.assertLessEqual(np.asarray(patch).max(), pixels.max())
        self.assertGreaterEqual(roi[2], 4)
        self.assertGreaterEqual(roi[3], 6)

    def test_neighbor_box_intersection_excludes_self(self):
        boxes = np.array([[2, 2, 5, 5], [7, 2, 9, 5], [20, 20, 30, 30]])
        self.assertEqual(neighbor_count(boxes, 0, (0, 0, 8, 8)), 1)

    def sources(self):
        return pd.DataFrame([
            dict(source_id='a', group='g1', split='train', official_split='trainval', test_pixel_sha256='a', temp_pixel_sha256='a'),
            dict(source_id='b', group='g2', split='val', official_split='trainval', test_pixel_sha256='b', temp_pixel_sha256='b'),
            dict(source_id='c', group='g1', split='test', official_split='test', test_pixel_sha256='c', temp_pixel_sha256='c'),
        ])

    def test_group_pixel_and_official_test_guards(self):
        assert_separation(self.sources())
        for column in ['group', 'test_pixel_sha256', 'temp_pixel_sha256']:
            frame = self.sources()
            frame.loc[1, column] = frame.loc[0, column]
            with self.assertRaises(ValueError):
                assert_separation(frame)
        frame = self.sources()
        frame.loc[2, 'split'] = 'train'
        with self.assertRaises(ValueError):
            assert_separation(frame)

    def test_reject_invalid_configuration(self):
        with self.assertRaises(ValueError):
            CropConfig(margin=-1)
        with self.assertRaises(ValueError):
            square_roi((3, 4, 2, 7), 16)


if __name__ == '__main__':
    unittest.main()
