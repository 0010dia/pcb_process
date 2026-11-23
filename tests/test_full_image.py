"""원본 한 장의 입력 계약, 원본 분할과 패치 가중치 복제를 검증한다."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import layers

from src.binary_model import build_binary_model
from src.full_image_data import assert_full_split, full_image_dataset
from src.full_image_model import assemble_full_model, build_full_encoder, build_full_head, unfreeze_last_convolutions
from src.full_image_predict import prepare_full_input
from src.preprocessing import digest


class FullImageTests(unittest.TestCase):
    def test_native_single_image_contract(self):
        image = Image.fromarray(np.full((640, 640, 3), 255, dtype=np.uint8))
        prepared = prepare_full_input(image)
        self.assertEqual(prepared.shape, (1, 640, 640, 1))
        self.assertEqual(float(prepared.min()), 255.0)
        with self.assertRaises(ValueError):
            prepare_full_input(Image.new('L', (128, 128)))

    def test_pairs_stay_together_and_cross_split_pixels_are_blocked(self):
        frame = pd.DataFrame([
            dict(image_id='a0', source_id='a', group='g1', split='train', official_split='trainval', target=0, pixel_sha256='a0'),
            dict(image_id='a1', source_id='a', group='g1', split='train', official_split='trainval', target=1, pixel_sha256='a1'),
            dict(image_id='b0', source_id='b', group='g2', split='val', official_split='trainval', target=0, pixel_sha256='b0'),
            dict(image_id='b1', source_id='b', group='g2', split='val', official_split='trainval', target=1, pixel_sha256='b1'),
        ])
        assert_full_split(frame)
        frame.loc[2, 'pixel_sha256'] = 'a0'
        with self.assertRaises(ValueError):
            assert_full_split(frame)

    def test_test_and_validation_augmentation_are_blocked(self):
        with self.assertRaises(ValueError):
            full_image_dataset('.', 'test')
        with self.assertRaises(ValueError):
            full_image_dataset('.', 'val', augment=True)

    def test_raw_jpeg_loader_matches_inference_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / 'data/split/full_image_v1'
            split.mkdir(parents=True)
            pixels = np.tile(np.arange(640, dtype=np.uint16) % 256, (640, 1)).astype(np.uint8)
            Image.fromarray(pixels).save(root / 'image.jpg')
            pd.DataFrame([dict(image_id='a', image_path='image.jpg', target=0, split='val', official_split='trainval')]).to_csv(split / 'val.csv', index=False)
            dataset, _ = full_image_dataset(root, 'val', batch_size=1)
            loaded, _ = next(iter(dataset))
            with Image.open(root / 'image.jpg') as image:
                expected = prepare_full_input(image)
            self.assertTrue(np.array_equal(loaded.numpy(), expected))

    def test_encoder_uses_cloned_weights_and_keeps_bn_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'patch.keras'
            patch = build_binary_model()
            patch.save(path)
            original = digest(path)
            encoder = build_full_encoder(path)
            self.assertEqual(encoder.input_shape, (None, 640, 640, 1))
            self.assertEqual(encoder.output_shape, (None, 224))
            head = build_full_head(np.zeros((4, 224), dtype=np.float32))
            model = assemble_full_model(encoder, head)
            self.assertEqual(model.output_shape, (None, 1))
            selected = unfreeze_last_convolutions(model, 2)
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(not layer.trainable for layer in encoder.layers if isinstance(layer, layers.BatchNormalization)))
            self.assertEqual(digest(path), original)


if __name__ == '__main__':
    unittest.main()
