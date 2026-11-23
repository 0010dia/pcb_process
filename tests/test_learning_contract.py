"""정규화 계약과 검증/test 접근 제한을 검증한다."""

import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

from src.dataset import make_dataset
from src.model import build_model
from src.predict import prepare_input


class LearningContractTests(unittest.TestCase):
    def test_test_and_validation_augmentation_are_rejected(self):
        with self.assertRaises(ValueError):
            make_dataset('.', 'test')
        with self.assertRaises(ValueError):
            make_dataset('.', 'val', augment=True)

    def test_model_normalizes_once_and_outputs_six_scores(self):
        image = Image.fromarray(np.full((32, 64), 255, dtype=np.uint8))
        inputs = prepare_input(image)
        self.assertEqual(inputs.shape, (1, 128, 128, 1))
        self.assertEqual(float(inputs.min()), 255.0)
        for variant in ['A', 'B']:
            model = build_model(variant)
            normalized = model.get_layer('normalize')(inputs).numpy()
            self.assertTrue(np.allclose(normalized, 1))
            scores = model(inputs, training=False).numpy()
            self.assertEqual(scores.shape, (1, 6))
            self.assertAlmostEqual(float(scores.sum()), 1, places=5)

    def test_graph_and_eager_agree_with_runtime_workaround(self):
        model = build_model('A')
        inputs = tf.reshape(tf.linspace(0.0, 255.0, 128*128), (1, 128, 128, 1))
        eager = model(inputs, training=False).numpy()
        graph = tf.function(lambda batch: model(batch, training=False))(inputs).numpy()
        self.assertTrue(np.allclose(eager, graph, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
