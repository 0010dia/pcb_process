"""처음부터 학습하는 원본 모델이 사전학습 파일을 읽지 않는지 확인한다."""

import unittest
from unittest.mock import patch

import numpy as np
from tensorflow.keras import layers

from src.full_scratch import build_scratch_model


class ScratchModelTests(unittest.TestCase):
    def test_no_pretrained_model_is_loaded_and_all_weights_are_trainable(self):
        with patch('src.full_scratch.keras.models.load_model', side_effect=AssertionError('checkpoint 로드 금지')):
            model = build_scratch_model()
        self.assertEqual(model.input_shape, (None, 640, 640, 1))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(len(model.non_trainable_weights), 0)
        self.assertTrue(any(isinstance(layer, layers.GroupNormalization) for layer in model.layers))
        self.assertFalse(any(isinstance(layer, layers.BatchNormalization) for layer in model.layers))
        white = np.full((1, 1, 1, 1), 255, dtype=np.float32)
        self.assertTrue(np.allclose(model.get_layer('pixel_normalization')(white).numpy(), 1))


if __name__ == '__main__':
    unittest.main()
