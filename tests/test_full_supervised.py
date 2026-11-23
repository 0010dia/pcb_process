"""원본 보조 정답의 위치·임계값과 모델의 단일 이미지 입력을 검증한다."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

from src.full_supervised import (WeightedMaskLoss, build_supervised_model, change_target,
                                 resume_history, RestoreBestCallbackState)


class SupervisedOriginalTests(unittest.TestCase):
    def test_single_changed_pixel_is_preserved_and_outside_boxes_ignored(self):
        template = np.zeros((640, 640), dtype=np.uint8)
        inspected = template.copy()
        inspected[10, 10] = 255
        inspected[100, 100] = 255
        inspected[11, 11] = 127
        target = change_target(template, inspected, [[8, 8, 12, 12]])
        self.assertEqual(target.shape, (160, 160))
        self.assertEqual(target.sum(), 1)
        self.assertEqual(target[2, 2], 1)

    def test_loss_and_gradient_are_finite_for_empty_and_sparse_masks(self):
        logits = tf.Variable(tf.zeros((2, 8, 8, 1)))
        targets = np.zeros((2, 8, 8, 1), dtype=np.float32)
        targets[1, 2, 2, 0] = 1
        with tf.GradientTape() as tape:
            loss = WeightedMaskLoss(100)(targets, logits)
        gradient = tape.gradient(loss, logits).numpy()
        self.assertTrue(np.isfinite(float(loss.numpy())))
        self.assertTrue(np.isfinite(gradient).all())

    def test_auxiliary_targets_are_not_model_inputs_and_no_checkpoint_loaded(self):
        with patch('src.full_supervised.keras.models.load_model', side_effect=AssertionError('사전학습 모델 로드 금지')):
            model = build_supervised_model(100)
        self.assertEqual(model.input_shape, (None, 640, 640, 1))
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.get_layer('decision').output.shape[-1], 1)
        self.assertEqual(model.get_layer('mask_logits').output.shape[1:], (160, 160, 1))

    def test_resume_rewinds_logs_to_best_saved_epoch(self):
        history = pd.DataFrame({'epoch': [0, 1, 2], 'val_macro_f1': [.8, .95, .94]})
        initial_epoch, best, retained = resume_history(history)
        self.assertEqual(initial_epoch, 2)
        self.assertEqual(best, .95)
        self.assertEqual(retained.epoch.tolist(), [0, 1])
        self.assertEqual(len(history), 3)
        with self.assertRaises(ValueError):
            resume_history(history.iloc[[0, 2]])

    def test_resume_callbacks_keep_previous_best_when_new_metric_is_lower(self):
        model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
        model.compile(optimizer=keras.optimizers.Adam(.00015), loss='mse')
        checkpoint = keras.callbacks.ModelCheckpoint('unused.keras', monitor='val_macro_f1', mode='max', save_best_only=True)
        early = keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=5, restore_best_weights=True)
        reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1', mode='max', patience=2, factor=.5)
        restore = RestoreBestCallbackState(checkpoint, early, reduce, .985, 11)
        for callback in [checkpoint, early, reduce, restore]:
            callback.set_model(model)
            callback.on_train_begin()
        early.on_epoch_end(11, {'val_macro_f1': .98})
        self.assertEqual(checkpoint.best, .985)
        self.assertEqual(early.best, .985)
        self.assertEqual(early.best_epoch, 10)
        self.assertEqual(early.wait, 1)
        reduce.on_epoch_end(11, {'val_macro_f1': .98})
        reduce.on_epoch_end(12, {'val_macro_f1': .97})
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .000075, places=8)


if __name__ == '__main__':
    unittest.main()
