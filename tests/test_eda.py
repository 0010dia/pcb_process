"""작은 합성 데이터로 주석 파싱과 분할 감사의 실패 조건을 검증한다."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.eda import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def test_processed_outputs_do_not_pollute_raw_inventory(self):
        self.add_pair('trainval', '001', '1 2 10 11 1\n', value=0)
        generated = self.root / 'processed/roi128_v1/train/0'
        generated.mkdir(parents=True)
        Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(generated / 'patch.png')
        audit = audit_dataset(self.root)
        self.assertEqual(audit.summary['physical_image_files'], 2)
        self.assertFalse(audit.issues.kind.eq('unlisted_image').any())

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'trainval.txt').write_text('', encoding='utf-8')
        (self.root / 'test.txt').write_text('', encoding='utf-8')

    def add_pair(self, split, name, annotation, value=0):
        # 원본 목록의 가상 .jpg 이름과 실제 _test/_temp 파일명을 똑같이 재현한다.
        folder = self.root / 'group01' / '01'
        labels = self.root / 'group01' / '01_not'
        folder.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        pixels = np.full((16, 16), value, dtype=np.uint8)
        for role in ('test', 'temp'):
            Image.fromarray(pixels).save(folder / f'{name}_{role}.jpg')
        (labels / f'{name}.txt').write_text(annotation, encoding='utf-8')
        with (self.root / f'{split}.txt').open('a', encoding='utf-8') as stream:
            stream.write(f'group01/01/{name}.jpg group01/01_not/{name}.txt\n')

    def test_paths_class_mapping_and_pixel_quarantine(self):
        self.add_pair('trainval', '001', '1 2 10 11 5\n2 3 8 9 6\n', value=0)
        self.add_pair('test', '002', '1 2 10 11 1\n', value=255)
        audit = audit_dataset(self.root)
        self.assertEqual(audit.summary['errors'], 0)
        self.assertEqual(audit.summary['physical_image_files'], 4)
        self.assertEqual(audit.summary['split_objects'], {'trainval': 2, 'test': 1})
        self.assertEqual(audit.objects.target_id.tolist(), [4, 5, 0])
        self.assertEqual(audit.objects.class_name.iloc[0], 'Spurious Copper')
        self.assertEqual(audit.objects.class_name.iloc[1], 'Pin Hole')
        self.assertEqual(audit.pixel_histograms['test'][0], 16*16)
        self.assertEqual(audit.pixel_histograms['test'][255], 0)

    def test_invalid_coordinates_unknown_class_and_bad_row(self):
        self.add_pair('trainval', '001', '-1 0 10 10 1\n1 2 9 10 7\nnot an annotation\n')
        audit = audit_dataset(self.root)
        self.assertEqual(set(audit.issues.kind), {'bbox_bounds', 'class_id', 'label_format'})
        self.assertEqual(audit.summary['valid_objects'], 0)
        self.assertEqual(audit.summary['annotation_lines'], 3)

    def test_missing_image_and_duplicate_source(self):
        self.add_pair('trainval', '001', '1 2 9 10 1\n')
        (self.root / 'group01/01/001_temp.jpg').unlink()
        (self.root / 'test.txt').write_text('group01/01/001.jpg group01/01_not/001.txt\n')
        audit = audit_dataset(self.root)
        self.assertIn('image_read', set(audit.issues.kind))
        self.assertIn('duplicate_source', set(audit.issues.kind))

    def test_identical_pixels_across_splits_are_flagged(self):
        self.add_pair('trainval', '001', '1 2 9 10 1\n', value=0)
        self.add_pair('test', '002', '1 2 9 10 1\n', value=0)
        audit = audit_dataset(self.root)
        self.assertEqual(audit.summary['pixel_duplicates']['test']['cross_split_hash_groups'], 1)
        self.assertIn('cross_split_test_pixels', set(audit.issues.kind))


if __name__ == '__main__':
    unittest.main()
