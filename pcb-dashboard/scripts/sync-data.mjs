import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { basename, extname, relative, resolve } from 'node:path';
import { parse } from 'csv-parse/sync';

const dashboard = fileURLToPath(new URL('../', import.meta.url));
const root = resolve(dashboard, '..');
const source = resolve(root, 'results/full_scratch/full_scratch_aux_001');
const destination = resolve(dashboard, 'public/validation');
const report = JSON.parse(await readFile(resolve(source, 'run_result.json'), 'utf8'));
const rows = parse(await readFile(resolve(source, 'validation_predictions.csv'), 'utf8'), {
  columns: true, bom: true, skip_empty_lines: true,
});
if (report.status !== 'completed' || rows.length !== report.metrics.samples || !rows.length) {
  throw new Error('완료된 실행의 검증 이미지 수와 예측 파일이 일치해야 합니다.');
}

const samples = rows.map((row, index) => {
  const target = Number(row.target);
  const score = Number(row.sigmoid_score);
  if (row.split !== 'val' || !['0', '1'].includes(row.target) ||
      !row.sigmoid_score?.trim() || !Number.isFinite(score) || score < 0 || score > 1 ||
      Number(row.predicted_target) !== Number(score > report.metrics.threshold) ||
      !/^data\/[\w/-]+\.(?:jpg|jpeg|png)$/i.test(row.image_path)) {
    throw new Error(`유효하지 않은 검증 예측 행 ${index + 2}`);
  }
  return {
    id: basename(row.image_path, extname(row.image_path)), target, score,
    image: `${String(index + 1).padStart(3, '0')}${extname(row.image_path)}`,
  };
});

await mkdir(destination, { recursive: true });
for (const [index, sample] of samples.entries()) {
  try {
    await copyFile(resolve(root, rows[index].image_path), resolve(destination, sample.image));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    sample.image = null;
  }
}
await writeFile(resolve(destination, 'predictions.json'), JSON.stringify({
  run: report.run_name,
  bestEpoch: report.best_epoch,
  threshold: report.metrics.threshold,
  group: rows[0].group,
  samples,
}));
console.log(`${samples.length}장 검증 데이터를 ${relative(dashboard, destination)}에 준비했습니다.`);
