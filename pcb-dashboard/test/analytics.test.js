import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { defaultScenario, evaluate, projectSuppliers, scenarioIssue } from '../src/analytics.js';

const { samples, threshold } = JSON.parse(await readFile(new URL('../public/validation/predictions.json', import.meta.url), 'utf8'));

test('재계산한 기본 성능이 학습 결과와 일치한다', () => {
  const metrics = evaluate(samples, threshold);
  assert.deepEqual([metrics.tn, metrics.fp, metrics.fn, metrics.tp], [196, 4, 1, 199]);
  assert.equal(metrics.accuracy, 0.9875);
  assert.equal(metrics.recall, 0.995);
  assert.equal(metrics.macroF1.toFixed(4), '0.9875');
});

test('임계값에 따라 양쪽 오류율과 협력사 손익이 변한다', () => {
  const base = evaluate(samples, threshold);
  assert.deepEqual(projectSuppliers(defaultScenario, base).map(row => row.profit), [558100, 576525, 477050, 511250]);
  const changed = evaluate(samples, 0.9);
  assert.deepEqual([changed.fp, changed.fn], [3, 2]);
  assert.equal(projectSuppliers(defaultScenario, changed)[1].profit, 577025);
});

test('임계값과 같은 점수는 정상이고 매출이 0이면 이익률은 계산하지 않는다', () => {
  assert.equal(evaluate([{ target: 1, score: 0.5 }], 0.5).fn, 1);
  const scenario = { ...defaultScenario, suppliers: [{ name: 'A', quantity: 5, defects: 5, price: 10 }] };
  assert.equal(projectSuppliers(scenario, evaluate(samples, threshold))[0].margin, null);
  scenario.suppliers[0].defects = 6;
  assert.match(scenarioIssue(scenario), /납품 수량 이하/);
});
