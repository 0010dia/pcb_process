export const defaultScenario = {
  sellingPrice: 1600,
  reviewFee: 200,
  missedFee: 20000,
  suppliers: [
    { name: 'A', quantity: 1000, defects: 20, price: 1000 },
    { name: 'B', quantity: 1000, defects: 5, price: 1010 },
    { name: 'C', quantity: 1000, defects: 10, price: 1100 },
    { name: 'D', quantity: 1000, defects: 50, price: 990 },
  ],
};

export const classifications = {
  tn: { title: '정상 판정', description: '실제 정상 · 예측 정상' },
  fp: { title: '정상 오탐', description: '실제 정상 · 예측 불량' },
  fn: { title: '불량 누락', description: '실제 불량 · 예측 정상' },
  tp: { title: '불량 탐지', description: '실제 불량 · 예측 불량' },
};

const divide = (numerator, denominator) => denominator ? numerator / denominator : 0;

export function evaluate(samples, threshold) {
  const groups = { tn: [], fp: [], fn: [], tp: [] };
  for (const sample of samples) {
    const predictedDefect = sample.score > threshold;
    const group = sample.target === 1 ? (predictedDefect ? 'tp' : 'fn') : (predictedDefect ? 'fp' : 'tn');
    groups[group].push(sample);
  }
  const { tn, fp, fn, tp } = Object.fromEntries(Object.entries(groups).map(([key, rows]) => [key, rows.length]));
  return {
    groups, tn, fp, fn, tp,
    accuracy: divide(tn + tp, samples.length),
    macroF1: (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)) / 2,
    recall: divide(tp, tp + fn),
    falsePositiveRate: divide(fp, tn + fp),
    missedRate: divide(fn, tp + fn),
  };
}

export function projectSuppliers(scenario, metrics) {
  return scenario.suppliers.map(supplier => {
    const normal = supplier.quantity - supplier.defects;
    const missed = supplier.defects * metrics.missedRate;
    const reviewed = normal * metrics.falsePositiveRate + supplier.defects - missed;
    const revenue = normal * scenario.sellingPrice;
    const purchaseCost = supplier.price * supplier.quantity;
    const reviewCost = reviewed * scenario.reviewFee;
    const missedCost = missed * scenario.missedFee;
    const profit = revenue - purchaseCost - reviewCost - missedCost;
    return {
      ...supplier, normal, missed, reviewed, revenue, purchaseCost, reviewCost, missedCost, profit,
      margin: revenue ? profit / revenue : null,
    };
  });
}

export function scenarioIssue(scenario) {
  for (const supplier of scenario.suppliers) {
    if (supplier.defects > supplier.quantity) {
      return `${supplier.name}사의 불량 수량은 납품 수량 이하여야 합니다.`;
    }
  }
  return '';
}
