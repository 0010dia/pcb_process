import { useEffect, useRef, useState } from 'react';
import {
  Activity, BarChart3, ChevronLeft, ChevronRight, CircleHelp, Cpu, Download,
  Factory, ScanLine, Settings2, ShieldCheck, Target, TrendingUp, X,
} from 'lucide-react';
import { classifications, defaultScenario, evaluate, projectSuppliers, scenarioIssue } from './analytics.js';

const dataPath = `${import.meta.env.BASE_URL}validation/`;
const integer = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 0 });
const expectedCount = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 4 });
const won = value => `${integer.format(value)}원`;
const percent = value => value === null ? '—' : `${(value * 100).toFixed(2)}%`;
const supplierTones = { A: 'blue', B: 'green', C: 'purple', D: 'amber' };
const pageSize = 4;

function Modal({ title, onClose, children }) {
  const ref = useRef(null);
  useEffect(() => { ref.current.showModal(); }, []);
  return (
    <dialog ref={ref} aria-label={title} className="modal" onClose={onClose}>
      <div className="modal-heading"><h2>{title}</h2><button className="icon-button" type="button" aria-label="닫기" onClick={() => ref.current.close()}><X size={20} /></button></div>
      {children}
    </dialog>
  );
}

function ScenarioEditor({ scenario, onClose, onApply }) {
  const [draft, setDraft] = useState(scenario);
  const [error, setError] = useState('');
  const numberInput = (label, value, update, min = 0, max = 100000000) => (
    <label className="field">
      <span>{label}</span>
      <input type="number" required step="1" min={min} max={max} value={value}
        onChange={event => update(event.target.value === '' ? '' : event.target.valueAsNumber)} />
    </label>
  );
  const changeSupplier = (name, key, value) => {
    setDraft(previous => ({ ...previous, suppliers: previous.suppliers.map(supplier =>
      supplier.name === name ? { ...supplier, [key]: value } : supplier) }));
  };

  function submit(event) {
    event.preventDefault();
    const issue = scenarioIssue(draft);
    setError(issue);
    if (!issue) onApply(draft);
  }

  return (
    <Modal title="시나리오 편집" onClose={onClose}>
      <form onSubmit={submit}>
        <h3>공통 가격과 비용</h3>
        <div className="form-grid">
          {numberInput('정상 판매가 (원)', draft.sellingPrice, value => setDraft({ ...draft, sellingPrice: value }))}
          {numberInput('건당 재검사비 (원)', draft.reviewFee, value => setDraft({ ...draft, reviewFee: value }))}
          {numberInput('건당 불량 누락 처리비 (원)', draft.missedFee, value => setDraft({ ...draft, missedFee: value }))}
        </div>
        <h3>업체별 납품 가정</h3>
        {draft.suppliers.map(supplier => (
          <fieldset key={supplier.name}>
            <legend>협력사 {supplier.name}</legend>
            <div className="form-grid">
              {numberInput('납품 수량', supplier.quantity, value => changeSupplier(supplier.name, 'quantity', value), 1, 1000000)}
              {numberInput('불량 수량', supplier.defects, value => changeSupplier(supplier.name, 'defects', value), 0, 1000000)}
              {numberInput('납품 단가 (원)', supplier.price, value => changeSupplier(supplier.name, 'price', value))}
            </div>
          </fieldset>
        ))}
        {error && <p className="form-error" role="alert">{error}</p>}
        <div className="modal-actions">
          <button type="button" className="button secondary" onClick={() => { setDraft(defaultScenario); setError(''); }}>기본값 복원</button>
          <button type="submit" className="button primary">적용하기</button>
        </div>
      </form>
    </Modal>
  );
}

function SupplierSection({ ranked, selected, setSelected, scenario, threshold, metrics, openSettings }) {
  const winner = ranked[0];
  const company = ranked.find(supplier => supplier.name === selected);
  const scale = Math.max(1, ...ranked.map(supplier => Math.abs(supplier.profit)));
  const cheapest = ranked.reduce((best, supplier) => supplier.price < best.price ? supplier : best);
  const costs = [
    { label: '정상 판매 매출', value: company.revenue },
    { label: '전체 매입비', value: company.purchaseCost },
    { label: '재검사비', value: company.reviewCost, note: `예상 ${expectedCount.format(company.reviewed)}건 × ${won(scenario.reviewFee)}` },
    { label: '불량 누락 처리비', value: company.missedCost, note: `예상 ${expectedCount.format(company.missed)}건 × ${won(scenario.missedFee)}` },
  ];

  return (
    <section className="section" id="suppliers" aria-labelledby="suppliers-title">
      <div className="section-heading">
        <div><div className="eyebrow"><Factory size={16} /> 납품 시뮬레이션</div><h2 id="suppliers-title">협력사 손익 비교</h2></div>
        <button className="button secondary" onClick={openSettings}><Settings2 size={17} /> 가정 편집</button>
      </div>
      <div className="assumptions">
        <span>정상 판매가 <strong>{won(scenario.sellingPrice)}</strong></span>
        <span>재검사비 <strong>{won(scenario.reviewFee)} / 건</strong></span>
        <span>누락 처리비 <strong>{won(scenario.missedFee)} / 건</strong></span>
        <span className="threshold-tag">모델 임계값 {threshold.toFixed(2)}</span>
      </div>
      <div className="analysis-grid">
        <article className="card chart-card">
          <div className="card-header"><h3>업체별 예상 이익</h3><span className="small-label">단위 원</span></div>
          <div className="profit-chart">
            {ranked.map(supplier => (
              <button key={supplier.name} className={`bar-row ${selected === supplier.name ? 'selected' : ''}`} aria-pressed={selected === supplier.name}
                onClick={() => setSelected(supplier.name)}>
                <span className={`supplier-letter ${supplierTones[supplier.name]}`}>{supplier.name}</span>
                <span className="bar-track"><span className={`bar ${supplier.profit < 0 ? 'loss' : ''}`} style={{ width: `${Math.abs(supplier.profit) / scale * 100}%` }} /></span>
                <strong>{integer.format(supplier.profit)}</strong>
              </button>
            ))}
          </div>
          <div className="chart-note"><TrendingUp size={17} /><span>{winner.name === cheapest.name
            ? <><strong>{winner.name}사</strong>의 납품 단가가 가장 낮고 예상 이익이 가장 높습니다.</>
            : winner.profit === cheapest.profit
              ? <>단가가 가장 낮은 {cheapest.name}사와 {winner.name}사의 예상 이익이 같습니다.</>
              : <>단가가 가장 낮은 {cheapest.name}사보다 <strong>{winner.name}사의 예상 이익이 {won(winner.profit - cheapest.profit)} 더 높습니다.</strong></>}</span></div>
        </article>
        <article className="card breakdown-card" aria-live="polite">
          <div className="card-header"><h3>협력사 {company.name} 손익 내역</h3><span className={`supplier-letter ${supplierTones[company.name]}`}>{company.name}</span></div>
          <dl className="cost-list">
            {costs.map(item => <div key={item.label}><dt>{item.label}{item.note && <small>{item.note}</small>}</dt><dd>{won(item.value)}</dd></div>)}
            <div className="profit-total"><dt>예상 이익</dt><dd><strong>{won(company.profit)}</strong><small>이익률 {percent(company.margin)}</small></dd></div>
          </dl>
        </article>
      </div>
      <article className="card suppliers-table">
        <div className="card-header"><h3>업체별 납품 현황</h3><span className="small-label">가상 협력사 A–D</span></div>
        <div className="table-scroll" tabIndex="0" aria-label="가상 협력사 납품 현황">
          <table>
            <thead><tr><th scope="col">협력사</th><th scope="col">납품 수량</th><th scope="col">단가</th><th scope="col">정상</th><th scope="col">불량</th><th scope="col">불량률</th><th scope="col">예상 이익</th><th scope="col">이익률</th></tr></thead>
            <tbody>{ranked.map(supplier => (
              <tr className={selected === supplier.name ? 'selected' : ''} key={supplier.name}>
                <th scope="row"><button aria-pressed={selected === supplier.name} onClick={() => setSelected(supplier.name)}><span className={`supplier-letter ${supplierTones[supplier.name]}`}>{supplier.name}</span> 협력사 {supplier.name}</button></th>
                <td>{integer.format(supplier.quantity)}</td><td>{won(supplier.price)}</td><td>{integer.format(supplier.normal)}</td><td>{integer.format(supplier.defects)}</td>
                <td>{percent(supplier.defects / supplier.quantity)}</td><td className="profit-value">{won(supplier.profit)}</td><td>{percent(supplier.margin)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <details className="method"><summary><CircleHelp size={16} /> 계산 가정</summary>
          <p>현재 검증 오탐률 {percent(metrics.falsePositiveRate)}와 누락률 {percent(metrics.missedRate)}를 가상 협력사에 동일하게 적용합니다. 불량 판정 건은 모두 재검사하며 정상으로 확인된 제품은 판매에 넣습니다. 전체 매입비를 비용으로 반영합니다.</p>
          <p>예상 이익은 정상 판매 매출에서 매입비, 재검사비 및 누락 처리비를 뺀 값입니다. 이익률은 정상 판매 매출 대비 예상 이익입니다. 건수는 기대값이라 소수일 수 있습니다.</p>
        </details>
      </article>
    </section>
  );
}

function ValidationSection({ metrics, threshold, setThreshold, report, preview }) {
  const [filter, setFilter] = useState('errors');
  const [page, setPage] = useState(0);
  const cases = filter === 'errors'
    ? [...metrics.groups.fn, ...metrics.groups.fp]
    : metrics.groups[filter];
  const lastPage = Math.max(0, Math.ceil(cases.length / pageSize) - 1);
  const viewPage = Math.min(page, lastPage);

  function selectFilter(key) {
    setFilter(key);
    setPage(0);
  }

  function changeThreshold(value) {
    setThreshold(value);
    setPage(0);
  }

  return (
    <section className="section" id="validation" aria-labelledby="validation-title">
      <div className="section-heading">
        <div><div className="eyebrow"><Activity size={16} /> 모델 검증</div><h2 id="validation-title">검사 성능과 판정 사례</h2></div>
        <span className="pill">검증 {report.samples.length}장 · Epoch {report.bestEpoch}</span>
      </div>
      <div className="analysis-grid model-grid">
        <article className="card confusion-card">
          <div className="card-header"><h3>혼동행렬</h3><span className="small-label">실제 × 예측</span></div>
          <table className="matrix" aria-label="검증 혼동행렬">
            <thead><tr><td></td><th scope="col">예측 정상</th><th scope="col">예측 불량</th></tr></thead>
            <tbody>{[['실제 정상', 'tn', 'fp'], ['실제 불량', 'fn', 'tp']].map(([label, ...keys]) => (
              <tr key={label}><th scope="row">{label}</th>{keys.map(key => (
                <td key={key}><button onClick={() => selectFilter(key)} className={`matrix-cell ${key} ${filter === key ? 'active' : ''}`}
                  aria-pressed={filter === key} aria-label={`${classifications[key].description}, ${metrics[key]}장. 사례 보기`}>
                  <strong>{metrics[key]}</strong><span>{classifications[key].title}</span>
                </button></td>
              ))}</tr>
            ))}</tbody>
          </table>
          <div className="slider-heading"><label htmlFor="threshold">불량 판정 임계값</label><output htmlFor="threshold">{threshold.toFixed(2)}</output></div>
          <input id="threshold" aria-valuetext={threshold.toFixed(2)} type="range" min="0" max="1" step="0.01" value={threshold}
            style={{ '--range-progress': `${threshold * 100}%` }} onChange={event => changeThreshold(event.target.valueAsNumber)} />
          <div className="slider-labels"><span>0.00</span><button onClick={() => changeThreshold(report.threshold)}>기본값 {report.threshold.toFixed(2)}</button><span>1.00</span></div>
          <p className="small-label">모델 점수가 임계값을 넘으면 불량으로 판정합니다.</p>
        </article>
        <article className="card examples-card">
          <div className="card-header"><h3>{filter === 'errors' ? '오판 사례' : classifications[filter].title}</h3><span className="small-label">{cases.length}장</span></div>
          <div className="filter-row" aria-label="사례 분류">
            {[['errors', '전체 오판'], ['fp', '정상 오탐'], ['fn', '불량 누락']].map(([key, name]) => (
              <button key={key} className={`chip ${filter === key ? 'active' : ''}`} aria-pressed={filter === key} onClick={() => selectFilter(key)}>{name}</button>
            ))}
          </div>
          <div className="examples">
            {cases.slice(viewPage * pageSize, (viewPage + 1) * pageSize).map(sample => (
              <button key={sample.id} className="sample" aria-label={`${sample.id} 이미지 상세 보기`} onClick={() => preview(sample)}>
                {sample.image && <img src={`${dataPath}${sample.image}`} alt={`${sample.id} PCB`} loading="lazy" />}
                <span><strong>{sample.id}</strong><small>점수 {sample.score.toFixed(4)}</small></span>
              </button>
            ))}
            {!cases.length && <p className="empty-state">이 임계값에 해당하는 사례가 없습니다.</p>}
          </div>
          {lastPage > 0 && <div className="pagination">
            <button className="icon-button" aria-label="이전 사례" disabled={viewPage === 0} onClick={() => setPage(viewPage - 1)}><ChevronLeft size={18} /></button>
            <span>{viewPage + 1} / {lastPage + 1}</span>
            <button className="icon-button" aria-label="다음 사례" disabled={viewPage === lastPage} onClick={() => setPage(viewPage + 1)}><ChevronRight size={18} /></button>
          </div>}
        </article>
      </div>
    </section>
  );
}

function downloadResults(ranked, scenario, metrics, threshold) {
  const headings = ['협력사', '납품 수량', '단가(원)', '정상', '불량', '불량률', '임계값', '모델 오탐률', '모델 누락률', '정상 판매가(원)', '건당 재검사비(원)', '건당 누락 처리비(원)', '예상 재검사 건수', '예상 누락 건수', '매출(원)', '매입비(원)', '재검사비(원)', '누락 비용(원)', '예상 이익(원)', '이익률'];
  const rows = ranked.map(supplier => [supplier.name, supplier.quantity, supplier.price, supplier.normal, supplier.defects,
    percent(supplier.defects / supplier.quantity), threshold.toFixed(2), percent(metrics.falsePositiveRate), percent(metrics.missedRate),
    scenario.sellingPrice, scenario.reviewFee, scenario.missedFee, supplier.reviewed.toFixed(4), supplier.missed.toFixed(4),
    supplier.revenue, supplier.purchaseCost, supplier.reviewCost.toFixed(2), supplier.missedCost.toFixed(2), supplier.profit.toFixed(2),
    supplier.margin === null ? '' : percent(supplier.margin)]);
  const csv = [headings, ...rows].map(row => row.join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'pcb-supplier-simulation.csv';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Dashboard({ report }) {
  const [scenario, setScenario] = useState(defaultScenario);
  const [threshold, setThreshold] = useState(report.threshold);
  const [selected, setSelected] = useState(null);
  const [editing, setEditing] = useState(false);
  const [sample, setSample] = useState(null);
  const metrics = evaluate(report.samples, threshold);
  const ranked = projectSuppliers(scenario, metrics).sort((a, b) => b.profit - a.profit || a.name.localeCompare(b.name));
  const best = ranked[0];
  const heroSample = report.samples.find(item => item.target === 1 && item.image);
  const kpis = [
    { icon: Target, tone: 'blue', name: '검증 정확도', value: percent(metrics.accuracy), note: `${metrics.tn + metrics.tp} / ${report.samples.length}장 정분류` },
    { icon: Activity, tone: 'purple', name: 'Macro F1', value: metrics.macroF1.toFixed(4), note: '정상·불량 클래스 평균' },
    { icon: ShieldCheck, tone: 'green', name: '불량 재현율', value: percent(metrics.recall), note: `불량 ${metrics.tp + metrics.fn}장 중 ${metrics.tp}장 탐지` },
    { icon: TrendingUp, tone: 'amber', name: '최고 예상 이익', value: won(best.profit), note: `협력사 ${best.name} · 이익률 ${percent(best.margin)}` },
  ];

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#overview"><span className="brand-icon"><Cpu size={22} /></span><span>PCB <strong>Insight</strong></span></a>
          <nav aria-label="주 메뉴"><a href="#overview">종합 분석</a><a href="#suppliers">협력사 손익</a><a href="#validation">검사 성능</a></nav>
          <button className="button secondary export" onClick={() => downloadResults(ranked, scenario, metrics, threshold)}><Download size={17} /> CSV 내보내기</button>
        </div>
      </header>
      <main className="shell" id="overview">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <span className="hero-tag"><ScanLine size={16} /> 자화전자 가상 구매 시나리오</span>
            <h1 id="page-title">PCB 납품 품질 분석</h1>
            <p>검사 모델의 판정 결과로 협력사별 품질 비용과 예상 이익을 비교합니다.</p>
          </div>
          {heroSample && <div className="hero-visual">
            <img src={`${dataPath}${heroSample.image}`} alt="DeepPCB 검증용 PCB 원본 샘플" />
            <span className="hero-caption"><ScanLine size={15} /> DeepPCB 검증 샘플</span>
          </div>}
        </section>
        <div className="kpis" aria-label="핵심 지표">
          {kpis.map(({ icon: Icon, tone, name, value, note }) => (
            <article className="card kpi" key={name}>
              <span className={`kpi-icon ${tone}`}><Icon size={19} /></span>
              <div><h2>{name}</h2><strong>{value}</strong><p>{note}</p></div>
            </article>
          ))}
        </div>
        <SupplierSection ranked={ranked} selected={selected ?? best.name} setSelected={setSelected} scenario={scenario}
          threshold={threshold} metrics={metrics} openSettings={() => setEditing(true)} />
        <ValidationSection metrics={metrics} threshold={threshold} setThreshold={setThreshold} report={report} preview={setSample} />
        <footer><BarChart3 size={16} /> DeepPCB {report.group} 검증 결과 · 협력사와 거래 조건은 시뮬레이션용입니다.</footer>
      </main>
      {editing && <ScenarioEditor scenario={scenario} onClose={() => setEditing(false)} onApply={next => { setScenario(next); setEditing(false); }} />}
      {sample && <Modal title={sample.id} onClose={() => setSample(null)}>
        {sample.image && <img className="case-image" src={`${dataPath}${sample.image}`} alt={`${sample.id} 640×640 PCB 검증 원본`} />}
        <div className="case-meta">
          <span>실제 <strong>{sample.target ? '불량' : '정상'}</strong></span>
          <span>예측 <strong>{sample.score > threshold ? '불량' : '정상'}</strong></span>
          <span>모델 점수 <strong>{sample.score.toFixed(6)}</strong></span>
        </div>
      </Modal>}
    </>
  );
}

export default function App() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${dataPath}predictions.json`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('데이터 응답 오류'); return response.json(); })
      .then(setReport)
      .catch(reason => { if (reason.name !== 'AbortError') setError(true); });
    return () => controller.abort();
  }, []);

  if (!report) return <main className="shell load-state" role={error ? 'alert' : 'status'}>
    {error ? '검증 데이터를 불러오지 못했습니다. 새로 고침 후 다시 확인해 주세요.' : '검증 데이터를 불러오는 중입니다…'}
  </main>;
  return <Dashboard report={report} />;
}
