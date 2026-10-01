// Shared renderer for the static app and Python-produced offline reports.
// No CDN, font, telemetry, or network dependency. All imported text is escaped.
(function () {
  const css = `:root{color-scheme:dark;font-family:system-ui,sans-serif;color:#e7edf8}body{margin:0;background:#0d1320}#report-root{max-width:1180px;margin:auto;padding:24px}.br h1{font-size:34px;letter-spacing:-1px;margin:12px 0}.br h2{font-size:22px}.br p,.br small{color:#acbdd3}.br .tag{display:inline-block;padding:6px 11px;background:#21394a;color:#6ee7c4;border-radius:18px;font-size:11px;font-weight:700;letter-spacing:1px;margin-right:8px}.br .warning{padding:14px 18px;background:#3c3030;border-left:3px solid #f7bc7b;margin:12px 0;color:#fbd9b0}.br .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:24px 0}.br .card{padding:20px;background:#172337;border:1px solid #30415a;border-radius:12px}.br .card b{display:block;font-size:27px;color:#6ee7c4}.br .card span{font-size:12px;color:#9bacc5}.br button{background:#1c2a3e;color:#c5d4e9;border:1px solid #38506b;padding:10px 15px;border-radius:7px;cursor:pointer;font:inherit;font-size:13px}.br button.active{border-color:#6ee7c4;color:#6ee7c4}.br .tabs{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}.br .charts{display:grid;grid-template-columns:1fr 1fr;gap:20px}.br .chart{background:#151e2d;border:1px solid #30415a;border-radius:12px;padding:15px;min-width:0}.br svg{width:100%;height:auto}.br svg text{fill:#aebed5;font-size:11px}.br svg circle{cursor:crosshair}.br .legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}.br .legend button{font-size:11px;padding:5px 9px}.br table{border-collapse:collapse;width:100%;font-size:12px}.br th,.br td{text-align:right;padding:12px 10px;border-bottom:1px solid #30415a;white-space:nowrap}.br th:first-child,.br td:first-child{text-align:left}.br th{background:#1c2b41;color:#cbd8e9}.br .table-scroll{overflow:auto}.br .bad{color:#ffb5b5}.br details{border:1px solid #30415a;border-radius:8px;margin:14px 0;padding:14px}.br summary{cursor:pointer;font-weight:600}.br pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#b6cde7;font-size:12px;max-height:450px;overflow:auto}.br [hidden]{display:none!important}.br .bars{display:grid;grid-template-columns:130px 1fr 55px;gap:12px;align-items:center;margin:8px 0;font-size:12px}.br .bar{background:#8badff;border-radius:5px;height:10px}.br .note{font-size:12px;color:#9bacc5;padding:15px;background:#182436;border-radius:8px}@media(max-width:760px){.br .charts{grid-template-columns:1fr}#report-root{padding:12px}.br h1{font-size:27px}}`;
  const escape = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : null;
  const get = (o, path) => path.split('.').reduce((a,k) => a && a[k], o);
  function percentile(s, n) {
    const p=s?.percentiles;
    if (Array.isArray(p)) {
      const item=p.find(v=>Number(Array.isArray(v)?v[0]:v.percentile)===n);
      return finite(Array.isArray(item)?item[1]:item?.value);
    }
    return finite(p?.[String(n)] ?? p?.[`${n}.0`] ?? p?.[`p${n}`]);
  }
  function normalize(raw, context) {
    if (raw?.metadata?.version !== 2 || raw?.metadata?.guidellm_version !== '0.8.0') throw new Error('Expected GuideLLM 0.8.0 report schema 2. Historical reports are not silently reinterpreted.');
    if (!Array.isArray(raw.benchmarks) || !raw.benchmarks.length) throw new Error('No benchmarks in this file.');
    return raw.benchmarks.map((b,i) => {
      if (!b.metrics || !b.metrics.request_totals) throw new Error('Missing GuideLLM benchmark metrics.');
      const m=b.metrics, s=m.time_to_first_token_ms?.successful;
      const count=m.request_totals;
      const mean=path=>finite(get(m,path+'.successful.mean'));
      const rate=finite(context.rate ?? b.config?.strategy?.rate);
      const warnings=[];
      const successful=finite(count.successful), errored=finite(count.errored), incomplete=finite(count.incomplete);
      if (successful === null || errored === null || incomplete === null) warnings.push('Missing request counts.');
      if ((errored||0)+(incomplete||0)>0) warnings.push('Failed or incomplete requests: inspect reliability before claiming capacity.');
      if ((successful||0)<368) warnings.push('Too few successful requests to bound P99 with a two-sided 95% interval.');
      const rps=mean('requests_per_second');
      if(rate!==null && rps!==null && rps<rate*.95)warnings.push('Achieved successful request rate is below 95% of offered load.');
      if(!context.dataset_checksum)warnings.push('Dataset checksum unavailable: matched-workload provenance is unverified.');
      return {...context, id:`${context.id||'import'}-${i}`,rate,rps,successful,errored,incomplete,ttft_mean_ms:finite(s?.mean),ttft_p50_ms:finite(s?.median)??percentile(s,50),ttft_p95_ms:percentile(s,95),ttft_p99_ms:percentile(s,99),ttft_p99_ci:s?.percentile_cis?.['99.0']??s?.percentile_cis?.['99']??null,itl_ms:mean('inter_token_latency_ms'),output_tps:mean('output_tokens_per_second'),goodput:mean('request_goodput'),attainment:finite(m.slo_attainment),dispatch_ms:mean('request_dispatch_delay')===null?null:mean('request_dispatch_delay')*1000,warnings};
    });
  }
  function validate(data) {
    if(data?.report_schema_version!==1 || !data.experiment?.plan || !Array.isArray(data.runs) || data.runs.length>10000)throw new Error('Invalid compact report schema.');
    const ids=new Set();
    for(const r of data.runs){
      if(!['service','llmd'].includes(r.routing) || !Number.isInteger(r.topology?.tp) || r.topology.tp<1 || !Number.isInteger(r.topology?.replicas) || r.topology.replicas<1)throw new Error('Every run needs explicit routing and replica/TP metadata.');
      if(typeof r.id!=='string' || ids.has(r.id))throw new Error('Missing or duplicate run ID.');ids.add(r.id);
      for(const key of ['rate','rps','successful','errored','incomplete','ttft_p99_ms','ttft_p95_ms','ttft_p50_ms','itl_ms','output_tps','goodput','attainment','dispatch_ms'])if(r[key]!=null && finite(r[key])===null)throw new Error(`Invalid ${key} metric.`);
      if(r.attainment!=null && r.attainment>1)throw new Error('SLO attainment must be a fraction.');
    }
    return data;
  }
  // Self-contained function: serialized verbatim into HTML so CLI and browser render identically.
  function render(root,data) {
    const esc = v => String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const num=(n,d=1)=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d}):'N/A';
    const e=data.experiment, rows=data.runs, slo=e.benchmark.slo_ms;
    const label=r=>`${r.routing==='llmd'?'Default llm-d':'Kubernetes Service'} · ${r.topology.replicas}R × TP${r.topology.tp}`;
    const eligible=rows.filter(r=>(!r.status||r.status==='completed')&&r.ttft_p99_ms!=null&&r.ttft_p99_ms<=slo&&r.rps!=null&&r.rate!=null&&r.rps>=r.rate*.95&&r.errored===0&&r.incomplete===0&&r.successful>=368);
    const best=eligible.reduce((a,r)=>!a||r.rps>a.rps?r:a,null);
    const warningCount=rows.filter(r=>(r.warnings||[]).length).length;
    root.classList.add('br');
    root.innerHTML=`<span class="tag">${data.demo?'ILLUSTRATIVE DEMO · NOT MEASURED':'BENCHMARK REPORT'}</span><span class="tag">LOCAL & OFFLINE</span><h1>${esc(e.name)} · routing & topology</h1><p>${esc(e.plan.model.id)} · ${esc(e.hardware.gpu)} · ${esc(e.hardware.memory_gib)} GiB per GPU · budget ${esc(e.hardware.budget)} GPUs</p>${data.demo?'<div class="warning">These are fabricated sample measurements for demonstrating the UI. Do not use them for capacity planning.</div>':''}<div class="cards"><div class="card"><b>${num(rows.length,0)}</b><span>Recorded trial results</span></div><div class="card"><b>${best?num(best.rps):'Not established'}</b><span>Best observed RPS passing completeness/load/P99 gates</span></div><div class="card"><b>${num(slo,0)} ms</b><span>TTFT P99 objective</span></div><div class="card"><b>${num(warningCount,0)}</b><span>Results with caveats</span></div></div><div class="tabs" role="tablist">${['Overview','Workload','Trials','Reproducibility'].map((t,i)=>`<button role="tab" aria-selected="${i===0}" data-tab="${i}" class="${i===0?'active':''}">${t}</button>`).join('')}</div><section data-content="0"><p class="note">Service versus llm-d compares complete serving paths, including routing overhead. Missing values are not zero. A single replica cannot establish multi-replica routing gains. Rankings are observations, not projected production capacity. GuideLLM SLO attainment excludes duration-cancelled requests; the completeness gate remains separate.</p><div class="charts"><div class="chart"><h3>TTFT P99 vs offered RPS</h3><div data-chart="ttft_p99_ms"></div></div><div class="chart"><h3>Successful throughput vs offered RPS</h3><div data-chart="rps"></div></div><div class="chart"><h3>Output tokens / second</h3><div data-chart="output_tps"></div></div><div class="chart"><h3>Objective-conforming goodput</h3><div data-chart="goodput"></div></div></div><h2>Matched comparisons</h2><div id="br-comparisons"></div></section><section data-content="1" hidden><h2>Dataset design & observed distributions</h2><p>Configured reuse is potential prefix overlap, not a guaranteed cache hit rate. Counts below come from the generator when available.</p><details open><summary>Workload specification</summary><pre>${esc(JSON.stringify(e.workload,null,2))}</pre></details><div id="br-datasets"></div></section><section data-content="2" hidden><h2>Every trial, including caveats</h2><div class="table-scroll"><table><thead><tr>${['Path / topology','Repeat','Offered RPS','Successful RPS','TTFT P50','P95','P99 (ms)','ITL (ms)','Output tok/s','Goodput','Success','Error','Incomplete','Cache hit %'].map(v=>`<th>${v}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(label(r))}</td><td>${num(r.repetition,0)}</td><td>${num(r.rate)}</td><td>${num(r.rps)}</td><td>${num(r.ttft_p50_ms)}</td><td>${num(r.ttft_p95_ms)}</td><td class="${r.ttft_p99_ms>slo?'bad':''}">${num(r.ttft_p99_ms)}</td><td>${num(r.itl_ms)}</td><td>${num(r.output_tps)}</td><td>${num(r.goodput)}</td><td>${num(r.successful,0)}</td><td>${num(r.errored,0)}</td><td>${num(r.incomplete,0)}</td><td>${num(r.cache_hit_ratio==null?null:r.cache_hit_ratio*100)}</td></tr>`).join('')}</tbody></table></div>${rows.map(r=>`<details><summary>${esc(label(r))} · ${num(r.rate)} RPS · repeat ${num(r.repetition,0)}</summary><p>${esc((r.warnings||[]).join(' ')||'No adapter caveats recorded; inspect runtime evidence too.')}</p><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`).join('')}</section><section data-content="3" hidden><h2>Exact recipe & provenance</h2><p>Secrets and raw prompts are excluded from compact reports. Keep the original result files alongside the report for independent verification.</p><details open><summary>Experiment / version lock</summary><pre>${esc(JSON.stringify(e,null,2))}</pre></details><details><summary>Collection status and artifacts</summary><pre>${esc(JSON.stringify({generated_at:data.generated_at,status:data.status,artifacts:data.artifacts,warnings:data.warnings},null,2))}</pre></details></section>`;
    if(data.status==='partial'){
      const warning=document.createElement('div');warning.className='warning';
      warning.textContent=`Partial experiment: ${data.missing_trials?.length??'some'} planned trial(s) are unfinished or lack collected results. This is not a completed sweep.`;
      root.prepend(warning);
    }
    root.querySelectorAll('[data-tab]').forEach(btn=>btn.onclick=()=>{
      root.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b===btn);b.setAttribute('aria-selected',String(b===btn));});
      root.querySelectorAll('[data-content]').forEach(p=>p.hidden=p.dataset.content!==btn.dataset.tab);
    });
    const colors=['#6ee7c4','#8badff','#ffb877','#d4a4ff','#ff8da5','#81d9ef','#dad86f','#a9e6a3'];
    for(const target of root.querySelectorAll('[data-chart]')){
      const key=target.dataset.chart, valid=rows.filter(r=>typeof r[key]==='number'&&r.rate!=null);
      if(!valid.length){target.innerHTML='<p>No measured values available.</p>';continue;}
      const groups=[...new Set(valid.map(label))], maxX=Math.max(...valid.map(r=>r.rate),1), maxY=Math.max(...valid.map(r=>r[key]),key==='ttft_p99_ms'?slo:1)*1.08;
      const x=n=>55+n/maxX*410,y=n=>230-n/maxY*190;
      let svg='<svg viewBox="0 0 510 270" role="img" aria-label="'+esc(key)+' vs offered requests per second">';
      for(let i=0;i<=4;i++)svg+=`<line x1="55" x2="470" y1="${y(maxY*i/4)}" y2="${y(maxY*i/4)}" stroke="#2b3b52"/><text x="48" y="${y(maxY*i/4)+4}" text-anchor="end">${num(maxY*i/4,0)}</text><text x="${x(maxX*i/4)}" y="250" text-anchor="middle">${num(maxX*i/4)}</text>`;
      if(key==='ttft_p99_ms')svg+=`<line x1="55" x2="470" y1="${y(slo)}" y2="${y(slo)}" stroke="#ffb877" stroke-dasharray="5 4"/><text x="470" y="${y(slo)-5}" text-anchor="end">SLO</text>`;
      groups.forEach((g,i)=>{
        const points=valid.filter(r=>label(r)===g).sort((a,b)=>a.rate-b.rate), byRate=[...new Set(points.map(r=>r.rate))];
        const medians=byRate.map(rate=>{const values=points.filter(r=>r.rate===rate).map(r=>r[key]).sort((a,b)=>a-b);return {rate,value:(values[Math.floor((values.length-1)/2)]+values[Math.floor(values.length/2)])/2};});
        svg+=`<g data-series="${i}" fill="${colors[i%colors.length]}"><polyline fill="none" stroke="${colors[i%colors.length]}" stroke-width="2" points="${medians.map(r=>`${x(r.rate)},${y(r.value)}`).join(' ')}"/>${points.map(r=>`<circle cx="${x(r.rate)}" cy="${y(r[key])}" r="4" opacity=".8"><title>${esc(g)} · repeat ${num(r.repetition,0)} · ${num(r.rate)} offered RPS · ${num(r[key])} ${esc(key)} · ${num(r.errored,0)} errors / ${num(r.incomplete,0)} incomplete</title></circle>`).join('')}</g>`;
      });
      target.innerHTML=svg+'</svg><small>Lines: median across repetitions; points: individual trials. Hover for evidence.</small><div class="legend">'+groups.map((g,i)=>`<button data-toggle="${i}" style="border-color:${colors[i%colors.length]}">${esc(g)}</button>`).join('')+'</div>';
      target.querySelectorAll('[data-toggle]').forEach(b=>b.onclick=()=>{const s=target.querySelector(`[data-series="${b.dataset.toggle}"]`);s.style.display=s.style.display==='none'?'':'none';b.style.opacity=s.style.display==='none'?'.4':'1';});
    }
    const comparisons=[];
    for(const r of rows.filter(r=>r.routing==='llmd')){
      const c=rows.find(s=>s.routing==='service'&&s.topology.tp===r.topology.tp&&s.topology.replicas===r.topology.replicas&&s.rate===r.rate&&s.repetition===r.repetition&&s.dataset_checksum&&s.dataset_checksum===r.dataset_checksum);
      if(c)comparisons.push(`<tr><td>${esc(label(r))}</td><td>${num(r.rate)}</td><td>${num(r.repetition,0)}</td><td>${num(c.ttft_p99_ms)}</td><td>${num(r.ttft_p99_ms)}</td><td>${c.ttft_p99_ms!=null&&r.ttft_p99_ms>0?num(c.ttft_p99_ms/r.ttft_p99_ms,2)+'×':'N/A'}</td><td>${(r.warnings||[]).length||(c.warnings||[]).length?'Caveats':'Inspectable'}</td></tr>`);
    }
    root.querySelector('#br-comparisons').innerHTML=comparisons.length?`<p>Ratio = Service P99 / llm-d P99. Above 1 favors llm-d; below 1 favors Service. Caveats still apply.</p><div class="table-scroll"><table><thead><tr><th>Topology</th><th>RPS</th><th>Repeat</th><th>Service P99</th><th>llm-d P99</th><th>Ratio</th><th>Evidence</th></tr></thead><tbody>${comparisons.join('')}</tbody></table></div>`:'<p>No matched pairs with identical dataset checksums. No speedup is inferred.</p>';
    root.querySelector('#br-datasets').innerHTML=Object.entries(data.datasets||{}).map(([name,d])=>`<details open><summary>${esc(name)} · ${num(d.records,0)} prompts</summary><p>SHA256: ${esc(d.sha256)}</p>${['input_histogram','output_histogram','prefix_groups'].map(k=>{const entries=Object.entries(d[k]||{}),max=Math.max(1,...entries.map(([,v])=>v));return `<h3>${esc(k.replaceAll('_',' '))}</h3>`+entries.slice(0,30).map(([l,n])=>`<div class="bars"><span>${esc(l)}</span><div class="bar" style="width:${n/max*100}%"></div><span>${num(n,0)}</span></div>`).join('');}).join('')}<pre>${esc(JSON.stringify(d,null,2))}</pre></details>`).join('')||'<p>No observed dataset summary supplied. The recipe specification above is not an observed distribution.</p>';
  }
  function html(data) {
    validate(data);
    const json=JSON.stringify(data).replace(/</g,'\\u003c').replace(/>/g,'\\u003e').replace(/&/g,'\\u0026');
    return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(data.experiment.name)} · Benchmark report</title><style>${css}</style></head><body><main id="report-root"></main><script type="application/json" id="report-data">${json}</script><script>(${render.toString()})(document.getElementById('report-root'),JSON.parse(document.getElementById('report-data').textContent));</script></body></html>`;
  }
  globalThis.BenchmarkReport={css,render,html,normalize,validate};
})();
