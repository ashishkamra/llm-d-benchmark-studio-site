import {plan,resolved,manifests,yaml,buckets,PRESETS} from './core.js';
import {zip,unzip} from './zip.js';
const $=id=>document.getElementById(id), form=$('recipe-form');
let step=0, experiment, bundle={}, reportData=null, inventoryNames=[];
const [models,lock]=await Promise.all(['models.json','versions.lock.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw new Error(`Unable to load ${url}`);return r.json();}));
models.forEach(m=>{const option=document.createElement('option');option.value=m.id;option.textContent=m.id;$('model').append(option);});
const style=document.createElement('style');style.textContent=BenchmarkReport.css;document.head.append(style);
const value=name=>form.elements.namedItem(name).value;
const download=(name,blob)=>{const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);};
const jsonBlob=v=>new Blob([yaml(v)],{type:'application/json'});
function readForm(){return {schema_version:1,name:value('name'),namespace:value('namespace'),storage_class:value('storage'),hardware:{vendor:'nvidia',gpu:value('gpu'),memory_gib:Number(value('memory')),budget:Number(value('budget')),node_slots:value('slots').split(',').map(Number),node_names:inventoryNames,selector:JSON.parse(value('selector')),interconnect:value('interconnect'),sharing:value('sharing')},model:value('model'),tp:value('tp'),context:Number(value('context')),sequences:Number(value('sequences')),precision:value('precision'),sweep:form.elements.sweep.checked,workload:{input:value('input'),output:value('output'),groups:Number(value('groups')),mode:value('prefix-mode'),prefix:value('prefix'),popularity:value('popularity'),seed:Number(value('seed'))},benchmark:{rates:value('rates').split(',').map(Number),seconds:Number(value('seconds')),warmup_seconds:Number(value('warmup')),repetitions:Number(value('repetitions')),slo_ms:Number(value('slo')),pilot:form.elements.pilot.checked}};}
function showError(error){$('error').hidden=false;$('error').textContent=error.message;}
function barChart(id,spec){const target=$(id);target.replaceChildren();const bs=buckets(spec),sum=bs.reduce((s,b)=>s+b.weight,0);bs.forEach(b=>{const row=document.createElement('div');row.className='bar-row';const text=document.createElement('span');text.textContent=`${b.lo}–${b.hi}`;const track=document.createElement('div'),bar=document.createElement('div');bar.className='bar';bar.style.width=`${100*b.weight/sum}%`;track.append(bar);const pct=document.createElement('span');pct.textContent=`${(100*b.weight/sum).toFixed(0)}%`;row.append(text,track,pct);target.append(row);});}
function update() {
  try {
    experiment=resolved(readForm(),models,lock);
    $('error').hidden=true;
    const p=experiment.plan;
    $('topology-preview').innerHTML='<div class="topology-grid">'+p.topologies.map(t=>`<div class="stat"><b>${t.replicas}R × TP${t.tp}</b><span>${t.gpus} GPUs · ~${t.estimated_gib.toFixed(1)} GiB/rank</span></div>`).join('')+'</div>';
    $('plan-summary').replaceChildren();
    const info=document.createElement('div');info.className='callout';
    info.textContent=`${p.model.id} · ${p.trials} trials · ~${p.estimated_minutes.toFixed(0)} measurement/warmup minutes, plus model startups, dataset generation and collection. ${p.warnings.join(' ')}`;
    $('plan-summary').append(info);
    barChart('input-chart',experiment.workload.input);
    barChart('output-chart',experiment.workload.output);
    $('workload-summary').textContent=`${experiment.workload.groups} prefix groups · ${experiment.workload.popularity} popularity · seed ${experiment.workload.seed}. Actual token lengths and reuse are verified after generation; cache hit rate is measured independently.`;
    bundle={'experiment.yaml':yaml(experiment),'versions.lock.json':yaml(lock)};
    p.topologies.forEach(t=>{
      for(const [name,obj]of Object.entries(manifests(experiment,t))) bundle[`manifests/tp${t.tp}/${name}`]=yaml(obj);
    });
    bundle['run.sh']='#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"\nexec python3 "$ROOT/runner.py" --experiment "$ROOT/experiment.yaml" "$@"\n';
    bundle['README.txt']='llm-d Benchmark Studio\n\nRead the included README.md for prerequisites, methodology and limitations.\n\n1. Review experiment.yaml and manifests/.\n2. bash run.sh --context YOUR_CONTEXT preflight\n3. bash run.sh --context YOUR_CONTEXT run --confirm-context YOUR_CONTEXT\n4. If interrupted: bash run.sh --context YOUR_CONTEXT collect --confirm-context YOUR_CONTEXT\n5. bash run.sh report\n\nResults: results/report.html and results/report-data.json. Import the latter or results/results.zip into the UI.\nCPU-only Jobs generate datasets and run GuideLLM. Install the pinned InferencePool CRD first; the helper never installs cluster-wide dependencies. Credentials remain in your kubeconfig and optional existing Secret <experiment>-hf (key token). Cleanup preserves the results PVC unless --delete-results is explicitly passed.\nNo GPU-cluster integration run has yet validated this new recipe; inspect and test before relying on it.\n';
    const selected=$('artifact-select').value;
    $('artifact-select').replaceChildren();
    Object.keys(bundle).forEach(name=>{
      const o=document.createElement('option');o.value=name;o.textContent=name;$('artifact-select').append(o);
    });
    if(Object.hasOwn(bundle,selected)) $('artifact-select').value=selected;
    $('artifact-preview').textContent=bundle[$('artifact-select').value];
    return true;
  } catch(error) {
    showError(error);experiment=null;return false;
  }
}
function setStep(n){step=Math.min(4,Math.max(0,n));document.querySelectorAll('[data-panel]').forEach(p=>p.hidden=Number(p.dataset.panel)!==step);document.querySelectorAll('[data-step]').forEach(b=>b.classList.toggle('active',Number(b.dataset.step)===step));$('back').disabled=step===0;$('next').hidden=step===4;$('step-label').textContent=`Step ${step+1} of 5`;update();}
form.addEventListener('submit',e=>e.preventDefault());form.addEventListener('input',update);form.addEventListener('change',update);$('back').onclick=()=>setStep(step-1);$('next').onclick=()=>{if(update())setStep(step+1);};document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>setStep(Number(b.dataset.step)));
form.elements.gpu.onchange=()=>{const memory={L4:24,L40S:48,A100:80,H100:80,H200:141}[value('gpu')];if(memory)form.elements.memory.value=memory;update();};
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{const p=PRESETS[b.dataset.preset];for(const [k,v]of Object.entries(p))form.elements.namedItem(k==='mode'?'prefix-mode':k).value=v;document.querySelectorAll('[data-preset]').forEach(x=>x.classList.toggle('active',x===b));update();});
$('artifact-select').onchange=()=>{$('artifact-preview').textContent=bundle[$('artifact-select').value];};
async function text(url){const r=await fetch(url);if(!r.ok)throw new Error(`Missing bundle asset ${url}; build/serve the site using scripts/build_site.py.`);return r.text();}
$('download').onclick=async()=>{try{if(!update())return;const button=$('download');button.disabled=true;const snapshot={...bundle},name=experiment.name;const assets=['runner.py','generate_dataset.py','discovery.py','report.js','README.md'];const entries=await Promise.all(assets.map(async n=>[n,await text(`bundle/${n}`)]));download(`${name}-recipe.zip`,zip({...snapshot,...Object.fromEntries(entries)}));button.disabled=false;}catch(error){$('download').disabled=false;showError(error);}};
$('save-config').onclick=()=>{if(update())download(`${experiment.name}.json`,jsonBlob(experiment));};
$('discovery-download').onclick=async()=>{try{download('discover-gpus.py',new Blob([await text('bundle/discovery.py')],{type:'text/x-python'}));}catch(e){showError(e);}};
$('inventory').onchange=async()=>{try{const inv=JSON.parse(await $('inventory').files[0].text());if(inv.inventory_schema_version!==1||!Array.isArray(inv.pools))throw new Error('Not a discovery inventory.');$('inventory-pools').replaceChildren();inv.pools.forEach(pool=>{const b=document.createElement('button');b.type='button';b.textContent=`Use ${pool.product}: ${pool.node_slots.reduce((a,n)=>a+n,0)} slots`;b.onclick=()=>{form.elements.gpu.value='custom';form.elements.memory.value=pool.memory_gib;form.elements.slots.value=pool.node_slots.join(',');form.elements.budget.value=pool.node_slots.reduce((a,n)=>a+n,0);form.elements.sharing.value=pool.sharing;form.elements.selector.value=JSON.stringify(pool.selector);inventoryNames=pool.node_names;update();};$('inventory-pools').append(b);});}catch(e){showError(e);}};
function loadForm(e){const map={name:e.name,namespace:e.namespace,storage:e.storage_class,gpu:e.hardware.gpu,memory:e.hardware.memory_gib,budget:e.hardware.budget,slots:e.hardware.node_slots.join(','),selector:JSON.stringify(e.hardware.selector),interconnect:e.hardware.interconnect,sharing:e.hardware.sharing,model:e.model,tp:e.tp,context:e.context,sequences:e.sequences,precision:e.precision,input:e.workload.input,output:e.workload.output,groups:e.workload.groups,'prefix-mode':e.workload.mode,prefix:e.workload.prefix,popularity:e.workload.popularity,seed:e.workload.seed,rates:e.benchmark.rates.join(','),seconds:e.benchmark.seconds,warmup:e.benchmark.warmup_seconds,repetitions:e.benchmark.repetitions,slo:e.benchmark.slo_ms};for(const [k,v]of Object.entries(map))form.elements.namedItem(k).value=v??'';form.elements.sweep.checked=e.sweep;form.elements.pilot.checked=e.benchmark.pilot;inventoryNames=e.hardware.node_names||[];}
$('load-config').onchange=async()=>{try{const e=JSON.parse(await $('load-config').files[0].text());resolved(e,models,lock);loadForm(e);update();}catch(e){showError(e);}};
function showMode(results){$('builder').hidden=results;$('results').hidden=!results;$('results-mode').classList.toggle('active',results);$('builder-mode').classList.toggle('active',!results);}
$('builder-mode').onclick=()=>showMode(false);$('results-mode').onclick=()=>showMode(true);
function showReport(data){BenchmarkReport.validate(data);reportData=data;BenchmarkReport.render($('report-root'),data);$('export-report').disabled=false;$('result-error').hidden=true;}
$('result-files').onchange=async()=>{
  try {
    const selected=[...$('result-files').files],objects=[];
    if(selected.reduce((sum,f)=>sum+f.size,0)>64*1024*1024) throw new Error('Imports exceed 64 MiB in total. Use the compact report-data.json from the helper.');
    for(const f of selected){
      if(f.name.endsWith('.zip')){
        const files=unzip(await f.arrayBuffer());
        const entries=Object.entries(files).filter(([name])=>name.split('/').pop()==='report-data.json');
        if(entries.length!==1) throw new Error('Expected exactly one report-data.json in bundle.');
        objects.push(JSON.parse(entries[0][1]));
      } else objects.push(JSON.parse(await f.text()));
    }
    if(objects.some(o=>!o||typeof o!=='object'||Array.isArray(o))) throw new Error('Every imported file must contain a report or experiment object.');
    const compact=objects.filter(o=>o.report_schema_version===1);
    if(compact.length){
      if(objects.length!==1) throw new Error('Import one compact report at a time; multiple experiments cannot be silently combined.');
      showReport(compact[0]);return;
    }
    const recipes=objects.filter(o=>o.schema_version===1&&o.hardware);
    if(recipes.length!==1||!recipes[0].plan) throw new Error('Import an exported experiment JSON with the raw GuideLLM files (exactly one experiment).');
    const e=recipes[0],raw=objects.filter(o=>o.metadata?.guidellm_version);
    if(!raw.length) throw new Error('No recognized report found.');
    const routing=window.prompt('Assign these raw files to routing: service or llmd','llmd');
    if(!['service','llmd'].includes(routing)) throw new Error('Explicit routing assignment required.');
    const tp=Number(window.prompt('Tensor parallelism for these files',String(e.plan.topologies[0].tp)));
    const topology=e.plan.topologies.find(t=>t.tp===tp);
    if(!topology) throw new Error('Topology is not in this experiment.');
    const runs=raw.flatMap((r,i)=>BenchmarkReport.normalize(r,{id:`raw-${Date.now()}-${i}`,routing,topology,repetition:1}));
    showReport({report_schema_version:1,experiment:e,runs,datasets:{},warnings:['Manually assigned raw reports lack dataset checksums; no paired speedup is inferred.'],generated_at:new Date().toISOString(),status:'imported'});
  } catch(e) {
    $('result-error').hidden=false;$('result-error').textContent=e.message;
  }
};
$('export-report').onclick=()=>download(`${reportData.experiment.name}-report.html`,new Blob([BenchmarkReport.html(reportData)],{type:'text/html'}));
$('demo').onclick=()=>{if(!update())return;const e=structuredClone(experiment),t=e.plan.topologies[0],runs=[];for(const routing of ['service','llmd'])for(const rate of [1,2,4,8])for(let repetition=1;repetition<=3;repetition++){const ttft=(routing==='llmd'?130:180)+rate*(routing==='llmd'?35:80)+repetition*8;runs.push({id:`${routing}-${rate}-${repetition}`,routing,topology:t,rate,repetition,rps:rate*.99,successful:1200,errored:0,incomplete:0,ttft_p50_ms:ttft*.6,ttft_p95_ms:ttft*.9,ttft_p99_ms:ttft,itl_ms:12,output_tps:rate*100,goodput:ttft<e.benchmark.slo_ms?rate*.99:rate*.4,attainment:ttft<e.benchmark.slo_ms?1:.4,dataset_checksum:'DEMO-NOT-A-REAL-CHECKSUM',warnings:['Fabricated demo values.']});}showReport({report_schema_version:1,experiment:e,runs,datasets:{},demo:true,status:'demo',generated_at:new Date().toISOString()});};
update();setStep(0);
