// Shared deterministic planner. No cluster access, credentials, or external requests.
export const PRESETS = {
  mixed: {input: '128-512:35,513-1024:35,1025-2048:30', output: '16-64:50,65-128:50', groups: 32, mode: 'ratio', prefix: '0'},
  agents: {input: '512-1024:40,1025-2048:40,2049-3072:20', output: '16-64:50,65-256:50', groups: 64, mode: 'length', prefix: '128-256:50,257-512:50'},
  hot: {input: '512-1024:40,1025-2048:60', output: '16-64:50,65-128:50', groups: 1, mode: 'ratio', prefix: '0.8'}
};
export function buckets(spec) {
  const result = String(spec).split(',').map(part => {
    const m = part.trim().match(/^(\d+)-(\d+):(\d+)$/);
    if (!m) throw new Error('Buckets must use min-max:weight, separated by commas.');
    const [lo, hi, weight] = m.slice(1).map(Number);
    if (![lo, hi, weight].every(Number.isSafeInteger) || lo < 1 || hi < lo || weight < 1) throw new Error('Bucket lengths and weights must be positive, ordered integers.');
    return {lo, hi, weight};
  });
  if (result.length > 50) throw new Error('Use at most 50 buckets.');
  if (!Number.isSafeInteger(result.reduce((sum, b) => sum + b.weight, 0))) throw new Error('Total bucket weight exceeds the safe integer limit.');
  return result;
}
const integer = (n, name, min, max) => {
  if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${name} must be an integer between ${min} and ${max}.`);
};
export function validate(e) {
  if (e.schema_version !== 1) throw new Error('Unsupported experiment schema.');
  if (!/^[a-z](?:[a-z0-9-]{0,38}[a-z0-9])?$/.test(e.name)) throw new Error('Name must start with a letter, end with a letter or digit, and use at most 40 lowercase letters, digits or hyphens.');
  if (!/^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(e.namespace)) throw new Error('Invalid Kubernetes namespace.');
  const h = e.hardware, w = e.workload, b = e.benchmark;
  integer(h.budget, 'GPU budget', 1, 1024); integer(h.memory_gib, 'GPU memory', 8, 512);
  integer(e.context, 'Context length', 128, 32768); integer(e.sequences, 'Concurrent sequences', 1, 256);
  if (!Array.isArray(h.node_slots) || !h.node_slots.length) throw new Error('Enter at least one node GPU count.');
  h.node_slots.forEach(n => integer(n, 'GPUs per node', 0, 128));
  if (h.budget > h.node_slots.reduce((a, n) => a + n, 0)) throw new Error('GPU budget exceeds available node slots.');
  if (h.sharing !== 'exclusive') throw new Error('Automatic performance recipes require exclusive GPUs. MIG/time-shared pools must be benchmarked separately, not counted as physical GPUs.');
  if (h.vendor !== 'nvidia') throw new Error('This release generates NVIDIA/CUDA recipes only; other vendors need a separately validated pinned runtime.');
  if (typeof h.selector !== 'object' || Array.isArray(h.selector) || h.selector === null) throw new Error('Node selector must be a JSON object.');
  for (const [k,v] of Object.entries(h.selector)) {
    if (!/^[\w./-]+$/.test(k) || typeof v !== 'string' || !/^[\w.-]*$/.test(v)) throw new Error('Invalid node selector label.');
  }
  const inputs = buckets(w.input), outputs = buckets(w.output);
  if (Math.max(...inputs.map(v => v.hi)) + Math.max(...outputs.map(v => v.hi)) > e.context) throw new Error('Maximum input + output tokens exceed the configured context. Nothing will be silently clipped.');
  integer(w.groups, 'Prefix groups', 1, 10000); integer(w.seed, 'Seed', 0, 2147483647);
  if (!/^(uniform|zipf(?::(?:\d+(?:\.\d+)?))?)$/.test(w.popularity)) throw new Error('Popularity must be uniform, zipf, or zipf:alpha.');
  if (w.popularity.startsWith('zipf:') && Number(w.popularity.split(':')[1]) > 10) throw new Error('Zipf exponent must be between 0 and 10.');
  if (w.mode === 'ratio') {
    if (String(w.prefix).trim() === '' || !Number.isFinite(Number(w.prefix)) || Number(w.prefix) < 0 || Number(w.prefix) > 1) throw new Error('Prefix ratio must be between 0 and 1.');
    if (Number(w.prefix) === 1) throw new Error('100% reuse leaves no unique suffix. Reduce the prefix ratio.');
  } else if (w.mode === 'length') {
    if (/^\d+$/.test(w.prefix)) integer(Number(w.prefix), 'Prefix length', 1, e.context);
    else if (buckets(w.prefix).some(b => b.hi > e.context)) throw new Error('Prefix length exceeds context.');
  } else throw new Error('Unsupported prefix mode.');
  integer(b.seconds, 'Duration', 5, 3600); integer(b.warmup_seconds, 'Warmup', 0, 600);
  integer(b.repetitions, 'Repetitions', 1, 10); integer(b.slo_ms, 'TTFT SLO', 1, 60000);
  if (!Array.isArray(b.rates) || !b.rates.length || b.rates.length > 20 || b.rates.some(n => !Number.isFinite(n) || n <= 0 || n > 10000)) throw new Error('Enter 1–20 positive request rates, at most 10,000 each.');
  if (new Set(b.rates).size !== b.rates.length) throw new Error('Request rates must be unique.');
  if (!['float16','bfloat16'].includes(e.precision)) throw new Error('Unsupported precision.');
  return e;
}
export function plan(e, models) {
  validate(e);
  const h = e.hardware;
  const options = model => model.tp.flatMap(tp => {
    if (e.tp !== 'auto' && tp !== Number(e.tp)) return [];
    const kv = model.layers * 2 * model.kv_heads * model.head_dim * 2 * e.context * e.sequences / 2**30;
    const memory = (model.weights_gib + kv) / tp + 4; // conservative full-context KV + per-rank workspace
    const placements = h.node_slots.reduce((a, n) => a + Math.floor(n / tp), 0);
    const replicas = Math.min(placements, Math.floor(h.budget / tp));
    return memory <= h.memory_gib * .9 && replicas >= 1 && e.context <= model.context ? [{tp, replicas, gpus:tp*replicas, estimated_gib:memory, kv_gib:kv/tp}] : [];
  });
  let model = models.find(m => m.id === e.model);
  if (e.model === 'auto') {
    const candidates = [...models].reverse().filter(m => options(m).some(t => t.replicas >= 2));
    model = candidates[0] || [...models].reverse().find(m => options(m).length);
  }
  if (!model) throw new Error('No catalog model fits this hardware, context and concurrency. Reduce context/sequences or increase the GPU budget.');
  let topologies = options(model);
  if (!topologies.length) throw new Error('Requested model/TP does not fit. GPUs must fit together on a node; total cluster VRAM cannot be pooled implicitly.');
  if (!e.sweep) topologies = topologies.slice(0, 1);
  const warnings = ['Memory sizing is an estimate. This hardware profile is not GPU integration-tested; readiness and runtime KV capacity must pass before measurement.', 'Service versus llm-d is an end-to-end path comparison, not a scorer-only ablation.'];
  if (topologies.every(t => t.replicas === 1)) warnings.push('Only one replica fits: benchmark capacity, but do not claim multi-replica routing gains.');
  if (h.interconnect === 'pcie' && topologies.some(t => t.tp > 1)) warnings.push('PCIe-only tensor parallelism may be communication-bound. Prefer TP1 when feasible; inspect results rather than assuming larger TP is better.');
  const trials = topologies.length * e.benchmark.rates.length * e.benchmark.repetitions * 2;
  return {model, topologies, warnings, trials, estimated_minutes:trials*(e.benchmark.seconds+e.benchmark.warmup_seconds)/60};
}
export function resolved(e, models, lock) {
  return {...e, versions:lock, plan:plan(e, models)};
}
// JSON is YAML 1.2-compatible, so Kubernetes and Helm consume these files without unsafe string interpolation.
export const yaml = obj => JSON.stringify(obj, null, 2) + '\n';
export function manifests(e, t) {
  const labels = {'app.kubernetes.io/part-of':e.name, 'studio.llm-d.ai/experiment':e.name};
  const app = {...labels, app:`${e.name}-model`};
  const name = `${e.name}-model`, ns=e.namespace;
  const meta = n => ({name:n, namespace:ns, labels});
  const requests = {cpu:'4',memory:'24Gi','nvidia.com/gpu':t.tp};
  const deployment = {apiVersion:'apps/v1',kind:'Deployment',metadata:meta(name),spec:{replicas:t.replicas,strategy:{type:'Recreate'},selector:{matchLabels:{app:app.app}},template:{metadata:{labels:app},spec:{nodeSelector:e.hardware.selector,automountServiceAccountToken:false,tolerations:[{key:'nvidia.com/gpu',operator:'Exists',effect:'NoSchedule'}],containers:[{name:'modelserver',image:e.versions.runtime.image,command:['vllm','serve'],args:[e.plan.model.id,`--revision=${e.plan.model.revision}`,`--tokenizer-revision=${e.plan.model.revision}`,`--tensor-parallel-size=${t.tp}`,`--dtype=${e.precision}`,`--max-model-len=${e.context}`,`--max-num-seqs=${e.sequences}`,'--gpu-memory-utilization=0.9','--enable-prefix-caching','--port=8000'],ports:[{containerPort:8000}],resources:{requests,limits:{...requests}},env:[{name:'HF_TOKEN',valueFrom:{secretKeyRef:{name:`${e.name}-hf`,key:'token',optional:true}}}],volumeMounts:[{name:'shm',mountPath:'/dev/shm'},{name:'model-cache',mountPath:'/root/.cache/huggingface'}],startupProbe:{httpGet:{path:'/health',port:8000},periodSeconds:10,failureThreshold:180},readinessProbe:{httpGet:{path:'/health',port:8000},periodSeconds:5}}],volumes:[{name:'shm',emptyDir:{medium:'Memory',sizeLimit:'4Gi'}},{name:'model-cache',emptyDir:{}}]}}}};
  if (e.hardware.node_names?.length) deployment.spec.template.spec.affinity = {nodeAffinity:{requiredDuringSchedulingIgnoredDuringExecution:{nodeSelectorTerms:[{matchFields:[{key:'metadata.name',operator:'In',values:e.hardware.node_names}]}]}}};
  const service = {apiVersion:'v1',kind:'Service',metadata:meta(`${e.name}-service`),spec:{selector:{app:app.app},ports:[{name:'http',port:8000,targetPort:8000}]}};
  const values = {router:{modelServers:{matchLabels:{app:app.app},targetPorts:[{number:8000}],type:'vllm',protocol:'http'},epp:{image:{registry:'ghcr.io/llm-d',repository:'llm-d-router-endpoint-picker',tag:e.versions.router.version},pluginsConfigFile:'default-plugins.yaml',resources:{requests:{cpu:'2',memory:'2Gi'},limits:{memory:'8Gi'}}},proxy:{resources:{requests:{cpu:'2',memory:'2Gi'},limits:{memory:'8Gi'}}}}};
  const pvc = {apiVersion:'v1',kind:'PersistentVolumeClaim',metadata:meta(`${e.name}-results`),spec:{accessModes:['ReadWriteOnce'],resources:{requests:{storage:'20Gi'}},...(e.storage_class ? {storageClassName:e.storage_class} : {})}};
  return {'model.yaml':deployment,'service.yaml':service,'router.values.yaml':values,'results-pvc.yaml':pvc};
}
