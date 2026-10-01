# llm-d Benchmark Studio

Design a repeatable routing experiment, execute it locally against your cluster,
and inspect the evidence in a standalone HTML report. The static UI has no
backend, telemetry, CDN assets, or credential collection. Imported results are
processed inside your browser; they are not uploaded.

**Validation boundary:** unit tests, Chromium interaction tests, exported bundle
execution (`render` and `report`), and real Qwen tokenizer checks are exercised
locally. No end-to-end GPU-cluster run has yet validated this new stack. Treat
the generated manifests as inspectable experimental recipes, not a certified
production deployment or a promise of faster routing.

## Build and open the UI

From the repository root, with Python 3.11+ and Node 22+:

```bash
npm run build
npm run serve
```

Open <http://127.0.0.1:8000>. The build copies only known static assets and bundle
sources into `dist/`; no Node packages are required. Do not open `index.html`
directly with `file://` because the builder fetches its local model catalog and
bundle assets. The **exported report**, in contrast, works directly from disk
without a server or network access.

1. **Hardware:** choose GPU memory, approved GPU budget, and available slots per
   node. Optionally download `discovery.py`, run it locally with
   `python3 discovery.py --context YOUR_CONTEXT > inventory.json`, and import the
   inventory. Discovery is read-only and does not reserve resources. It honors
   `KUBECONFIG`; see [Select your kubeconfig and context](#select-your-kubeconfig-and-context)
   to choose a non-default configuration file before running it.
2. **Deployment:** auto-select a catalog model or choose one explicitly. Auto
   prefers the largest feasible model with at least two replicas, then the
   lowest feasible TP for that model. It may use TP4 on eight 24-GiB GPUs rather
   than eight TP1 replicas; select TP1 explicitly if that is the desired study.
3. **Workload:** configure input/output token buckets, independent prefix
   families, fixed/bucketed lengths or prefix ratios, popularity, and seed.
4. **Benchmark:** choose explicit offered rates or a pilot, duration, separate
   warmup, repetitions, and TTFT P99 objective.
5. **Export:** download the recipe ZIP and review its manifests, version lock,
   `experiment.yaml`, scripts, and this runbook before execution.

JSON documents in `.yaml` files are valid YAML 1.2. The runner deliberately
accepts this JSON subset only, avoiding executable YAML tags and parser
dependencies. Editing a recipe after a run starts requires a **new results
directory**, preventing accidental mixing of experiments.

## Pinned stack and supported scope

| Component | Pin |
| --- | --- |
| llm-d reference release | `v0.10.0` |
| Standalone router Helm chart / EPP | `v0.11.0` |
| GuideLLM client and result adapter | `0.8.0`, report schema `2` |
| vLLM runtime | `0.30.0` |
| Gateway API / Inference Extension reference | `v1.5.1` / `v1.5.0` |
| Model/tokenizer | Qwen3 4B/8B/14B/32B, immutable Hub revisions in `models.json` |

`versions.lock.json` records release references and source commits. Runtime and
client images may be replaced with verified digests from the same repositories;
the EPP image is selected by the pinned upstream chart/tag. Tags are explicit
but not cryptographically immutable: retain resolved `imageID` evidence from
the actual run. Do not silently upgrade a component or feed GuideLLM 0.5/0.6
results to this adapter. The old scripts remain for historical results.

The recipe uses the upstream standalone router chart and its
`default-plugins.yaml`. It does **not** copy historical tuned EPP weights,
replace plugins, install KServe, or alter cluster-wide dependencies. GPU memory
estimates include weights, full-context BF16/FP16 KV cache, and workspace, but
readiness and runtime capacity remain the final checks.

Supported planning scope is homogeneous, exclusive NVIDIA/CUDA GPUs and
aggregated prefill/decode with **node-local** TP. CPU-only Jobs generate data and
run GuideLLM. One GPU is allowed, but one replica cannot establish multi-replica
routing gains. Cross-node TP, MIG/time-sharing, pipeline/expert parallelism,
non-NVIDIA runtimes, quantization, and arbitrary user models need separately
validated recipes. Unknown labels, taints, available CPU/RAM, disk space, GPU
compute capability, and interconnect must be reviewed by the operator.

## Cluster prerequisites

- Local Python 3.11+, Bash, `kubectl`, Helm with OCI chart support, and a working
  kubeconfig. No local GPU, GuideLLM, or model weights are required.
- NVIDIA device plugin/runtime and compatible drivers on eligible GPU nodes.
- The pinned InferencePool CRD (`inferencepools.inference.networking.k8s.io`)
  installed by a cluster administrator using the upstream release instructions.
  The helper never installs CRDs or other cluster-wide components implicitly.
- A dedicated experiment namespace/name, permissions to create that namespace
  and the chart's namespaced RBAC/InferencePool/resources, list nodes/pods,
  inspect logs, exec into the temporary collector, and access pod metrics.
- A default StorageClass, or an explicit class in the recipe, for a 20-GiB
  ReadWriteOnce results PVC. Size it before running if necessary.
- Registry access for all pinned images and Hub access for model/tokenizer
  downloads. Model-cache volumes are ephemeral: model restarts can redownload
  weights. Allow substantial startup time, bandwidth, RAM, and ephemeral disk.
- Namespace policy permitting the selected vLLM and router images. For
  OpenShift, set top-level `"openshift": true` in the downloaded experiment
  **before the first run**, so client Jobs leave `fsGroup` to the SCC. Review
  runtime-image/SCC compatibility separately; this is not an OpenShift
  certification.

The catalog models are public and require no token for normal use. An existing
Secret named `<experiment>-hf` with key `token` can supply an optional runtime
`HF_TOKEN`. Do not place credentials in the experiment, UI, repository, or
exported reports. No `.env` or API key is needed for the static application.

## Execute the exported bundle

Unzip it into a dedicated local directory. Run the following commands **there**,
in a terminal on the machine with Python, `kubectl`, Helm, and cluster access.

### Select your kubeconfig and context

Use a trusted kubeconfig file already present on that machine. Set its absolute
path, list the contexts in that file, then replace `your-context-name` with the
intended entry from the **NAME** column:

```bash
export KUBECONFIG="/absolute/path/to/your/kubeconfig"
kubectl config get-contexts
CONTEXT="your-context-name"
```

- Both `kubectl` and Helm inherit the exported `KUBECONFIG` from the runner.
  The discovery helper uses the same environment variable.
- If `KUBECONFIG` is unset, the standard default is `~/.kube/config`; omit the
  `export` line to use that file when no overriding variable is already set.
- `--context` selects a **context name**, not a file path. The bundle's `run.sh`
  and `runner.py` do **not** have a `--kubeconfig` flag; use the environment
  variable instead. You do not need to change your kubeconfig's current context.
- Keep the kubeconfig and its credentials local. Do not upload them to the
  website, put them in `experiment.yaml`, or include them in shared bundles,
  reports, or Git commits. Kubeconfigs can invoke authentication plugins, so
  only use files from trusted sources.
- Continue in the same terminal. In a new terminal, repeat the kubeconfig and
  `CONTEXT` assignments before collecting, resuming, or cleaning up a run.

### Review, preflight, and run

```bash
bash run.sh render
bash run.sh --context "$CONTEXT" preflight
bash run.sh --context "$CONTEXT" run --confirm-context "$CONTEXT"
```

`render` and `report` are cluster-free and do not require a kubeconfig.
Preflight is read-only. Every mutating
action—including collection, which creates a temporary Pod—requires explicit
context confirmation: `--confirm-context` must match the selected context.
The runner refuses to overwrite resources lacking this
experiment's ownership label. Use a new dedicated namespace; do not relabel
someone else's resources to bypass the guard.

`run` renders and deploys each topology, freezes pilot-derived rates when
enabled, generates seeded datasets, alternates Service/llm-d trial order across
repetitions, resets owned model/EPP deployments between trials, executes
warmup and measurement separately, then collects results. Restarting between
trials can make the wall-clock duration much longer than the UI's measurement
estimate. Kubernetes Service routing is connection-level load balancing, **not
a promise of per-request round robin**.

For interrupted experiments:

```bash
bash run.sh status
bash run.sh --context "$CONTEXT" collect --confirm-context "$CONTEXT"
bash run.sh report
bash run.sh --context "$CONTEXT" run --resume --confirm-context "$CONTEXT"
```

Completed cells are skipped on resume; partial cells are retried explicitly.
The results directory is bound to the confirmed kubeconfig context and a SHA256
of its API-server address. Switching contexts or changing that address requires
a fresh `--output` directory; the server address itself is not put in reports.
Older state that already contains cluster work but lacks this binding cannot
be resumed for cluster actions. Local reporting remains available. This guard
prevents accidental target changes; it is not cryptographic cluster identity
and cannot detect a cluster replaced behind the same API-server address.
Jobs have bounded deadlines and no automatic retries. Retain the results PVC
until collection succeeds. A pilot that lacks a reliable capacity estimate
stops instead of inventing rates; inspect its results, or start a fresh recipe
with explicit rates. No concurrent runner processes should share an experiment
namespace or results directory.

After reviewing/collecting results:

```bash
bash run.sh --context "$CONTEXT" cleanup --confirm-context "$CONTEXT"
# Only when permanently discarding the retained cluster results:
bash run.sh --context "$CONTEXT" cleanup --confirm-context "$CONTEXT" --delete-results
```

Cleanup preserves the namespace and, unless explicitly requested, the results
PVC. It never deletes the entire namespace.

## Datasets and interpreting results

- Generation uses the pinned model tokenizer with special-token insertion
  disabled. Every prompt is re-encoded to verify its requested length; the
  strict runner rejects duplicates and verifies shared-token overlap, allowing
  one boundary token to change under retokenization.
- Input counts use exact proportional allocation; output lengths use seeded
  weighted sampling. A summary records observed buckets, prefix-family counts,
  reusable-token fraction, and a SHA256. The summary describes the combined
  warmup/measurement dataset; measured-subset checksums identify matched trials.
- Prefix families are shared between warmup and measurement, but full prompts
  are distinct. Both routing paths receive the same measured data in the same
  order. Different TP pilots may select different rates: compare routing at
  fixed topology/rate, not unrelated load points.
- Random-token content models length and cache locality, not semantic quality
  or all production behavior. Requested output lengths are not guarantees of
  observed output length; check actual token throughput and engine behavior.
- Prefix reuse potential is **not** a cache-hit rate. Cache ratios use measured
  counter deltas; missing metrics, changed pods, counter resets, and zero
  queries produce `N/A`, not zero.
- Input + requested output must fit the configured context. No silent clipping.
  Full-prefix reuse can duplicate prompts, so leave a unique suffix. Very short
  suffixes, prefix lengths clamped to short prompts, or tiny vocabularies can
  still trigger the generator's duplicate guard.
- Default limits bound recipes to 20 offered rates, 10 repetitions, 10,000
  prefix groups, and 200,000 generated prompts per rate. Generation retains
  prompts in memory: long-prompt datasets can hit client RAM/PVC limits sooner.

The `results/` directory contains:

| Artifact | Purpose |
| --- | --- |
| `raw/` | Original GuideLLM JSON/CSV/HTML, datasets, and summaries collected from PVC |
| `evidence/` | Model pod metadata/imageIDs and before/after cache metrics |
| `logs/` | Client Job logs, including failed Jobs |
| `router-tp*-rendered.yaml` | Exact rendered upstream router manifests |
| `state.json`, `experiment.json` | Resume state and provenance |
| `report-data.json`, `report.html` | Compact UI import and self-contained offline report |
| `results.zip` | Compact report/state bundle, **not** the potentially large raw files |

Import `report-data.json` or the helper-generated uncompressed `results.zip` in
**Explore results**. The UI caps the total selected files at 64 MiB and checks
ZIP contents against their central directory and end record. Import one compact
report at a time. Individual raw GuideLLM files require exactly one exported
experiment JSON in the same selection and explicit routing/topology assignment;
they never inherit the current builder form's model or hardware. Without dataset checksums,
the report does not infer matched speedups.

Unreadable/truncated raw trial files are retained on disk, flagged in report
warnings, and excluded from measurements instead of preventing other trials
from being reported. Affected sweeps stay partial. Corrupt cache snapshots
yield `N/A`; unreadable dataset summaries produce warnings. Inspect the
Reproducibility tab and original artifacts before trusting partial evidence.

Reports show individual trials, repetition medians, TTFT P50/P95/P99, output
throughput, goodput, errors, incompletes, dataset summaries, and provenance.
Partial sweeps are visibly marked. The best-observed throughput card requires
zero errors/incompletes, at least 95% of offered load, sufficient samples, and
P99 within the configured objective. It is not a confidence-bound production
capacity estimate. A P99 needs at least 368 successes even to bound a two-sided
95% interval nonparametrically; inspect the stored percentile intervals and
repeatability, not just that minimum count. Duration-cancelled requests and
client dispatch lag remain important caveats.

Service versus llm-d is an **end-to-end path comparison**, including proxy
overhead, not a scorer-only ablation. In matched comparisons, Service P99 /
llm-d P99 above 1 favors llm-d; below 1 favors Service. Report losses and ties
as honestly as gains. The demo is fabricated and prominently labeled.

## Tests

```bash
npm test
python3 -m unittest discover -s tests -p test_runner.py -v
python3 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m unittest discover -s tests -p test_browser.py -v
```

Browser tests build the site, run a loopback-only temporary server, exercise
mobile layouts/imports/exports, execute the downloaded runner without cluster
access, and verify that exported HTML makes no network requests. CI installs
Chromium's system dependencies with `playwright install --with-deps chromium`.

Optional real-tokenizer verification (Hub network access; **no model weights**):

```bash
.venv/bin/python -m pip install -r tests/requirements-tokenizer.txt
STUDIO_TOKENIZER_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_tokenizer.py -v
```

## GitHub Pages

The repository's **Benchmark Studio** workflow tests and builds on pushes/PRs.
Publishing is deliberately **manual**, and only from `main`:

1. Commit/push the reviewed implementation when ready.
2. In the fork's **Settings → Pages**, select **GitHub Actions** as the source.
3. In **Actions → Benchmark Studio → Run workflow**, select `main`.
4. The workflow runs tests, uploads only `dist/`, and deploys using the
   `github-pages` environment. Use the deployment URL shown by GitHub.

No benchmark results or credentials are included in the site artifact. Paths
are relative, so the UI works under a repository Pages subpath. Exported
reports remain local unless you deliberately share them. This runbook does
not imply that Pages has already been enabled or deployed.

## Upstream references

- [llm-d v0.10.0](https://github.com/llm-d/llm-d/tree/v0.10.0)
- [Standalone router chart v0.11.0](https://github.com/llm-d/llm-d-router/tree/v0.11.0/config/charts/llm-d-router-standalone)
- [GuideLLM v0.8.0](https://github.com/vllm-project/guidellm/tree/v0.8.0)
- [GuideLLM Kubernetes client guidance](https://vllm-project.github.io/guidellm/stable/examples/kubernetes-openshift-job/)
- [Custom GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
