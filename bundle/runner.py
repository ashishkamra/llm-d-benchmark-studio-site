#!/usr/bin/env python3
"""Local orchestrator for exported Benchmark Studio recipes (Python 3.11+).

Only run/deploy/cleanup change cluster resources, and require explicit context
confirmation. Cluster-wide CRDs are prerequisites, never installed implicitly.
"""

import argparse
import datetime
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

EXPECTED = {
    "llm_d": "v0.10.0",
    "router": "v0.11.0",
    "guidellm": "0.8.0",
    "runtime": "0.30.0",
}


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            invalid_constant(value)
        return result

    return json.loads(
        Path(path).read_text(),
        parse_constant=invalid_constant,
        parse_float=finite_float,
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def positive(value, name, minimum=1, maximum=10000):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"Invalid {name}: expected {minimum}–{maximum}.")


def integer(value, name, minimum=1, maximum=10000):
    positive(value, name, minimum, maximum)
    if type(value) is not int:
        raise ValueError(f"Invalid {name}: expected an integer.")


def validate(e):
    if e.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema.")
    if not re.fullmatch(
        r"[a-z](?:[a-z0-9-]{0,38}[a-z0-9])?", e["name"]
    ) or not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", e["namespace"]):
        raise ValueError("Invalid experiment name or namespace.")
    for key, version in EXPECTED.items():
        if e["versions"][key]["version"] != version:
            raise ValueError(f"This runner supports only {key} {version}.")
    images = {
        "router": "ghcr.io/llm-d/llm-d-router-endpoint-picker",
        "guidellm": "ghcr.io/vllm-project/guidellm",
        "runtime": "docker.io/vllm/vllm-openai",
    }
    for key, repository in images.items():
        image = e["versions"][key]["image"]
        tag = EXPECTED[key] if key == "router" else f"v{EXPECTED[key]}"
        digest = key != "router" and re.fullmatch(
            re.escape(repository) + r"@sha256:[0-9a-f]{64}", image
        )
        if image != f"{repository}:{tag}" and not digest:
            raise ValueError(
                f"{key} image must use its pinned release or an immutable digest."
            )
    if (
        e["versions"]["router"]["chart"]
        != "oci://ghcr.io/llm-d/charts/llm-d-router-standalone"
    ):
        raise ValueError("Unexpected upstream chart.")
    if e["hardware"]["vendor"] != "nvidia" or e["hardware"]["sharing"] != "exclusive":
        raise ValueError("Only exclusive NVIDIA GPUs are supported by this recipe.")
    h = e["hardware"]
    integer(h["budget"], "GPU budget", maximum=1024)
    integer(h["memory_gib"], "GPU memory", 8, 512)
    if (
        int(h["budget"]) != h["budget"]
        or not h["node_slots"]
        or any(type(n) is not int or not 0 <= n <= 128 for n in h["node_slots"])
    ):
        raise ValueError("GPU counts must be integers.")
    if sum(h["node_slots"]) < h["budget"]:
        raise ValueError("GPU budget exceeds node slots.")
    integer(e["context"], "context", 128, 32768)
    integer(e["sequences"], "sequences", 1, 256)
    if e["precision"] not in ("float16", "bfloat16"):
        raise ValueError("Unsupported precision.")
    model = e["plan"]["model"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model["id"]) or not re.fullmatch(
        r"[0-9a-f]{40}", model["revision"]
    ):
        raise ValueError(
            "Model and tokenizer must use a Hub ID and immutable revision."
        )
    for t in e["plan"]["topologies"]:
        if (
            type(t["tp"]) is not int
            or t["tp"] not in model["tp"]
            or type(t["replicas"]) is not int
            or t["replicas"] < 1
        ):
            raise ValueError("Invalid topology.")
        if (
            t["tp"] * t["replicas"] > h["budget"]
            or sum(n // t["tp"] for n in h["node_slots"]) < t["replicas"]
        ):
            raise ValueError("Topology exceeds the budget or node-local capacity.")
    if not e["plan"]["topologies"]:
        raise ValueError("No feasible topologies.")
    w, b = e["workload"], e["benchmark"]
    try:
        from generate_dataset import parse_buckets, generate_prefix_weights
    except ModuleNotFoundError:
        from dataset_generator.generate_dataset import (
            parse_buckets,
            generate_prefix_weights,
        )

    def workload_buckets(spec):
        if not re.fullmatch(r"\s*\d+-\d+:\d+\s*(?:,\s*\d+-\d+:\d+\s*)*", spec):
            raise ValueError("Buckets must use min-max:weight, separated by commas.")
        result = parse_buckets(spec)
        if len(result) > 50 or any(c < 1 for _, _, c in result):
            raise ValueError("Use at most 50 buckets with positive weights.")
        if (
            any(hi > 2**53 - 1 for _, hi, _ in result)
            or sum(c for _, _, c in result) > 2**53 - 1
        ):
            raise ValueError("Bucket values exceed the safe integer limit.")
        return result

    inputs, outputs = workload_buckets(w["input"]), workload_buckets(w["output"])
    if max(v[1] for v in inputs) + max(v[1] for v in outputs) > e["context"]:
        raise ValueError("Input plus output exceeds context; no silent clipping.")
    integer(w["groups"], "prefix groups", maximum=10000)
    integer(w["seed"], "seed", 0, 2147483647)
    if not re.fullmatch(r"uniform|zipf(?::\d+(?:\.\d+)?)?", w["popularity"]):
        raise ValueError("Popularity must be uniform, zipf, or zipf:alpha.")
    generate_prefix_weights(w["groups"], w["popularity"])
    if w["mode"] == "ratio":
        positive(float(w["prefix"]), "prefix ratio", 0, 1)
        if float(w["prefix"]) == 1:
            raise ValueError(
                "100% reuse leaves no unique suffix. Reduce the prefix ratio."
            )
    elif w["mode"] == "length":
        prefix = str(w["prefix"])
        if re.fullmatch(r"\d+", prefix):
            integer(int(prefix), "prefix length", 1, e["context"])
        elif any(hi > e["context"] for _, hi, _ in workload_buckets(prefix)):
            raise ValueError("Prefix length exceeds context.")
    else:
        raise ValueError("Unsupported prefix mode.")
    integer(b["seconds"], "duration", 5, 3600)
    integer(b["warmup_seconds"], "warmup", 0, 600)
    integer(b["repetitions"], "repetitions", 1, 10)
    integer(b["slo_ms"], "SLO", 1, 60000)
    if (
        not b["rates"]
        or len(b["rates"]) > 20
        or len(set(b["rates"])) != len(b["rates"])
    ):
        raise ValueError("Rates must be unique and nonempty (at most 20).")
    for rate in b["rates"]:
        positive(rate, "rate")
    return e


def resources(e, t):
    """Equivalent to core.js manifests. JSON documents are valid YAML 1.2."""
    name, ns = e["name"], e["namespace"]
    labels = {"app.kubernetes.io/part-of": name, "studio.llm-d.ai/experiment": name}
    app = f"{name}-model"

    def meta(n):
        return {"name": n, "namespace": ns, "labels": labels}

    req = {"cpu": "4", "memory": "24Gi", "nvidia.com/gpu": t["tp"]}
    m = e["plan"]["model"]
    container = {
        "name": "modelserver",
        "image": e["versions"]["runtime"]["image"],
        "command": ["vllm", "serve"],
        "args": [
            m["id"],
            f"--revision={m['revision']}",
            f"--tokenizer-revision={m['revision']}",
            f"--tensor-parallel-size={t['tp']}",
            f"--dtype={e['precision']}",
            f"--max-model-len={e['context']}",
            f"--max-num-seqs={e['sequences']}",
            "--gpu-memory-utilization=0.9",
            "--enable-prefix-caching",
            "--port=8000",
        ],
        "ports": [{"containerPort": 8000}],
        "resources": {"requests": req, "limits": req},
        "env": [
            {
                "name": "HF_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"{name}-hf",
                        "key": "token",
                        "optional": True,
                    }
                },
            }
        ],
        "volumeMounts": [
            {"name": "shm", "mountPath": "/dev/shm"},
            {"name": "model-cache", "mountPath": "/root/.cache/huggingface"},
        ],
        "startupProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "periodSeconds": 10,
            "failureThreshold": 180,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "periodSeconds": 5,
        },
    }
    spec = {
        "nodeSelector": e["hardware"]["selector"],
        "automountServiceAccountToken": False,
        "tolerations": [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
        "containers": [container],
        "volumes": [
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
            {"name": "model-cache", "emptyDir": {}},
        ],
    }
    if e["hardware"].get("node_names"):
        spec["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchFields": [
                                {
                                    "key": "metadata.name",
                                    "operator": "In",
                                    "values": e["hardware"]["node_names"],
                                }
                            ]
                        }
                    ]
                }
            }
        }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": meta(app),
        "spec": {
            "replicas": t["replicas"],
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"app": app}},
            "template": {"metadata": {"labels": {**labels, "app": app}}, "spec": spec},
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": meta(f"{name}-service"),
        "spec": {
            "selector": {"app": app},
            "ports": [{"name": "http", "port": 8000, "targetPort": 8000}],
        },
    }
    values = {
        "router": {
            "modelServers": {
                "matchLabels": {"app": app},
                "targetPorts": [{"number": 8000}],
                "type": "vllm",
                "protocol": "http",
            },
            "epp": {
                "image": {
                    "registry": "ghcr.io/llm-d",
                    "repository": "llm-d-router-endpoint-picker",
                    "tag": e["versions"]["router"]["version"],
                },
                "pluginsConfigFile": "default-plugins.yaml",
                "resources": {
                    "requests": {"cpu": "2", "memory": "2Gi"},
                    "limits": {"memory": "8Gi"},
                },
            },
            "proxy": {
                "resources": {
                    "requests": {"cpu": "2", "memory": "2Gi"},
                    "limits": {"memory": "8Gi"},
                }
            },
        }
    }
    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": meta(f"{name}-results"),
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": "20Gi"}},
        },
    }
    if e.get("storage_class"):
        pvc["spec"]["storageClassName"] = e["storage_class"]
    return {
        "model.yaml": deployment,
        "service.yaml": service,
        "router.values.yaml": values,
        "results-pvc.yaml": pvc,
    }


def number(v):
    return (
        v
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v >= 0
        else None
    )


def normalize(raw, context):
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("metadata"), dict)
        or raw["metadata"].get("version") != 2
        or raw.get("metadata", {}).get("guidellm_version") != "0.8.0"
    ):
        raise ValueError(
            "Expected GuideLLM 0.8.0 schema 2; historical reports require their original adapter."
        )
    rows = []
    if not isinstance(raw.get("benchmarks"), list) or not raw["benchmarks"]:
        raise ValueError("Empty benchmark report.")
    for i, b in enumerate(raw["benchmarks"]):
        if not isinstance(b, dict) or not isinstance(b.get("metrics"), dict):
            raise ValueError("Missing GuideLLM benchmark metrics.")
        m = b["metrics"]
        count = m.get("request_totals")
        if not isinstance(count, dict):
            raise ValueError("Missing GuideLLM request counts.")

        def mean(key):
            return number(((m.get(key) or {}).get("successful") or {}).get("mean"))

        s = (m.get("time_to_first_token_ms") or {}).get("successful") or {}

        def percentile(n):
            p = s.get("percentiles") or {}
            if isinstance(p, list):
                for item in p:
                    key, v = (
                        item
                        if isinstance(item, list)
                        else (item["percentile"], item["value"])
                    )
                    if float(key) == n:
                        return number(v)
                return None
            return number(p.get(str(n), p.get(f"{n}.0", p.get(f"p{n}"))))

        rate = number(
            context.get("rate", b.get("config", {}).get("strategy", {}).get("rate"))
        )
        successful, errored, incomplete = [
            number(count.get(k)) for k in ("successful", "errored", "incomplete")
        ]
        rps = mean("requests_per_second")
        warnings = []
        if None in (successful, errored, incomplete):
            warnings.append("Missing request counts.")
        if (errored or 0) + (incomplete or 0):
            warnings.append(
                "Failed or incomplete requests: inspect reliability before claiming capacity."
            )
        if (successful or 0) < 368:
            warnings.append(
                "Too few successful requests to bound P99 with a two-sided 95% interval."
            )
        if rate is not None and rps is not None and rps < rate * 0.95:
            warnings.append(
                "Achieved successful request rate is below 95% of offered load."
            )
        if not context.get("dataset_checksum"):
            warnings.append(
                "Dataset checksum unavailable: matched-workload provenance is unverified."
            )
        dispatch = mean("request_dispatch_delay")
        rows.append(
            {
                **context,
                "id": f"{context.get('id', 'import')}-{i}",
                "rate": rate,
                "rps": rps,
                "successful": successful,
                "errored": errored,
                "incomplete": incomplete,
                "ttft_mean_ms": number(s.get("mean")),
                "ttft_p50_ms": number(s.get("median"))
                if number(s.get("median")) is not None
                else percentile(50),
                "ttft_p95_ms": percentile(95),
                "ttft_p99_ms": percentile(99),
                "ttft_p99_ci": (s.get("percentile_cis") or {}).get(
                    "99.0", (s.get("percentile_cis") or {}).get("99")
                ),
                "itl_ms": mean("inter_token_latency_ms"),
                "output_tps": mean("output_tokens_per_second"),
                "goodput": mean("request_goodput"),
                "attainment": number(m.get("slo_attainment")),
                "dispatch_ms": dispatch * 1000 if dispatch is not None else None,
                "warnings": warnings,
            }
        )
    return rows


def report_html(data, renderer):
    payload = (
        json.dumps(data, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    # The renderer contains its own HTML exporter. Its literal closing script
    # tags must not terminate this enclosing script in the HTML parser.
    renderer = re.sub(r"</script", r"<\\/script", renderer, flags=re.IGNORECASE)
    # The shared renderer exposes both rendering and its CSS; no JS execution on the host.
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + html.escape(data["experiment"]["name"])
        + ' · Benchmark report</title></head><body><main id="report-root"></main><script type="application/json" id="report-data">'
        + payload
        + "</script><script>"
        + renderer
        + '\nconst style=document.createElement("style");style.textContent=BenchmarkReport.css;document.head.append(style);BenchmarkReport.render(document.getElementById("report-root"),BenchmarkReport.validate(JSON.parse(document.getElementById("report-data").textContent)));</script></body></html>'
    )


class Runner:
    def __init__(self, experiment, context=None, output=None):
        self.path = Path(experiment).resolve()
        self.root = self.path.parent
        # Exported JSON is YAML 1.2. No executable YAML tags or shell evaluation.
        sys.path.insert(0, str(self.root))
        self.e = validate(read_json(self.path))
        self.name, self.ns = self.e["name"], self.e["namespace"]
        self.release = f"{self.name}-router"
        self.context = context
        self.out = Path(output).resolve() if output else self.root / "results"
        self.state_path = self.out / "state.json"
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.state = (
            read_json(self.state_path)
            if self.state_path.exists()
            else {
                "experiment_hash": self.hash,
                "trials": {},
                "datasets": {},
                "created_at": timestamp(),
            }
        )
        if self.state["experiment_hash"] != self.hash:
            raise ValueError(
                "Recipe changed since this results directory was created. Choose a fresh --output directory to avoid mixing experiments."
            )

    def execute(self, argv, *, input=None, allow_failure=False):
        print("+", " ".join(argv), flush=True)
        p = subprocess.run(argv, input=input, text=True, capture_output=True)
        if p.returncode and not allow_failure:
            raise RuntimeError(p.stderr or p.stdout or f"Command failed: {argv[0]}")
        return p

    def k(self, *args, **kwargs):
        return self.execute(
            ["kubectl"]
            + (["--context", self.context] if self.context else [])
            + list(args),
            **kwargs,
        )

    def helm(self, *args):
        return self.execute(
            ["helm"]
            + (["--kube-context", self.context] if self.context else [])
            + list(args)
        )

    def save(self):
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(self.state_path, self.state)
        write_json(self.out / "experiment.json", self.e)

    def confirm(self, supplied):
        current = self.context or self.k("config", "current-context").stdout.strip()
        if supplied != current:
            raise ValueError(
                f"Cluster mutation requires --confirm-context {current!r}. Review the kubeconfig context first."
            )
        self.context = current
        server = self.k(
            "config",
            "view",
            "--minify",
            "-o",
            "jsonpath={.clusters[0].cluster.server}",
        ).stdout.strip()
        if not server:
            raise ValueError("Cannot identify the execution target from kubeconfig.")
        target = {
            "context": current,
            "server_sha256": hashlib.sha256(server.encode()).hexdigest(),
        }
        recorded = self.state.get("execution_target")
        if recorded is not None and recorded != target:
            raise ValueError(
                "Execution target changed. Use a fresh --output directory; results from different targets cannot be resumed or mixed."
            )
        if recorded is None and any(
            self.state.get(k) for k in ("trials", "datasets", "topology")
        ):
            raise ValueError(
                "Existing state has no verified execution target. Use a fresh --output directory for cluster actions; local reporting remains available."
            )
        self.state["execution_target"] = target
        self.save()

    def owned(self, kind, name, namespace=True):
        args = ["get", kind, name, "-o", "json", "--ignore-not-found"]
        if namespace:
            args.extend(["-n", self.ns])
        p = self.k(*args)
        if p.stdout.strip():
            existing = json.loads(p.stdout)
            if (
                existing["metadata"].get("labels", {}).get("studio.llm-d.ai/experiment")
                != self.name
            ):
                raise ValueError(
                    f"Refusing to modify unowned {kind}/{name}. Use a dedicated experiment namespace/name."
                )

    def apply(self, obj):
        self.owned(
            obj["kind"].lower(), obj["metadata"]["name"], obj["kind"] != "Namespace"
        )
        return self.k("apply", "-f", "-", input=json.dumps(obj))

    def preflight(self):
        for tool in ("kubectl", "helm"):
            if not shutil.which(tool):
                raise ValueError(f"{tool} is required on this machine.")
        print("Pinned stack:", EXPECTED)
        self.k("version", "--client=true")
        self.helm("version", "--short")
        self.k("get", "crd", "inferencepools.inference.networking.k8s.io")
        self.owned("namespace", self.ns, False)
        for resource in (
            "deployments",
            "services",
            "configmaps",
            "jobs",
            "pods",
            "persistentvolumeclaims",
        ):
            result = self.k(
                "auth", "can-i", "create", resource, "-n", self.ns
            ).stdout.strip()
            if result != "yes":
                raise ValueError(
                    f"Permission to create {resource} in {self.ns} is required."
                )
        nodes = json.loads(self.k("get", "nodes", "-o", "json").stdout)
        pods = json.loads(self.k("get", "pods", "-A", "-o", "json").stdout)
        pods["items"] = [
            p
            for p in pods["items"]
            if not (
                p["metadata"].get("namespace") == self.ns
                and p["metadata"].get("labels", {}).get("studio.llm-d.ai/experiment")
                == self.name
            )
        ]
        try:
            from discovery import inventory
        except ModuleNotFoundError:
            from studio.discovery import inventory

        inv = inventory(nodes, pods)
        eligible = []
        selector = self.e["hardware"]["selector"]
        requested = set(self.e["hardware"].get("node_names", []))
        for node in nodes["items"]:
            labels = node["metadata"].get("labels", {})
            if all(labels.get(k) == v for k, v in selector.items()) and (
                not requested or node["metadata"]["name"] in requested
            ):
                eligible.append(node["metadata"]["name"])
        pool_nodes = [
            (p, n, slots)
            for p in inv["pools"]
            for n, slots in zip(p["node_names"], p["node_slots"])
            if n in eligible
        ]
        products = {p["product"] for p, _, _ in pool_nodes}
        if len(products) != 1 or not pool_nodes:
            raise ValueError(
                "Choose one available homogeneous GPU pool using a node selector or discovery inventory."
            )
        if any(
            p["sharing"] != "exclusive"
            or p["memory_gib"] < self.e["hardware"]["memory_gib"]
            for p, _, _ in pool_nodes
        ):
            raise ValueError(
                "Discovered GPU memory/access mode differs from recipe. Verify labels rather than silently resizing."
            )
        if sum(s for _, _, s in pool_nodes) < self.e["hardware"]["budget"]:
            raise ValueError("Available GPU slots no longer meet the approved budget.")
        for t in self.e["plan"]["topologies"]:
            if sum(s // t["tp"] for _, _, s in pool_nodes) < t["replicas"]:
                raise ValueError(
                    "Node-local slots no longer accommodate this TP topology."
                )
        print("Preflight passed. GPU slots are not reserved until scheduling succeeds.")

    def render(self):
        for t in self.e["plan"]["topologies"]:
            for name, obj in resources(self.e, t).items():
                write_json(self.root / "manifests" / f"tp{t['tp']}" / name, obj)
        print("Rendered YAML-compatible JSON manifests and upstream Helm values.")

    def deploy(self, t):
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": self.ns,
                    "labels": {"studio.llm-d.ai/experiment": self.name},
                },
            }
        )
        self.render()
        objs = resources(self.e, t)
        for name in ("results-pvc.yaml", "model.yaml", "service.yaml"):
            self.apply(objs[name])
        values = self.root / "manifests" / f"tp{t['tp']}" / "router.values.yaml"
        charts = self.e["versions"]["router"]
        # A pre-existing release must have our model selector and default EPP policy.
        existing = json.loads(self.helm("list", "-n", self.ns, "-o", "json").stdout)
        if any(r["name"] == self.release for r in existing):
            v = json.loads(
                self.helm(
                    "get", "values", self.release, "-n", self.ns, "-o", "json"
                ).stdout
            )
            if v.get("router", {}).get("modelServers", {}).get("matchLabels") != {
                "app": f"{self.name}-model"
            }:
                raise ValueError(
                    "Refusing to modify a Helm release selecting unrelated servers."
                )
        rendered = self.helm(
            "template",
            self.release,
            charts["chart"],
            "--version",
            charts["version"],
            "-n",
            self.ns,
            "-f",
            str(values),
        ).stdout
        (self.out / f"router-tp{t['tp']}-rendered.yaml").write_text(rendered)
        self.helm(
            "upgrade",
            "--install",
            self.release,
            charts["chart"],
            "--version",
            charts["version"],
            "-n",
            self.ns,
            "-f",
            str(values),
            "--wait",
            "--timeout",
            "10m",
        )
        self.k(
            "rollout",
            "status",
            f"deployment/{self.name}-model",
            "-n",
            self.ns,
            "--timeout=30m",
        )
        scripts = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.name}-scripts",
                "namespace": self.ns,
                "labels": {"studio.llm-d.ai/experiment": self.name},
            },
            "data": {
                "generate_dataset.py": (self.root / "generate_dataset.py").read_text()
            },
        }
        self.apply(scripts)
        self.state["topology"] = t
        self.save()

    def router_service(self):
        items = json.loads(
            self.k("get", "services", "-n", self.ns, "-o", "json").stdout
        )["items"]
        candidates = [
            s
            for s in items
            if s["metadata"].get("annotations", {}).get("meta.helm.sh/release-name")
            == self.release
            and any(p["port"] == 8081 for p in s["spec"].get("ports", []))
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Cannot uniquely discover the pinned standalone router HTTP service."
            )
        return candidates[0]["metadata"]["name"]

    def epp_deployments(self):
        items = json.loads(
            self.k("get", "deployments", "-n", self.ns, "-o", "json").stdout
        )["items"]
        return [
            d["metadata"]["name"]
            for d in items
            if d["metadata"].get("annotations", {}).get("meta.helm.sh/release-name")
            == self.release
            and any(
                "llm-d-router-endpoint-picker" in c.get("image", "")
                for c in d["spec"]["template"]["spec"]["containers"]
            )
        ]

    def reset(self):
        self.owned("deployment", f"{self.name}-model")
        self.k("rollout", "restart", f"deployment/{self.name}-model", "-n", self.ns)
        self.k(
            "rollout",
            "status",
            f"deployment/{self.name}-model",
            "-n",
            self.ns,
            "--timeout=30m",
        )
        epps = self.epp_deployments()
        if len(epps) != 1:
            raise ValueError("Expected one default active EPP deployment.")
        for d in epps:
            self.k("rollout", "restart", f"deployment/{d}", "-n", self.ns)
            self.k(
                "rollout", "status", f"deployment/{d}", "-n", self.ns, "--timeout=10m"
            )

    def job(self, identifier, command, seconds, mounted_scripts=False):
        full_name = f"{self.name}-{identifier}"
        name = (
            full_name
            if len(full_name) <= 63
            else (
                full_name[:50].rstrip("-")
                + "-"
                + hashlib.sha256(full_name.encode()).hexdigest()[:12]
            )
        )
        labels = {"studio.llm-d.ai/experiment": self.name}
        self.owned("job", name)
        self.k(
            "delete", "job", name, "-n", self.ns, "--ignore-not-found", "--wait=true"
        )
        mounts = [
            {"name": "results", "mountPath": "/results"},
            {"name": "home", "mountPath": "/home/guidellm"},
            {"name": "tmp", "mountPath": "/tmp"},
        ]
        volumes = [
            {
                "name": "results",
                "persistentVolumeClaim": {"claimName": f"{self.name}-results"},
            },
            {"name": "home", "emptyDir": {}},
            {"name": "tmp", "emptyDir": {}},
        ]
        if mounted_scripts:
            mounts.append({"name": "scripts", "mountPath": "/scripts"})
            volumes.append(
                {"name": "scripts", "configMap": {"name": f"{self.name}-scripts"}}
            )
        pod = {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "client",
                    "image": self.e["versions"]["guidellm"]["image"],
                    "command": command,
                    "resources": {
                        "requests": {"cpu": "2", "memory": "4Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": mounts,
                }
            ],
            "volumes": volumes,
        }
        # Vanilla Kubernetes PVC access; OpenShift allocates UID/GID through its SCC.
        if not self.e.get("openshift", False):
            pod["securityContext"]["fsGroup"] = 1001
        obj = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": self.ns, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": seconds,
                "template": {
                    "metadata": {
                        "labels": labels,
                        "annotations": {"sidecar.istio.io/inject": "false"},
                    },
                    "spec": pod,
                },
            },
        }
        self.apply(obj)
        deadline = time.monotonic() + seconds + 120
        while time.monotonic() < deadline:
            status = json.loads(
                self.k("get", "job", name, "-n", self.ns, "-o", "json").stdout
            ).get("status", {})
            if status.get("succeeded"):
                break
            if status.get("failed"):
                self.capture_job_logs(name)
                raise RuntimeError(
                    f"Job {name} failed. No automatic retry; collect partial results and inspect logs."
                )
            time.sleep(5)
        else:
            self.capture_job_logs(name)
            raise TimeoutError(
                f"Job {name} did not complete; its activeDeadlineSeconds bounds execution."
            )
        self.capture_job_logs(name)

    def capture_job_logs(self, name):
        result = self.k("logs", f"job/{name}", "-n", self.ns, allow_failure=True)
        path = self.out / "logs" / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stdout + result.stderr)

    def dataset(self, rate, index):
        key = f"rate-{index}"
        if key in self.state["datasets"]:
            return key
        b, w = self.e["benchmark"], self.e["workload"]
        warm = (
            math.ceil(rate * b["warmup_seconds"] * 1.5 + 100)
            if b["warmup_seconds"]
            else 0
        )
        measured = math.ceil(rate * b["seconds"] * 1.5 + 100)
        total = warm + measured
        if total > 200000:
            raise ValueError(
                "Dataset exceeds 200,000 prompts per rate. Reduce rates/duration, or explicitly adapt storage/generation for larger experiments."
            )
        m = self.e["plan"]["model"]
        args = [
            "python3",
            "/scripts/generate_dataset.py",
            "--model",
            m["id"],
            "--revision",
            m["revision"],
            "--buckets",
            w["input"],
            "--output-buckets",
            w["output"],
            "--num-prefixes",
            str(w["groups"]),
            "--prefix-popularity",
            w["popularity"],
            "--seed",
            str(w["seed"] + index),
            "--total",
            str(total),
            "--max-context",
            str(self.e["context"]),
            "--strict",
            "--summary",
            f"/results/{key}-summary.json",
            "--output",
            f"/results/{key}-all.jsonl",
            "--prefix-ratio" if w["mode"] == "ratio" else "--prefix-length",
            str(w["prefix"]),
        ]
        # Splitting preserves the prefix families but uses distinct full prompts.
        split = "import pathlib,sys; p=pathlib.Path('/results'); lines=(p/sys.argv[1]).read_text().splitlines(True); n=int(sys.argv[2]); (p/sys.argv[3]).write_text(''.join(lines[:n])); (p/sys.argv[4]).write_text(''.join(lines[n:]))"
        code = "import subprocess,sys; from importlib.metadata import version; assert version('guidellm')=='0.8.0'; subprocess.run(sys.argv[1:],check=True)"
        self.job(f"dataset-{index}", ["python3", "-c", code, *args], 3600, True)
        self.job(
            f"split-{index}",
            [
                "python3",
                "-c",
                split,
                f"{key}-all.jsonl",
                str(warm),
                f"{key}-warm.jsonl",
                f"{key}-measure.jsonl",
            ],
            600,
        )
        self.state["datasets"][key] = {
            "rate": rate,
            "warm_count": warm,
            "measure_count": measured,
        }
        self.save()
        return key

    def benchmark_command(
        self, routing, dataset, filename, rate, seconds, count, pilot=False
    ):
        service = (
            f"{self.name}-service" if routing == "service" else self.router_service()
        )
        port = 8000 if routing == "service" else 8081
        m = self.e["plan"]["model"]
        backend = {
            "kind": "openai_http",
            "target": f"http://{service}.{self.ns}.svc.cluster.local:{port}",
            "model": m["id"],
            "request_format": "/v1/completions",
            "stream": True,
        }
        profile = (
            {"kind": "concurrent", "streams": max(1, self.e["sequences"] // 2)}
            if pilot
            else {"kind": "constant", "rate": rate}
        )
        return [
            "guidellm",
            "run",
            "--backend",
            json.dumps(backend),
            "--profile",
            json.dumps(profile),
            "--constraint",
            f"kind=max_duration,seconds={seconds}",
            "--constraint",
            f"kind=max_requests,count={count}",
            "--data",
            f"kind=json_file,path=/results/{dataset}",
            "--data-column-mapper",
            json.dumps(
                {
                    "kind": "generative_column_mapper",
                    "column_mappings": {
                        "text_column": "prompt",
                        "prompt_tokens_count_column": "prompt_tokens_count",
                        "output_tokens_count_column": "output_tokens_count",
                    },
                }
            ),
            "--tokenizer",
            json.dumps(
                {
                    "kind": "huggingface_auto",
                    "model": m["id"],
                    "load_kwargs": {"revision": m["revision"]},
                }
            ),
            "--data-loader",
            "kind=pytorch,shuffle=false,num_workers=0",
            "--seed",
            f"kind=static,value={self.e['workload']['seed']}",
            "--metrics",
            json.dumps(
                {
                    "kind": "generative",
                    "sample_size": 50,
                    "slo": {"ttft_ms": self.e["benchmark"]["slo_ms"]},
                }
            ),
            "--output",
            f"kind=json,path=/results/{filename}.json",
            "--output",
            f"kind=csv,path=/results/{filename}.csv",
            "--output",
            f"kind=html,path=/results/{filename}.html",
            "--disable-console-interactive",
        ]

    def metrics(self, label):
        pods = json.loads(
            self.k(
                "get",
                "pods",
                "-n",
                self.ns,
                "-l",
                f"app={self.name}-model",
                "-o",
                "json",
            ).stdout
        )
        record = {}
        write_json(self.out / "evidence" / f"{label}-pods.json", pods)
        for pod in pods["items"]:
            name = pod["metadata"]["name"]
            result = self.k(
                "get",
                "--raw",
                f"/api/v1/namespaces/{self.ns}/pods/{name}:8000/proxy/metrics",
                allow_failure=True,
            )
            record[name] = result.stdout if result.returncode == 0 else None
        write_json(self.out / "evidence" / f"{label}-metrics.json", record)
        return record

    def run(self, resume=False):
        if self.state["trials"] and not resume:
            raise ValueError(
                "Existing trials found. Use --resume to skip completed trials and explicitly retry partial cells."
            )
        self.save()
        self.preflight()
        for t in self.e["plan"]["topologies"]:
            self.deploy(t)
            rates = self.state.get(f"rates-tp{t['tp']}")
            if rates is None:
                rates = self.e["benchmark"]["rates"]
                if self.e["benchmark"]["pilot"]:
                    key = self.dataset(rates[0], 9000 + t["tp"])
                    count = self.state["datasets"][key]["measure_count"]
                    self.reset()
                    filename = f"pilot-tp{t['tp']}"
                    self.job(
                        filename,
                        self.benchmark_command(
                            "service",
                            f"{key}-measure.jsonl",
                            filename,
                            rates[0],
                            min(30, self.e["benchmark"]["seconds"]),
                            count,
                            True,
                        ),
                        600,
                    )
                    self.collect(report=False)
                    r = normalize(
                        read_json(self.out / "raw" / f"{filename}.json"),
                        {"id": filename, "topology": t, "routing": "service"},
                    )[0]
                    if (
                        r["rps"] is None
                        or r["rps"] <= 0
                        or r["errored"]
                        or r["incomplete"]
                    ):
                        raise ValueError(
                            "Pilot lacks a complete reliable capacity estimate. Inspect it or disable pilot and enter explicit rates."
                        )
                    rates = [
                        round(r["rps"] * fraction, 3)
                        for fraction in (0.25, 0.5, 0.75, 1)
                    ]
                self.state[f"rates-tp{t['tp']}"] = rates
                self.save()
            for index, rate in enumerate(rates):
                # Different TP pilots may produce different load schedules: compare routing at fixed TP, never implicitly match different rates.
                dataset = self.dataset(rate, t["tp"] * 100 + index)
                counts = self.state["datasets"][dataset]
                for repetition in range(1, self.e["benchmark"]["repetitions"] + 1):
                    modes = (
                        ["service", "llmd"] if repetition % 2 else ["llmd", "service"]
                    )
                    for mode in modes:
                        key = f"tp{t['tp']}-r{index}-n{repetition}-{mode}"
                        if (
                            self.state["trials"].get(key, {}).get("status")
                            == "completed"
                        ):
                            continue
                        self.state["trials"][key] = {
                            "id": key,
                            "routing": mode,
                            "topology": t,
                            "rate": rate,
                            "repetition": repetition,
                            "dataset": dataset,
                            "status": "started",
                            "started_at": timestamp(),
                        }
                        self.save()
                        self.reset()
                        if counts["warm_count"]:
                            self.job(
                                f"{key}-warm",
                                self.benchmark_command(
                                    mode,
                                    f"{dataset}-warm.jsonl",
                                    f"{key}-warm",
                                    rate,
                                    self.e["benchmark"]["warmup_seconds"],
                                    counts["warm_count"],
                                ),
                                self.e["benchmark"]["warmup_seconds"] + 600,
                            )
                        self.metrics(f"{key}-before")
                        self.job(
                            key,
                            self.benchmark_command(
                                mode,
                                f"{dataset}-measure.jsonl",
                                key,
                                rate,
                                self.e["benchmark"]["seconds"],
                                counts["measure_count"],
                            ),
                            self.e["benchmark"]["seconds"] + 600,
                        )
                        self.metrics(f"{key}-after")
                        self.state["trials"][key].update(
                            status="completed", ended_at=timestamp()
                        )
                        self.save()
        self.collect()

    def collect(self, report=True):
        """Mount retained PVC in a temporary pod; export even after Jobs exit."""
        self.out.mkdir(parents=True, exist_ok=True)
        name = f"{self.name}-export"
        labels = {"studio.llm-d.ai/experiment": self.name}
        self.owned("pod", name)
        self.k(
            "delete", "pod", name, "-n", self.ns, "--ignore-not-found", "--wait=true"
        )
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": self.ns, "labels": labels},
            "spec": {
                "restartPolicy": "Never",
                "automountServiceAccountToken": False,
                "securityContext": {"runAsNonRoot": True},
                "containers": [
                    {
                        "name": "export",
                        "image": self.e["versions"]["guidellm"]["image"],
                        "command": ["python3", "-c", "import time; time.sleep(3600)"],
                        "volumeMounts": [
                            {
                                "name": "results",
                                "mountPath": "/results",
                                "readOnly": True,
                            }
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "results",
                        "persistentVolumeClaim": {"claimName": f"{self.name}-results"},
                    }
                ],
            },
        }
        # Prefer the existing PVC attachment node (RWO) without guessing storage topology.
        jobs = json.loads(
            self.k(
                "get",
                "pods",
                "-n",
                self.ns,
                "-l",
                f"studio.llm-d.ai/experiment={self.name}",
                "-o",
                "json",
            ).stdout
        )["items"]
        attached = next(
            (
                p["spec"].get("nodeName")
                for p in reversed(jobs)
                if p["spec"].get("nodeName")
                and any(
                    v.get("persistentVolumeClaim", {}).get("claimName")
                    == f"{self.name}-results"
                    for v in p["spec"].get("volumes", [])
                )
            ),
            None,
        )
        if attached:
            pod["spec"]["nodeName"] = attached
        self.apply(pod)
        try:
            self.k(
                "wait",
                "--for=jsonpath={.status.phase}=Running",
                f"pod/{name}",
                "-n",
                self.ns,
                "--timeout=5m",
            )
            code = "import tarfile,sys; t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|'); t.add('/results',arcname='results'); t.close()"
            argv = (
                ["kubectl"]
                + (["--context", self.context] if self.context else [])
                + ["exec", "-n", self.ns, name, "--", "python3", "-c", code]
            )
            rawdir = self.out / "raw"
            rawdir.mkdir(exist_ok=True)
            with tempfile.TemporaryFile() as stderr:
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr)
                try:
                    with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                        for member in archive:
                            parts = Path(member.name).parts
                            if (
                                not parts
                                or parts[0] != "results"
                                or ".." in parts
                                or member.issym()
                                or member.islnk()
                                or member.name.startswith("/")
                            ):
                                raise ValueError("Unsafe path in results export.")
                            if not member.isfile():
                                continue
                            dest = rawdir.joinpath(*parts[1:])
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            source = archive.extractfile(member)
                            if source is None:
                                raise ValueError(
                                    "Missing file content in results export."
                                )
                            with source, dest.open("wb") as out:
                                shutil.copyfileobj(source, out)
                    if proc.wait(timeout=120):
                        stderr.seek(0)
                        raise RuntimeError(stderr.read().decode())
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
        finally:
            self.k(
                "delete",
                "pod",
                name,
                "-n",
                self.ns,
                "--ignore-not-found",
                allow_failure=True,
            )
        if report:
            self.report()

    def missing_trials(self):
        missing = []
        for t in self.e["plan"]["topologies"]:
            rates = self.state.get(f"rates-tp{t['tp']}", self.e["benchmark"]["rates"])
            for index, _ in enumerate(rates):
                for repetition in range(1, self.e["benchmark"]["repetitions"] + 1):
                    for mode in ("service", "llmd"):
                        key = f"tp{t['tp']}-r{index}-n{repetition}-{mode}"
                        if (
                            self.state["trials"].get(key, {}).get("status")
                            != "completed"
                            or not (self.out / "raw" / f"{key}.json").is_file()
                        ):
                            missing.append(key)
        return missing

    def report(self):
        self.save()
        datasets, runs, artifacts = {}, [], []
        unreadable, collection_warnings = [], []
        for path in (self.out / "raw").glob("rate-*-summary.json"):
            try:
                summary = read_json(path)
                if not isinstance(summary, dict):
                    raise ValueError("Expected a dataset summary object")
                datasets[path.stem.removesuffix("-summary")] = summary
            except (ValueError, OSError) as error:
                collection_warnings.append(
                    f"Unreadable dataset summary {path.name}: {error}"
                )
        for key, context in self.state["trials"].items():
            path = self.out / "raw" / f"{key}.json"
            if not path.exists():
                continue
            dataset = context["dataset"]
            measured = self.out / "raw" / f"{dataset}-measure.jsonl"
            dataset_checksum = checksum(measured) if measured.exists() else None
            try:
                rows = normalize(
                    read_json(path), {**context, "dataset_checksum": dataset_checksum}
                )
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
                unreadable.append(key)
                collection_warnings.append(f"Unreadable trial {path.name}: {error}")
                continue
            for row in rows:
                row["cache_hit_ratio"] = self.cache_ratio(key)
                if context["status"] != "completed":
                    row["warnings"].append(
                        "Partial trial: orchestration did not confirm completion."
                    )
            runs.extend(rows)
            artifacts.append(
                {
                    "path": f"raw/{path.name}",
                    "sha256": checksum(path),
                }
            )
        missing = sorted(set(self.missing_trials()) | set(unreadable))
        data = {
            "report_schema_version": 1,
            "experiment": self.e,
            "generated_at": timestamp(),
            "runs": runs,
            "datasets": datasets,
            "status": "partial" if missing else "complete",
            "missing_trials": missing,
            "execution_target": self.state.get("execution_target"),
            "artifacts": artifacts,
            "warnings": collection_warnings
            + (
                [
                    f"{len(missing)} trial(s) are unfinished or lack usable collected raw results."
                ]
                if missing
                else []
            )
            + [
                "Upstream defaults retained. Inspect errors, incomplete requests and dispatch lag; speedups are not guaranteed.",
                "Runtime evidence contains resolved container imageIDs; source lock pins release references.",
            ],
        }
        write_json(self.out / "report-data.json", data)
        renderer = (self.root / "report.js").read_text()
        (self.out / "report.html").write_text(report_html(data, renderer))
        with zipfile.ZipFile(
            self.out / "results.zip", "w", compression=zipfile.ZIP_STORED
        ) as z:
            for filename in (
                "report-data.json",
                "report.html",
                "experiment.json",
                "state.json",
            ):
                z.write(self.out / filename, filename)
        print(
            f"Report ready: {self.out / 'report.html'}. Import report-data.json into the UI; raw results remain local."
        )

    def cache_ratio(self, key):
        def counters(phase):
            path = self.out / "evidence" / f"{key}-{phase}-metrics.json"
            if not path.exists():
                return None
            try:
                sources = read_json(path)
            except (ValueError, OSError):
                return None
            if (
                not isinstance(sources, dict)
                or not sources
                or any(not isinstance(v, str) for v in sources.values())
            ):
                return None
            pods = {}
            for pod, text in sources.items():
                counts = {"hits": 0.0, "queries": 0.0}
                found = set()
                for line in text.splitlines():
                    match = re.match(
                        r"^vllm:prefix_cache_(hits|queries)(?:_total)?(?:\{[^}]*\})?\s+(\S+)",
                        line,
                    )
                    if match:
                        metric, value = match.groups()
                        try:
                            value = float(value)
                        except ValueError:
                            return None
                        if not math.isfinite(value) or value < 0:
                            return None
                        counts[metric] += value
                        if not math.isfinite(counts[metric]):
                            return None
                        found.add(metric)
                if len(found) != 2:
                    return None
                pods[pod] = counts
            return pods

        before, after = counters("before"), counters("after")
        if before is None or after is None or before.keys() != after.keys():
            return None
        deltas = [
            {key: after[pod][key] - before[pod][key] for key in ("hits", "queries")}
            for pod in before
        ]
        if any(d["hits"] < 0 or d["queries"] < 0 for d in deltas):
            return None
        queries = sum(d["queries"] for d in deltas)
        hits = sum(d["hits"] for d in deltas)
        if not math.isfinite(queries) or not math.isfinite(hits):
            return None
        return hits / queries if queries > 0 and 0 <= hits <= queries else None

    def cleanup(self, delete_results=False):
        self.owned("namespace", self.ns, False)
        # Guard Helm selector before uninstall; never delete a whole namespace.
        releases = json.loads(self.helm("list", "-n", self.ns, "-o", "json").stdout)
        if any(r["name"] == self.release for r in releases):
            values = json.loads(
                self.helm(
                    "get", "values", self.release, "-n", self.ns, "-o", "json"
                ).stdout
            )
            if values.get("router", {}).get("modelServers", {}).get("matchLabels") != {
                "app": f"{self.name}-model"
            }:
                raise ValueError("Helm release is not owned by this recipe.")
            self.helm("uninstall", self.release, "-n", self.ns)
        self.k(
            "delete",
            "deployment,service,configmap,job,pod",
            "-n",
            self.ns,
            "-l",
            f"studio.llm-d.ai/experiment={self.name}",
            "--ignore-not-found",
        )
        if delete_results:
            self.owned("pvc", f"{self.name}-results")
            self.k(
                "delete",
                "pvc",
                f"{self.name}-results",
                "-n",
                self.ns,
                "--ignore-not-found",
            )
        print(
            "Experiment resources removed. Namespace and results PVC retained unless --delete-results was explicitly selected."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment.yaml")
    parser.add_argument("--context")
    parser.add_argument("--output")
    parser.add_argument(
        "action",
        choices=[
            "render",
            "preflight",
            "deploy",
            "run",
            "status",
            "collect",
            "report",
            "cleanup",
        ],
    )
    parser.add_argument("--confirm-context")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delete-results", action="store_true")
    args = parser.parse_args()
    try:
        runner = Runner(args.experiment, args.context, args.output)
        if args.action in ("deploy", "run", "cleanup", "collect"):
            runner.confirm(args.confirm_context)
        if args.action == "render":
            runner.render()
        elif args.action == "preflight":
            runner.preflight()
        elif args.action == "deploy":
            runner.save()
            runner.preflight()
            runner.deploy(runner.e["plan"]["topologies"][0])
        elif args.action == "run":
            runner.run(args.resume)
        elif args.action == "status":
            print(json.dumps(runner.state, indent=2))
        elif args.action == "collect":
            runner.collect()
        elif args.action == "report":
            runner.save()
            runner.report()
        elif args.action == "cleanup":
            runner.cleanup(args.delete_results)
    except (
        ValueError,
        RuntimeError,
        KeyError,
        FileNotFoundError,
        TimeoutError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
