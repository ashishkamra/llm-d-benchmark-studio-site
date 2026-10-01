#!/usr/bin/env python3
"""Read-only node/pod discovery; prints inventory JSON. Never sends credentials."""

import argparse
import datetime
import json
import subprocess


def inventory(nodes, pods):
    def gpu_request(container):
        resources = container.get("resources", {})
        return int(
            resources.get("requests", {}).get(
                "nvidia.com/gpu", resources.get("limits", {}).get("nvidia.com/gpu", 0)
            )
        )

    used = {}
    for pod in pods.get("items", []):
        if pod.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        spec = pod.get("spec", {})
        regular = sum(gpu_request(c) for c in spec.get("containers", []))
        sidecars, init_peak = 0, 0
        for container in spec.get("initContainers", []):
            request = gpu_request(container)
            # Restartable init containers remain active alongside subsequent
            # init containers and the application, as in Kubernetes PodRequests.
            if container.get("restartPolicy") == "Always":
                sidecars += request
                active = sidecars
            else:
                active = sidecars + request
            init_peak = max(init_peak, active)
        node = spec.get("nodeName")
        used[node] = used.get(node, 0) + max(regular + sidecars, init_peak)
    pools = {}
    for node in nodes.get("items", []):
        spec, status = node.get("spec", {}), node.get("status", {})
        if spec.get("unschedulable") or not any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in status.get("conditions", [])
        ):
            continue
        labels = node["metadata"].get("labels", {})
        product = labels.get("nvidia.com/gpu.product", "unknown")
        memory = int(labels.get("nvidia.com/gpu.memory", 0)) // 1024
        sharing = (
            "shared"
            if int(labels.get("nvidia.com/gpu.replicas", 1)) > 1
            or product.endswith("-SHARED")
            else ("mig" if "MIG" in product else "exclusive")
        )
        available = max(
            0,
            int(status.get("allocatable", {}).get("nvidia.com/gpu", 0))
            - used.get(node["metadata"]["name"], 0),
        )
        if not available:
            continue
        key = (product, memory, sharing)
        p = pools.setdefault(
            key,
            {
                "product": product,
                "memory_gib": memory,
                "sharing": sharing,
                "node_slots": [],
                "node_names": [],
                "selector": {"nvidia.com/gpu.product": product}
                if product != "unknown"
                else {},
            },
        )
        p["node_slots"].append(available)
        p["node_names"].append(node["metadata"]["name"])
    return {
        "inventory_schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pools": list(pools.values()),
        "warnings": [
            "Allocatable minus pod GPU requests is a point-in-time estimate, not a reservation. Verify memory, taints and access mode before executing.",
            "Unknown GPU labels require manual configuration. MIG mixed-resource pools are not automatically planned.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", help="Explicit kubeconfig context; otherwise use current context"
    )
    args = parser.parse_args()
    command = ["kubectl"] + (["--context", args.context] if args.context else [])

    def get(resource):
        return json.loads(
            subprocess.check_output(
                command + ["get", resource, "-A", "-o", "json"], text=True
            )
        )

    print(json.dumps(inventory(get("nodes"), get("pods")), indent=2))


if __name__ == "__main__":
    main()
