#!/usr/bin/env python3
"""Surveillance & Orchestration Monitor for 65-Band DIAS Suite.

Monitors training and evaluation kernels on Kaggle, stages clean notebooks,
tracks execution states, auto-triggers downstream conformal evaluation jobs,
and syncs completed artifacts back to the local repository.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

USERNAME = "daneshselwal"
DATASET_SOURCE = "daneshselwal/dias-dataset-k01"

KERNELS = [
    {
        "slug": "dias-mambahsi-k16",
        "name": "Model_training_mambahsi.ipynb",
        "type": "training",
        "folder": "mambahsi",
        "enable_gpu": True,
        "dependents": ["dias-mambahsi-uq-k17"],
    },
    {
        "slug": "dias-mambahsi-uq-k17",
        "name": "Model_uncertainty_comparison_mambahsi.ipynb",
        "type": "evaluation",
        "folder": "mambahsi",
        "enable_gpu": False,
        "parent": "dias-mambahsi-k16",
    },
    {
        "slug": "dias-dofa-k18",
        "name": "Model_training.ipynb",
        "type": "training",
        "folder": "dofa",
        "enable_gpu": True,
        "dependents": ["dias-dofa-uq-k19"],
    },
    {
        "slug": "dias-dofa-uq-k19",
        "name": "Model_uncertainty_comparison.ipynb",
        "type": "evaluation",
        "folder": "dofa",
        "enable_gpu": False,
        "parent": "dias-dofa-k18",
    },
    {
        "slug": "dias-georsclip-k20",
        "name": "Model_training.ipynb",
        "type": "training",
        "folder": "georsclip",
        "enable_gpu": True,
        "dependents": ["dias-georsclip-uq-k21"],
    },
    {
        "slug": "dias-georsclip-uq-k21",
        "name": "Model_uncertainty_comparison.ipynb",
        "type": "evaluation",
        "folder": "georsclip",
        "enable_gpu": False,
        "parent": "dias-georsclip-k20",
    },
    {
        "slug": "dias-spatialgcn-k22",
        "name": "Model_training.ipynb",
        "type": "training",
        "folder": "spatialgcn",
        "enable_gpu": True,
        "dependents": ["dias-spatialgcn-uq-k23"],
    },
    {
        "slug": "dias-spatialgcn-uq-k23",
        "name": "Model_uncertainty_comparison.ipynb",
        "type": "evaluation",
        "folder": "spatialgcn",
        "enable_gpu": False,
        "parent": "dias-spatialgcn-k22",
    },
    {
        "slug": "dias-dapm-k24",
        "name": "Model_training_dapm_full.ipynb",
        "type": "training",
        "folder": "dapm",
        "enable_gpu": True,
        "dependents": ["dias-dapm-uq-k25"],
    },
    {
        "slug": "dias-dapm-uq-k25",
        "name": "Model_uncertainty_dapm_full.ipynb",
        "type": "evaluation",
        "folder": "dapm",
        "enable_gpu": False,
        "parent": "dias-dapm-k24",
    },
]

def get_kernel_status(slug: str) -> str:
    """Query Kaggle API for kernel worker status."""
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", f"{USERNAME}/{slug}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = res.stdout.strip()
        if "has status" in out:
            status = out.split('"')[1].replace("KernelWorkerStatus.", "")
            return status
        return out or "UNKNOWN"
    except Exception as e:
        return f"ERR: {e}"

def generate_metadata(kernel_def: dict, repo_root: Path) -> dict:
    """Generate Kaggle metadata JSON dictionary."""
    slug = kernel_def["slug"]
    nb_name = kernel_def["name"]
    enable_gpu = kernel_def["enable_gpu"]

    meta = {
        "id": f"{USERNAME}/{slug}",
        "title": slug,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET_SOURCE],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }

    if "parent" in kernel_def:
        meta["kernel_sources"].append(f"{USERNAME}/{kernel_def['parent']}")

    return meta

def retrigger_kernel(slug: str, repo_root: Path) -> bool:
    """Stage notebook and metadata, then push to Kaggle."""
    kernel_def = next((k for k in KERNELS if k["slug"] == slug), None)
    if not kernel_def:
        print(f"[{slug}] Kernel definition not found.")
        return False

    staging_dir = Path(f"/tmp/dias_staging/{slug}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write kernel metadata
    meta = generate_metadata(kernel_def, repo_root)
    with open(staging_dir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 2. Stage notebook stripped of previous outputs
    src_nb = repo_root / kernel_def["folder"] / kernel_def["name"]
    dest_nb = staging_dir / kernel_def["name"]
    if not src_nb.exists():
        print(f"[{slug}] Source notebook {src_nb} does not exist.")
        return False

    try:
        with open(src_nb, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        with open(dest_nb, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
    except Exception as e:
        shutil.copy2(src_nb, dest_nb)

    # 3. Push to Kaggle
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(staging_dir)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        out = res.stdout.strip() or res.stderr.strip()
        print(f"[{slug}] Kaggle push result: {out}")
        return "successfully pushed" in out.lower()
    except Exception as e:
        print(f"[{slug}] Push exception: {e}")
        return False

def sync_completed_artifacts(slug: str, repo_root: Path) -> bool:
    """Download outputs and sync to local repository."""
    kernel_def = next((k for k in KERNELS if k["slug"] == slug), None)
    if not kernel_def:
        return False

    folder = kernel_def["folder"]
    dest_dir = repo_root / folder / "results"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{slug}] COMPLETE! Downloading outputs to verify and sync...")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            res = subprocess.run(
                ["kaggle", "kernels", "output", f"{USERNAME}/{slug}", "-p", tmpdir],
                capture_output=True,
                text=True,
                timeout=180,
            )
            print(f"[{slug}] Download output: {res.stdout.strip()[:300]}")
            # Find and copy results
            copied = 0
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith((".csv", ".xlsx", ".json", ".png", ".keras")):
                        src_path = Path(root) / f
                        rel_dest = dest_dir / f
                        shutil.copy2(src_path, rel_dest)
                        copied += 1
            print(f"[{slug}] Successfully synced {copied} file(s) -> {dest_dir}")
            return copied > 0
        except Exception as e:
            print(f"[{slug}] Error syncing artifacts: {e}")
            return False

def take_reading(repo_root: Path) -> dict:
    """Take a snapshot reading of all kernels and log."""
    readings = {}
    now_str = datetime.now().isoformat()
    print(f"=== 65-Band DIAS Surveillance Reading at {now_str} ===")
    print(f"{'Kernel Slug':<24} | {'Type':<12} | {'Status':<15} | {'Notebook'}")
    print("-" * 80)
    for k in KERNELS:
        slug = k["slug"]
        status = get_kernel_status(slug)
        readings[slug] = status
        print(f"{slug:<24} | {k['type']:<12} | {status:<15} | {k['name']}")

    # Write log
    log_file = repo_root / "surveillance_log_dias.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": now_str, "readings": readings}) + "\n")

    return readings

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    take_reading(root)
