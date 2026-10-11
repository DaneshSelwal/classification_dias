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
        "enable_gpu": True,
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
        "enable_gpu": True,
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
        "enable_gpu": True,
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
        "enable_gpu": True,
        "parent": "dias-spatialgcn-k22",
    },
    {
        "slug": "dias-dapm-k24",
        "name": "Model_training_dapm_full.ipynb",
        "type": "training",
        "folder": "dapm",
        "enable_gpu": True,
        "kernel_sources": ["daneshselwal/dias-baseline-k03"],
        "dependents": ["dias-dapm-uq-k25"],
    },
    {
        "slug": "dias-dapm-uq-k25",
        "name": "Model_uncertainty_dapm_full.ipynb",
        "type": "evaluation",
        "folder": "dapm",
        "enable_gpu": True,
        "kernel_sources": ["daneshselwal/dias-baseline-k03"],
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

    if "kernel_sources" in kernel_def:
        for ks in kernel_def["kernel_sources"]:
            if ks not in meta["kernel_sources"]:
                meta["kernel_sources"].append(ks)

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

SYNC_TARGETS = {
    "dias-spatialgcn-k22": {
        "models_dir": "spatialgcn/models",
        "results_dir": "spatialgcn/results",
        "primary_file": "SpatialGCN_best.keras",
    },
    "dias-spatialgcn-uq-k23": {
        "results_dir": "spatialgcn/results/uncertainty_results",
        "primary_file": "conformal_reports_all_models.xlsx",
    },
    "dias-mambahsi-k16": {
        "models_dir": "mambahsi/models",
        "results_dir": "mambahsi/results",
        "primary_file": "MambaHSI_Base_best.keras",
    },
    "dias-mambahsi-uq-k17": {
        "results_dir": "mambahsi/results/uncertainty_results",
        "primary_file": "conformal_reports_all_models.xlsx",
    },
    "dias-dofa-k18": {
        "models_dir": "dofa/models",
        "results_dir": "dofa/results",
        "primary_file": "DOFA_Hiera_Fusion_best.keras",
    },
    "dias-dofa-uq-k19": {
        "results_dir": "dofa/results/uncertainty_results",
        "primary_file": "conformal_reports_all_models.xlsx",
    },
    "dias-georsclip-k20": {
        "models_dir": "georsclip/models",
        "results_dir": "georsclip/results",
        "primary_file": "GeoRSCLIP_ViT_best.keras",
    },
    "dias-georsclip-uq-k21": {
        "results_dir": "georsclip/results/uncertainty_results",
        "primary_file": "conformal_reports_all_models.xlsx",
    },
    "dias-dapm-k24": {
        "models_dir": "dapm/models",
        "results_dir": "dapm/results",
        "primary_file": "dapm_full_training_summary.csv",
    },
    "dias-dapm-uq-k25": {
        "results_dir": "dapm/results",
        "primary_file": "dapm_full_summary_metrics.csv",
    },
}

def sync_completed_artifacts(slug: str, repo_root: Path) -> bool:
    """Download outputs and sync to local repository."""
    kernel_def = next((k for k in KERNELS if k["slug"] == slug), None)
    if not kernel_def:
        return False

    folder = kernel_def["folder"]
    target_info = SYNC_TARGETS.get(slug, {})
    results_dir = repo_root / target_info.get("results_dir", f"{folder}/results")
    models_dir = repo_root / target_info.get("models_dir", f"{folder}/models")
    results_dir.mkdir(parents=True, exist_ok=True)
    if "models_dir" in target_info:
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{slug}] COMPLETE! Downloading outputs to verify and sync...")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            res = subprocess.run(
                ["kaggle", "kernels", "output", f"{USERNAME}/{slug}", "-p", tmpdir],
                capture_output=True,
                text=True,
                timeout=600,
            )
            print(f"[{slug}] Download output: {res.stdout.strip()[:300]}")
            copied = 0
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith((".csv", ".xlsx", ".json", ".png")):
                        src_path = Path(root) / f
                        shutil.copy2(src_path, results_dir / f)
                        copied += 1
                    elif f.endswith((".keras", ".h5")):
                        src_path = Path(root) / f
                        shutil.copy2(src_path, models_dir / f)
                        copied += 1
            print(f"[{slug}] Successfully synced {copied} file(s) -> {results_dir}")
            return copied > 0
        except Exception as e:
            print(f"[{slug}] Error syncing artifacts: {e}")
            return False

def git_sync_push(repo_root: Path):
    """Automatically commit and push newly synced artifacts with strict attribution."""
    try:
        env = os.environ.copy()
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"
        env["GIT_CONFIG_SYSTEM"] = "/dev/null"
        subprocess.run(["git", "add", "."], cwd=repo_root, env=env, check=True)
        commit_res = subprocess.run(
            ["git", "commit", "-m", "feat(dias): sync fresh 65-band evaluation and model artifacts\n\nCo-authored-by: Claude <claude@anthropic.com>",
             "--author=Danesh Selwal <selwaldanesh@gmail.com>"],
            cwd=repo_root, env=env, capture_output=True, text=True
        )
        if commit_res.returncode == 0:
            print("Git commit successful:", commit_res.stdout.strip())
            push_res = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=repo_root, env=env, capture_output=True, text=True
            )
            print("Git push result:", push_res.stdout.strip() or push_res.stderr.strip())
    except Exception as e:
        print(f"Git sync push exception: {e}")

def check_and_auto_trigger(readings: dict, repo_root: Path) -> list:
    """Auto-trigger downstream evaluation and launch next training as GPU slots open."""
    state_file = repo_root / "scripts" / ".trigger_state.json"
    state = {}
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except Exception:
            state = {}

    triggered = []
    synced_any = False

    # 1. Sync completed kernels
    for k in KERNELS:
        slug = k["slug"]
        status = readings.get(slug, "UNKNOWN")

        if status == "COMPLETE":
            if slug not in state or not state[slug].get("synced", False):
                success = sync_completed_artifacts(slug, repo_root)
                state.setdefault(slug, {})["synced"] = success
                if success:
                    synced_any = True

    # 2. Check active GPU sessions (RUNNING or QUEUED)
    active_gpu_kernels = [
        k["slug"] for k in KERNELS
        if k["enable_gpu"] and readings.get(k["slug"]) in ("RUNNING", "QUEUED")
    ]

    # 3. PRIORITY 1: Trigger or retrigger evaluation kernels for COMPLETED parent models
    for k in KERNELS:
        if len(active_gpu_kernels) >= 2:
            break
        if k["type"] == "evaluation":
            slug = k["slug"]
            parent_slug = k.get("parent")
            parent_status = readings.get(parent_slug, "UNKNOWN")
            cur_status = readings.get(slug, "UNKNOWN")

            if parent_status == "COMPLETE" and cur_status not in ("COMPLETE", "RUNNING", "QUEUED"):
                last_trig = state.get(slug, {}).get("last_triggered_ts", 0)
                # Allow retriggering if not triggered in the last 120s
                if time.time() - last_trig > 120:
                    print(f"[{slug}] Parent {parent_slug} COMPLETE, status is {cur_status}. Triggering evaluation...")
                    if retrigger_kernel(slug, repo_root):
                        state.setdefault(slug, {})["triggered"] = True
                        state[slug]["last_triggered_ts"] = time.time()
                        triggered.append(slug)
                        active_gpu_kernels.append(slug)

    # 4. PRIORITY 2: If GPU slots still open (< 2), launch next pending training kernel
    if len(active_gpu_kernels) < 2:
        training_order = [
            "dias-mambahsi-k16", "dias-spatialgcn-k22",
            "dias-dofa-k18", "dias-georsclip-k20", "dias-dapm-k24"
        ]
        for t_slug in training_order:
            if len(active_gpu_kernels) >= 2:
                break
            t_status = readings.get(t_slug, "UNKNOWN")
            if t_status not in ("COMPLETE", "RUNNING", "QUEUED") and not state.get(t_slug, {}).get("started", False):
                print(f"GPU slot available ({len(active_gpu_kernels)} active). Launching next: {t_slug}")
                if retrigger_kernel(t_slug, repo_root):
                    state.setdefault(t_slug, {})["started"] = True
                    triggered.append(t_slug)
                    active_gpu_kernels.append(t_slug)

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    if synced_any:
        git_sync_push(repo_root)

    return triggered

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

    triggered = check_and_auto_trigger(readings, repo_root)

    # Write log
    log_file = repo_root / "surveillance_log_dias.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": now_str, "readings": readings, "triggered": triggered}) + "\n")

    if triggered:
        print(f"Auto-triggered kernels: {triggered}")

    return readings

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    take_reading(root)
