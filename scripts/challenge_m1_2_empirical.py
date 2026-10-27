#!/usr/bin/env python3
"""
scripts/challenge_m1_2_empirical.py

Adversarial Stress Test Harness for Challenger M1-2:
1. Exhaustive AST parsing of all code cells in DIAS notebooks.
2. Strict detection of unquoted best_model_path, tab20, and stale paths.
3. Sequential model loading memory behavior under Keras 3 / TensorFlow 2.21.
4. Cross-model weight interference and layer registry collision tests.
5. Interleaved repeated inference memory leak detection.
6. Session teardown and reverse-order reload verification.
"""

import os
import sys
import gc
import ast
import json
import math
import subprocess
from pathlib import Path

# Paths
WORKSPACE_ROOT = Path("/Users/danesh/Desktop/Reports_dias")
DIAS_DIR = WORKSPACE_ROOT / "Kaggle_Computed_65Bands_DIAS"
MODELS_DIR = DIAS_DIR / "multicp" / "models"
SCRIPTS_DIR = DIAS_DIR / "scripts"

sys.path.insert(0, str(DIAS_DIR))
from scripts.verify_models_smoke import (
    CUSTOM_OBJECTS,
    CLASS_COLOR_BASE,
    BACKGROUND_COLOR,
    UNCERTAIN_COLOR,
    CERTAIN_COLOR,
    NUM_CLASSES,
    K_HEADS,
    INPUT_SHAPE,
)

import numpy as np
import tensorflow as tf
from tensorflow import keras

TARGET_NOTEBOOKS = [
    DIAS_DIR / "multicp" / "Model_uncertainty_multicp.ipynb",
    DIAS_DIR / "multicp_sacp" / "Model_uncertainty_multicp_sacp.ipynb",
    DIAS_DIR / "multicp" / "Model_training_multihead.ipynb",
]

TARGET_MODELS = [
    ("AlexNet_CNN_MultiHead", MODELS_DIR / "AlexNet_CNN_MultiHead_best.keras"),
    ("GFNet_MultiHead", MODELS_DIR / "GFNet_MultiHead_best.keras"),
    ("ViT_UNet_MultiHead", MODELS_DIR / "ViT_UNet_MultiHead_best.keras"),
]


def get_process_rss_mb():
    """Get current process RSS in MB on macOS via ps."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).decode().strip()
        return float(out) / 1024.0
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# ==============================================================================
# Suite 1: Adversarial AST Parsing & Static Code Analysis
# ==============================================================================

def run_ast_and_static_challenge():
    print("\n" + "=" * 70)
    print("CHALLENGE SUITE 1: Adversarial AST Parsing & Static Code Audit")
    print("=" * 70)

    results = {
        "notebooks_audited": len(TARGET_NOTEBOOKS),
        "total_cells": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "ast_nodes_inspected": 0,
        "syntax_errors": [],
        "unquoted_best_model_path_flags": [],
        "tab20_occurrences": [],
        "stale_path_flags": [],
        "layer_signature_issues": [],
    }

    for nb_path in TARGET_NOTEBOOKS:
        print(f"\n--- Auditing: {nb_path.name} ---")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        results["total_cells"] += len(cells)
        nb_code_cells = 0

        for cell_idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type")
            raw_source = "".join(cell.get("source", []))

            if cell_type == "markdown":
                results["markdown_cells"] += 1
                continue

            if cell_type != "code":
                continue

            nb_code_cells += 1
            results["code_cells"] += 1

            # 1. Direct text scans
            if "tab20" in raw_source:
                results["tab20_occurrences"].append({
                    "notebook": nb_path.name,
                    "cell_idx": cell_idx,
                    "snippet": [line for line in raw_source.splitlines() if "tab20" in line]
                })

            # Check unquoted best_model_path
            for line_no, line in enumerate(raw_source.splitlines(), start=1):
                if "{best_model_path:" in line or "[best_model_path]" in line or "info[best_model_path]" in line:
                    results["unquoted_best_model_path_flags"].append({
                        "notebook": nb_path.name,
                        "cell_idx": cell_idx,
                        "line_no": line_no,
                        "line": line.strip()
                    })

                # Check stale /content/drive paths that aren't inside Colab guards
                if "/content/drive" in line and not line.strip().startswith("#"):
                    # Check if cell has colab guard
                    if "google.colab" not in raw_source and "IN_COLAB" not in raw_source:
                        results["stale_path_flags"].append({
                            "notebook": nb_path.name,
                            "cell_idx": cell_idx,
                            "line_no": line_no,
                            "line": line.strip(),
                            "reason": "Unguarded /content/drive path"
                        })

                # Check hardcoded /kaggle/working/results/models without fallback
                if "/kaggle/working/results/models" in line and not line.strip().startswith("#"):
                    if "resolve_multihead_model_file" not in raw_source and "exists" not in raw_source:
                        results["stale_path_flags"].append({
                            "notebook": nb_path.name,
                            "cell_idx": cell_idx,
                            "line_no": line_no,
                            "line": line.strip(),
                            "reason": "Hardcoded /kaggle/working/results/models without fallback"
                        })

            # 2. IPython syntax cleaning for pure Python AST parsing
            clean_lines = []
            for line in raw_source.splitlines():
                stripped = line.strip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    clean_lines.append(f"# IPYTHON_MAGIC: {line}")
                elif stripped.startswith("?") or stripped.endswith("?"):
                    clean_lines.append(f"# IPYTHON_HELP: {line}")
                else:
                    clean_lines.append(line)
            code_str = "\n".join(clean_lines)

            # 3. AST Parsing
            try:
                tree = ast.parse(code_str, filename=f"{nb_path.name}[cell_{cell_idx}]")
            except SyntaxError as e:
                results["syntax_errors"].append({
                    "notebook": nb_path.name,
                    "cell_idx": cell_idx,
                    "error": str(e),
                    "lineno": e.lineno,
                    "offset": e.offset,
                })
                continue

            # 4. AST Node Walking
            for node in ast.walk(tree):
                results["ast_nodes_inspected"] += 1

                # Detect ast.Name with id='best_model_path' in load context
                if isinstance(node, ast.Name) and node.id == "best_model_path" and isinstance(node.ctx, ast.Load):
                    results["unquoted_best_model_path_flags"].append({
                        "notebook": nb_path.name,
                        "cell_idx": cell_idx,
                        "ast_node": "ast.Name(id='best_model_path', ctx=Load())",
                        "lineno": getattr(node, "lineno", None),
                    })

                # Check class definitions for ViT_Weighted_add and GF_Expand_Dims in uncertainty notebooks
                if nb_path.name in ("Model_uncertainty_multicp.ipynb", "Model_uncertainty_multicp_sacp.ipynb") and isinstance(node, ast.ClassDef):
                    if node.name == "ViT_Weighted_add":
                        init_fn = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
                        if init_fn:
                            arg_names = [a.arg for a in init_fn.args.args]
                            # Expect self, wt_name=None, name=None or similar kwargs
                            has_wt_name = "wt_name" in arg_names
                            has_name = "name" in arg_names
                            has_kwargs = init_fn.args.kwarg is not None
                            print(f"  [{nb_path.name} Cell {cell_idx}] Found ViT_Weighted_add.__init__ args: {arg_names}, kwarg: {init_fn.args.kwarg.arg if init_fn.args.kwarg else None}")
                            if not (has_wt_name and has_kwargs):
                                results["layer_signature_issues"].append({
                                    "notebook": nb_path.name,
                                    "cell_idx": cell_idx,
                                    "class": "ViT_Weighted_add",
                                    "args": arg_names,
                                    "issue": "Missing wt_name or **kwargs in __init__"
                                })

                    elif node.name == "GF_Expand_Dims":
                        init_fn = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
                        if init_fn:
                            arg_names = [a.arg for a in init_fn.args.args]
                            # Check default value for ndim
                            has_defaults = len(init_fn.args.defaults) > 0
                            print(f"  [{nb_path.name} Cell {cell_idx}] Found GF_Expand_Dims.__init__ args: {arg_names}, defaults: {len(init_fn.args.defaults)}")
                            if "ndim" in arg_names and not has_defaults:
                                results["layer_signature_issues"].append({
                                    "notebook": nb_path.name,
                                    "cell_idx": cell_idx,
                                    "class": "GF_Expand_Dims",
                                    "issue": "ndim parameter lacks default value"
                                })

        print(f"  Code cells parsed: {nb_code_cells}")

    print("\n--- AST & Static Audit Summary ---")
    print(f"Total cells audited       : {results['total_cells']} ({results['code_cells']} code, {results['markdown_cells']} markdown)")
    print(f"AST nodes inspected       : {results['ast_nodes_inspected']}")
    print(f"Syntax errors             : {len(results['syntax_errors'])}")
    print(f"Unquoted best_model_path  : {len(results['unquoted_best_model_path_flags'])}")
    print(f"tab20 occurrences         : {len(results['tab20_occurrences'])}")
    print(f"Stale path flags          : {len(results['stale_path_flags'])}")
    print(f"Layer signature defects   : {len(results['layer_signature_issues'])}")

    assert len(results["syntax_errors"]) == 0, f"Syntax errors found: {results['syntax_errors']}"
    assert len(results["unquoted_best_model_path_flags"]) == 0, f"Unquoted best_model_path found: {results['unquoted_best_model_path_flags']}"
    assert len(results["tab20_occurrences"]) == 0, f"tab20 occurrences found: {results['tab20_occurrences']}"
    assert len(results["stale_path_flags"]) == 0, f"Stale path defects found: {results['stale_path_flags']}"
    assert len(results["layer_signature_issues"]) == 0, f"Layer signature defects in uncertainty notebooks: {results['layer_signature_issues']}"


    print("[PASS] Suite 1 passed: Zero syntax errors, zero unquoted variables, zero tab20, zero path defects.")
    return results


# ==============================================================================
# Suite 2: Sequential Model Loading, Memory Footprint & Layer Registry Stress
# ==============================================================================

def run_sequential_memory_stress():
    print("\n" + "=" * 70)
    print("CHALLENGE SUITE 2: Sequential Model Loading & Memory Footprint Stress")
    print("=" * 70)

    results = {
        "initial_rss_mb": 0.0,
        "stages": {},
        "cross_talk_diffs": {},
        "interleaved_memory": [],
        "gc_rss_mb": 0.0,
        "reverse_load_ok": False,
    }

    # Initial RSS before any model loading
    initial_rss = get_process_rss_mb()
    results["initial_rss_mb"] = initial_rss
    print(f"Initial Process RSS: {initial_rss:.2f} MB")

    dummy_input = np.random.uniform(0.0, 1.0, INPUT_SHAPE).astype(np.float32)
    loaded_models = {}
    baseline_predictions = {}

    # Sequential loading: AlexNet -> GFNet -> ViT_UNet
    prev_rss = initial_rss
    for model_name, model_path in TARGET_MODELS:
        assert model_path.exists(), f"Model file not found: {model_path}"
        print(f"\n---> Loading [{model_name}] from {model_path.name} (file size: {model_path.stat().st_size / (1024*1024):.2f} MB)...")
        
        m = keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )
        loaded_models[model_name] = m
        current_rss = get_process_rss_mb()
        delta_rss = current_rss - prev_rss
        total_delta = current_rss - initial_rss
        prev_rss = current_rss

        print(f"     Loaded OK. Process RSS: {current_rss:.2f} MB (Delta: +{delta_rss:.2f} MB | Cumulative: +{total_delta:.2f} MB)")

        # Run forward prediction on dummy tensor
        raw_out = m.predict(dummy_input, verbose=0)
        heads = list(raw_out) if isinstance(raw_out, (list, tuple)) else [raw_out]
        
        assert len(heads) == K_HEADS, f"Expected {K_HEADS} output heads, got {len(heads)}"
        for h_idx, h in enumerate(heads):
            assert h.shape == (1, NUM_CLASSES), f"Head {h_idx} shape {h.shape} != (1, {NUM_CLASSES})"
            assert np.all(h >= 0.0) and np.all(h <= 1.0 + 1e-6), f"Head {h_idx} values outside [0, 1]"
            assert np.isclose(np.sum(h), 1.0, atol=1e-5), f"Head {h_idx} does not sum to 1.0 (sum={np.sum(h)})"

        baseline_predictions[model_name] = [np.copy(h) for h in heads]
        results["stages"][model_name] = {
            "rss_mb": current_rss,
            "delta_mb": delta_rss,
            "num_heads": len(heads),
            "output_shape": str(heads[0].shape),
            "sum_head1": float(np.sum(heads[0])),
        }

    # Verify All 3 models are co-resident in memory simultaneously
    print(f"\nAll 3 models co-resident in memory simultaneously!")
    print(f"Total Process RSS: {get_process_rss_mb():.2f} MB (Net increase from initial: +{get_process_rss_mb() - initial_rss:.2f} MB)")
    assert len(loaded_models) == 3, f"Expected 3 models loaded, got {len(loaded_models)}"

    # Cross-Model Isolation / Cross-Talk Test
    # Re-predict AlexNet and GFNet now that ViT-UNet is also loaded
    print("\n--- Testing Cross-Model Prediction Isolation ---")
    for model_name in ["AlexNet_CNN_MultiHead", "GFNet_MultiHead"]:
        m = loaded_models[model_name]
        re_out = m.predict(dummy_input, verbose=0)
        re_heads = list(re_out) if isinstance(re_out, (list, tuple)) else [re_out]
        orig_heads = baseline_predictions[model_name]

        max_diff = 0.0
        for h_idx in range(K_HEADS):
            diff = np.max(np.abs(re_heads[h_idx] - orig_heads[h_idx]))
            if diff > max_diff:
                max_diff = diff

        print(f"  {model_name}: Max discrepancy between baseline and post-co-load prediction = {max_diff:.2e}")
        assert max_diff < 1e-6, f"Cross-model weight contamination detected in {model_name}! Max diff={max_diff}"
        results["cross_talk_diffs"][model_name] = float(max_diff)

    # Interleaved Inference Stress Test (Check for memory leak during repeated inference)
    print("\n--- Testing Interleaved Repeated Forward Inference (Memory Leak Check) ---")
    checkpoints = [1, 5, 10, 20, 30]
    for iter_i in range(1, 31):
        for m_name in ["AlexNet_CNN_MultiHead", "GFNet_MultiHead", "ViT_UNet_MultiHead"]:
            _ = loaded_models[m_name].predict(dummy_input, verbose=0)

        if iter_i in checkpoints:
            rss = get_process_rss_mb()
            print(f"  Iteration {iter_i:02d}: Process RSS = {rss:.2f} MB")
            results["interleaved_memory"].append({"iteration": iter_i, "rss_mb": rss})

    rss_iter5 = results["interleaved_memory"][1]["rss_mb"]
    rss_iter30 = results["interleaved_memory"][-1]["rss_mb"]
    leak_delta = rss_iter30 - rss_iter5
    print(f"Memory growth between iter 5 and iter 30: {leak_delta:+.2f} MB")
    assert leak_delta < 20.0, f"Memory leak detected during repeated inference: grew by {leak_delta:.2f} MB"

    # Session Cleanup & Reverse-Order Reload
    print("\n--- Testing Session Teardown & Reverse-Order Reload ---")
    del loaded_models
    keras.backend.clear_session()
    gc.collect()
    gc_rss = get_process_rss_mb()
    results["gc_rss_mb"] = gc_rss
    print(f"Process RSS after clear_session + gc.collect(): {gc_rss:.2f} MB")

    print("\nRe-loading in REVERSE order: ViT_UNet -> GFNet -> AlexNet ...")
    reverse_models = {}
    for model_name, model_path in reversed(TARGET_MODELS):
        m = keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )
        reverse_models[model_name] = m
        rev_out = m.predict(dummy_input, verbose=0)
        rev_heads = list(rev_out) if isinstance(rev_out, (list, tuple)) else [rev_out]
        orig_heads = baseline_predictions[model_name]
        max_diff = max(np.max(np.abs(rev_heads[i] - orig_heads[i])) for i in range(K_HEADS))
        print(f"  Reverse loaded {model_name}: OK (prediction match max_diff = {max_diff:.2e})")
        assert max_diff < 1e-6, f"Reverse-order load produced differing predictions for {model_name}: {max_diff}"

    results["reverse_load_ok"] = True
    print("\n[PASS] Suite 2 passed: All 3 models loaded sequentially and in reverse, co-resident without memory corruption or cross-talk.")
    return results


# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    print("=" * 80)
    print("EMPIRICAL CHALLENGER M1-2: ADVERSARIAL STRESS TEST HARNESS")
    print(f"Python: {sys.version.split()[0]} | TensorFlow: {tf.__version__} | Keras: {keras.__version__}")
    print("=" * 80)

    # 1. AST Parsing & Static Checks
    ast_results = run_ast_and_static_challenge()

    # 2. Sequential Model Loading & Memory Checks
    mem_results = run_sequential_memory_stress()

    # Save summary report artifact
    report = {
        "verdict": "APPROVE",
        "ast_audit": ast_results,
        "memory_stress": mem_results,
    }
    
    out_json = DIAS_DIR / "scripts" / "challenge_m1_2_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote empirical results to: {out_json}")

    print("\n" + "=" * 80)
    print("FINAL CHALLENGE VERDICT: APPROVE")
    print("All empirical tests passed with zero errors or deviations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
