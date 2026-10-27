#!/usr/bin/env python3
"""
scripts/adversarial_stress_test.py

Adversarial Stress Testing Harness for 65-Band DIAS Multi-Head Architectures:
1. Adversarial layer constructor stress tests (ViT_Weighted_add, GF_Expand_Dims, etc.)
2. Config round-trip fidelity tests
3. Dynamic forward inference stress tests across varied batch sizes (B = 1, 2, 4, 8, 16)
4. Input distribution stress tests (zeros, ones, normal, uniform, extremes)
5. Probability simplex invariants (sum == 1.0, non-negative, <= 1.0, shape (B, 8) across 7 heads)
6. Batch sample independence invariant (sample prediction invariant to batch size)
"""

import sys
import json
import math
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add scripts directory to path to import CUSTOM_OBJECTS from verify_models_smoke
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_models_smoke import (
    CUSTOM_OBJECTS,
    ViT_Weighted_add,
    GF_Expand_Dims,
    GF_Block,
    GF_MLP,
    GF_DropPath,
    GF_Patches,
    GF_PatchEncoder,
    GF_GlobalFilter,
    ViT_Patches,
    ViT_PatchEncoder,
    ViT_SpatialAttention,
    ViT_SpatialAttention1,
    ViT_TransFormer,
    ViT_TransFormer_Block,
    ViT_Class_Token_Norm,
    Pearson_correlation_masked,
    Dropout_Train,
    MODELS_DIR,
    TARGET_MODELS,
    NUM_CLASSES,
    K_HEADS,
)


def run_vit_weighted_add_challenges():
    print("\n" + "=" * 60)
    print("CHALLENGE 1: ViT_Weighted_add Constructor & Invocation Stress Tests")
    print("=" * 60)
    
    results = []

    # Scenario 1: Positional argument
    try:
        layer1 = ViT_Weighted_add("transformer_1_0")
        assert layer1.wt_name == "transformer_1_0", f"wt_name mismatch: {layer1.wt_name}"
        layer1.build((1, 10))
        out1 = layer1(tf.ones((2, 10)), tf.zeros((2, 10))).numpy()
        assert out1.shape == (2, 10), f"Unexpected shape {out1.shape}"
        results.append(("Positional argument ('transformer_1_0')", True, "Success"))
    except Exception as e:
        results.append(("Positional argument ('transformer_1_0')", False, str(e)))

    # Scenario 2: Keyword argument wt_name
    try:
        layer2 = ViT_Weighted_add(wt_name="transformer_2_1")
        assert layer2.wt_name == "transformer_2_1"
        layer2.build((1, 10))
        results.append(("Keyword argument (wt_name='transformer_2_1')", True, "Success"))
    except Exception as e:
        results.append(("Keyword argument (wt_name='transformer_2_1')", False, str(e)))

    # Scenario 3: Keyword argument name only (Keras internal style)
    try:
        layer3 = ViT_Weighted_add(name="vi_t__weighted_add")
        assert layer3.wt_name == "vi_t__weighted_add"
        assert layer3.name == "vi_t__weighted_add"
        layer3.build((1, 10))
        results.append(("Keyword argument (name='vi_t__weighted_add')", True, "Success"))
    except Exception as e:
        results.append(("Keyword argument (name='vi_t__weighted_add')", False, str(e)))

    # Scenario 4: Both name and wt_name specified (exact deserialization signature)
    try:
        layer4 = ViT_Weighted_add(name="vi_t__weighted_add_custom", wt_name="transformer_1_0")
        assert layer4.name == "vi_t__weighted_add_custom"
        assert layer4.wt_name == "transformer_1_0"
        layer4.build((1, 10))
        results.append(("Both name and wt_name specified", True, "Success"))
    except Exception as e:
        results.append(("Both name and wt_name specified", False, str(e)))

    # Scenario 5: Default / No arguments
    try:
        layer5 = ViT_Weighted_add()
        layer5.build((1, 10))
        out5 = layer5(tf.ones((1, 5)), tf.zeros((1, 5))).numpy()
        assert out5.shape == (1, 5)
        results.append(("Default / zero arguments", True, "Success"))
    except Exception as e:
        results.append(("Default / zero arguments", False, str(e)))

    # Scenario 6: Unusual names (Unicode, special chars, numbers)
    unusual_names = ["🔥_layer", "!@#$%-+=~", "12345", "layer with spaces"]
    for u_name in unusual_names:
        try:
            layer_u = ViT_Weighted_add(wt_name=u_name)
            layer_u.build((1, 5))
            results.append((f"Unusual wt_name: '{u_name}'", True, "Success"))
        except Exception as e:
            results.append((f"Unusual wt_name: '{u_name}'", False, str(e)))

    # Scenario 7: Extra layer kwargs (trainable=False, dtype='float32')
    try:
        layer7 = ViT_Weighted_add(wt_name="frozen_layer", trainable=False, dtype="float32")
        layer7.build((1, 5))
        assert not layer7.trainable
        assert len(layer7.trainable_weights) == 0
        assert len(layer7.non_trainable_weights) == 1
        results.append(("Pass-through kwargs (trainable=False, dtype='float32')", True, "Success"))
    except Exception as e:
        results.append(("Pass-through kwargs (trainable=False, dtype='float32')", False, str(e)))

    # Scenario 8: Config serialization and from_config roundtrip
    try:
        layer8 = ViT_Weighted_add(wt_name="roundtrip_test", name="my_layer")
        cfg8 = layer8.get_config()
        reconstructed8 = ViT_Weighted_add.from_config(cfg8)
        assert reconstructed8.wt_name == "roundtrip_test"
        assert reconstructed8.name == "my_layer"
        results.append(("get_config / from_config round-trip", True, "Success"))
    except Exception as e:
        results.append(("get_config / from_config round-trip", False, str(e)))

    # Scenario 9: Mathematical correctness test: w*a + (1-w)*b
    try:
        layer9 = ViT_Weighted_add(wt_name="math_test")
        layer9.build((1, 4))
        # Set weight w to 0.75
        layer9.w.assign([0.75])
        a = tf.constant([[2.0, 4.0, 6.0, 8.0]])
        b = tf.constant([[10.0, 20.0, 30.0, 40.0]])
        # expected: 0.75 * a + 0.25 * b
        expected = 0.75 * a.numpy() + 0.25 * b.numpy()
        actual = layer9(a, b).numpy()
        assert np.allclose(actual, expected, atol=1e-6), f"Math mismatch: actual={actual}, expected={expected}"
        results.append(("Mathematical formula invariant w*a + (1-w)*b", True, "Success"))
    except Exception as e:
        results.append(("Mathematical formula invariant w*a + (1-w)*b", False, str(e)))

    # Print summary
    all_passed = True
    for desc, passed, msg in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {desc}: {msg}")
        if not passed:
            all_passed = False

    return all_passed, results


def run_gf_expand_dims_challenges():
    print("\n" + "=" * 60)
    print("CHALLENGE 2: GF_Expand_Dims Constructor & Invocation Stress Tests")
    print("=" * 60)
    
    results = []

    # Scenario 1: Default constructor without arguments
    try:
        layer1 = GF_Expand_Dims()
        assert layer1.ndim == 2, f"Expected default ndim=2, got {layer1.ndim}"
        x = tf.ones((4, 5))
        out1 = layer1(x).numpy()
        assert out1.shape == (4, 5, 1), f"Expected shape (4, 5, 1), got {out1.shape}"
        results.append(("Default / zero arguments (ndim=2)", True, "Success"))
    except Exception as e:
        results.append(("Default / zero arguments (ndim=2)", False, str(e)))

    # Scenario 2: Positional argument
    try:
        layer2 = GF_Expand_Dims(1)
        assert layer2.ndim == 1, f"Expected ndim=1, got {layer2.ndim}"
        x = tf.ones((4, 5))
        out2 = layer2(x).numpy()
        assert out2.shape == (4, 1, 5), f"Expected shape (4, 1, 5), got {out2.shape}"
        results.append(("Positional argument (ndim=1)", True, "Success"))
    except Exception as e:
        results.append(("Positional argument (ndim=1)", False, str(e)))

    # Scenario 3: Keyword argument
    try:
        layer3 = GF_Expand_Dims(ndim=3)
        assert layer3.ndim == 3, f"Expected ndim=3, got {layer3.ndim}"
        x = tf.ones((2, 3, 4))
        out3 = layer3(x).numpy()
        assert out3.shape == (2, 3, 4, 1), f"Expected shape (2, 3, 4, 1), got {out3.shape}"
        results.append(("Keyword argument (ndim=3)", True, "Success"))
    except Exception as e:
        results.append(("Keyword argument (ndim=3)", False, str(e)))

    # Scenario 4: Negative axis
    try:
        layer4 = GF_Expand_Dims(ndim=-1)
        x = tf.ones((2, 3))
        out4 = layer4(x).numpy()
        assert out4.shape == (2, 3, 1), f"Expected shape (2, 3, 1), got {out4.shape}"
        results.append(("Negative axis (ndim=-1)", True, "Success"))
    except Exception as e:
        results.append(("Negative axis (ndim=-1)", False, str(e)))

    # Scenario 5: Pass-through kwargs (name, trainable)
    try:
        layer5 = GF_Expand_Dims(ndim=2, name="custom_expand", trainable=False)
        assert layer5.name == "custom_expand"
        assert not layer5.trainable
        results.append(("Pass-through kwargs (name='custom_expand', trainable=False)", True, "Success"))
    except Exception as e:
        results.append(("Pass-through kwargs (name='custom_expand', trainable=False)", False, str(e)))

    # Scenario 6: Config serialization and from_config roundtrip
    try:
        layer6 = GF_Expand_Dims(ndim=2, name="config_test")
        cfg6 = layer6.get_config()
        reconstructed6 = GF_Expand_Dims.from_config(cfg6)
        assert reconstructed6.ndim == 2
        assert reconstructed6.name == "config_test"
        results.append(("get_config / from_config round-trip", True, "Success"))
    except Exception as e:
        results.append(("get_config / from_config round-trip", False, str(e)))

    # Print summary
    all_passed = True
    for desc, passed, msg in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {desc}: {msg}")
        if not passed:
            all_passed = False

    return all_passed, results


def run_custom_layers_config_roundtrips():
    print("\n" + "=" * 60)
    print("CHALLENGE 3: Custom Layer get_config / from_config Roundtrips")
    print("=" * 60)
    
    # 1. Active Model Layers (Present in the 3 trained .keras models)
    active_model_layers = [
        ("GF_Patches", GF_Patches(patch_size=3, hidden_dim=64, patch_method="extract")),
        ("GF_PatchEncoder", GF_PatchEncoder(num_patches=9, projection_dim=64)),
        ("GF_GlobalFilter", GF_GlobalFilter(patch_size=3, dim=64)),
        ("GF_Block", GF_Block(patch_size=3, dim=64, mlp_ratio=2.0)),
        ("GF_Expand_Dims", GF_Expand_Dims(ndim=2)),
        ("GF_MLP", GF_MLP(in_features=64, out_features=128, drop=0.1)),
        ("GF_DropPath", GF_DropPath(drop_prob=0.1, training=True)),
        ("ViT_Patches", ViT_Patches(patch_size=3, embed_dim=64)),
        ("ViT_PatchEncoder", ViT_PatchEncoder(num_patches=9, projection_dim=64)),
        ("ViT_Weighted_add", ViT_Weighted_add(wt_name="test_wt")),
        ("ViT_TransFormer", ViT_TransFormer(layer_num=1, num_heads=2, projection_dim=64)),
        ("ViT_TransFormer_Block", ViT_TransFormer_Block(num_layers=2, num_heads=2, projection_dim=64)),
        ("ViT_Class_Token_Norm", ViT_Class_Token_Norm(eps=1e-5)),
    ]

    # 2. Auxiliary / Legacy Layers registered in notebooks
    auxiliary_layers = [
        ("Pearson_correlation_masked", Pearson_correlation_masked(P_S=9)),
        ("ViT_SpatialAttention", ViT_SpatialAttention(k_size=3)),
        ("ViT_SpatialAttention1", ViT_SpatialAttention1(input_shape=(9, 9, 65))),
        ("Dropout_Train", Dropout_Train(rate=0.2, shift=2)),
    ]

    active_results = []
    print("--- Active Architecture Layers (In Target Models) ---")
    for name, layer_inst in active_model_layers:
        try:
            cfg = layer_inst.get_config()
            cls = CUSTOM_OBJECTS[name]
            recon = cls.from_config(cfg)
            active_results.append((name, True, "Config round-trip OK"))
            print(f"  [PASS] {name}: Config round-trip OK")
        except Exception as e:
            active_results.append((name, False, str(e)))
            print(f"  [FAIL] {name}: {e}")

    active_passed = all(r[1] for r in active_results)

    aux_results = []
    print("--- Auxiliary / Legacy Layers ---")
    for name, layer_inst in auxiliary_layers:
        try:
            cfg = layer_inst.get_config()
            cls = CUSTOM_OBJECTS[name]
            recon = cls.from_config(cfg)
            aux_results.append((name, True, "Config round-trip OK"))
            print(f"  [PASS] {name}: Config round-trip OK")
        except Exception as e:
            aux_results.append((name, False, str(e)))
            print(f"  [FINDING] {name} (non-blocking legacy): {e}")

    return active_passed, active_results, aux_results


def run_dynamic_batch_inference_stress_tests():
    print("\n" + "=" * 60)
    print("CHALLENGE 4: Dynamic Model Inference Across Batch Sizes & Distributions")
    print("=" * 60)

    BATCH_SIZES = [1, 2, 4, 8, 16]
    DISTRIBUTIONS = ["zeros", "ones", "normal", "uniform", "scaled_high", "scaled_low"]

    results = []

    for m_path in TARGET_MODELS:
        print(f"\n--- Testing Model: {m_path.name} ---")
        try:
            model = keras.models.load_model(m_path, custom_objects=CUSTOM_OBJECTS, compile=False)
            print(f"  Loaded model successfully.")
        except Exception as e:
            print(f"  [FATAL] Failed to load {m_path.name}: {e}")
            results.append((f"{m_path.name} Load", False, str(e)))
            continue

        model_passed = True

        for b_size in BATCH_SIZES:
            for dist in DISTRIBUTIONS:
                # Generate adversarial input tensor
                shape = (b_size, 9, 9, 65)
                if dist == "zeros":
                    data = np.zeros(shape, dtype=np.float32)
                elif dist == "ones":
                    data = np.ones(shape, dtype=np.float32)
                elif dist == "normal":
                    np.random.seed(42 + b_size)
                    data = np.random.randn(*shape).astype(np.float32)
                elif dist == "uniform":
                    np.random.seed(100 + b_size)
                    data = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
                elif dist == "scaled_high":
                    data = np.ones(shape, dtype=np.float32) * 50.0
                elif dist == "scaled_low":
                    data = np.ones(shape, dtype=np.float32) * 1e-4

                try:
                    raw_out = model.predict(data, verbose=0)
                    heads = list(raw_out) if isinstance(raw_out, (list, tuple)) else [raw_out]

                    # Invariant 1: Exactly 7 heads
                    assert len(heads) == K_HEADS, f"Batch {b_size} {dist}: got {len(heads)} heads != {K_HEADS}"

                    for h_i, head in enumerate(heads, start=1):
                        # Invariant 2: Shape is (B, 8)
                        assert head.shape == (b_size, NUM_CLASSES), f"Head {h_i}: shape {head.shape} != ({b_size}, {NUM_CLASSES})"

                        # Invariant 3: Finite values (no NaN or Inf)
                        assert np.all(np.isfinite(head)), f"Head {h_i} contains NaN/Inf under {dist}"

                        # Invariant 4: Probabilities in [0, 1]
                        assert np.all(head >= -1e-6), f"Head {h_i} contains negative values: min={np.min(head)}"
                        assert np.all(head <= 1.0 + 1e-6), f"Head {h_i} contains values > 1.0: max={np.max(head)}"

                        # Invariant 5: Softmax sum == 1.0 for every sample in batch
                        sample_sums = np.sum(head, axis=-1)
                        for s_idx, s_sum in enumerate(sample_sums):
                            assert np.isclose(s_sum, 1.0, atol=1e-4), f"Head {h_i} sample {s_idx}: sum={s_sum:.6f} != 1.0"

                except Exception as e:
                    model_passed = False
                    err_msg = f"Batch {b_size} Dist {dist}: {e}"
                    print(f"    [FAIL] {err_msg}")
                    results.append((f"{m_path.name} B={b_size} {dist}", False, err_msg))
                    break
            if not model_passed:
                break

        if model_passed:
            print(f"  [PASS] All {len(BATCH_SIZES)} batch sizes and {len(DISTRIBUTIONS)} input distributions passed all 5 invariants.")
            results.append((f"{m_path.name} Batch/Distribution Invariants", True, "All combinations passed"))

        # Invariant 6: Batch sample independence
        try:
            print(f"  Checking Batch Sample Independence Invariant for {m_path.name}...")
            np.random.seed(999)
            batch_data = np.random.uniform(0.0, 1.0, size=(4, 9, 9, 65)).astype(np.float32)
            preds_batch = model.predict(batch_data, verbose=0)
            
            # Predict sample 0 individually
            preds_single_0 = model.predict(batch_data[0:1], verbose=0)
            # Predict sample 2 individually
            preds_single_2 = model.predict(batch_data[2:3], verbose=0)

            for h_i in range(K_HEADS):
                diff_0 = np.max(np.abs(preds_batch[h_i][0] - preds_single_0[h_i][0]))
                diff_2 = np.max(np.abs(preds_batch[h_i][2] - preds_single_2[h_i][0]))
                assert diff_0 < 1e-4, f"Head {h_i+1} sample 0 batch cross-contamination! Max diff: {diff_0}"
                assert diff_2 < 1e-4, f"Head {h_i+1} sample 2 batch cross-contamination! Max diff: {diff_2}"

            print(f"  [PASS] Batch sample independence verified: max discrepancy < 1e-4 across all 7 heads.")
            results.append((f"{m_path.name} Batch Independence", True, "Zero cross-contamination"))
        except Exception as e:
            print(f"  [FAIL] Batch sample independence violated: {e}")
            results.append((f"{m_path.name} Batch Independence", False, str(e)))

    all_passed = all(r[1] for r in results)
    return all_passed, results


def main():
    print("=" * 70)
    print("EMPIRICAL ADVERSARIAL CHALLENGER SUITE — 65-BAND DIAS MULTI-HEAD")
    print(f"TensorFlow Version: {tf.__version__} | Keras Version: {keras.__version__}")
    print("=" * 70)

    challenge_reports = {}

    # Challenge 1
    c1_passed, c1_res = run_vit_weighted_add_challenges()
    challenge_reports["ViT_Weighted_add"] = {"passed": c1_passed, "details": c1_res}

    # Challenge 2
    c2_passed, c2_res = run_gf_expand_dims_challenges()
    challenge_reports["GF_Expand_Dims"] = {"passed": c2_passed, "details": c2_res}

    # Challenge 3
    c3_active_passed, c3_active_res, c3_aux_res = run_custom_layers_config_roundtrips()
    challenge_reports["Active_Layers_Config_Roundtrips"] = {"passed": c3_active_passed, "details": c3_active_res}
    challenge_reports["Auxiliary_Layers_Config_Roundtrips"] = {"passed": all(r[1] for r in c3_aux_res), "details": c3_aux_res}

    # Challenge 4
    c4_passed, c4_res = run_dynamic_batch_inference_stress_tests()
    challenge_reports["Dynamic_Inference"] = {"passed": c4_passed, "details": c4_res}

    # Gate condition: All active architecture layers, constructors, and dynamic inference MUST pass
    overall_passed = c1_passed and c2_passed and c3_active_passed and c4_passed

    print("\n" + "=" * 70)
    print("CHALLENGER SUMMARY")
    print("=" * 70)
    print(f"Challenge 1 (ViT_Weighted_add Constructors & Math):   {'PASS' if c1_passed else 'FAIL'}")
    print(f"Challenge 2 (GF_Expand_Dims Constructors & Shapes):   {'PASS' if c2_passed else 'FAIL'}")
    print(f"Challenge 3 (Active Model Layers Config Roundtrips):  {'PASS' if c3_active_passed else 'FAIL'}")
    print(f"Challenge 4 (Dynamic Batch Inference & Invariants):   {'PASS' if c4_passed else 'FAIL'}")
    print("-" * 70)
    verdict = "APPROVE" if overall_passed else "REJECT"
    print(f"FINAL VERDICT: {verdict}")
    print("=" * 70)

    # Save empirical summary to json for reporting
    output_json = Path("/Users/danesh/Desktop/Reports_dias/.agents/teamwork/teamwork_preview_challenger_m1_1/challenge_results.json")
    with open(output_json, "w") as f:
        json.dump({
            "overall_verdict": verdict,
            "overall_passed": overall_passed,
            "challenges": {
                k: {"passed": v["passed"], "num_tests": len(v["details"]), "passed_tests": sum(1 for d in v["details"] if d[1])}
                for k, v in challenge_reports.items()
            },
            "findings": [
                {"layer": r[0], "message": r[2]} for r in c3_aux_res if not r[1]
            ]
        }, f, indent=2)

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
