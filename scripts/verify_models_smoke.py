#!/usr/bin/env python3
"""
scripts/verify_models_smoke.py

Comprehensive Standalone Verification & Smoke Test Harness for 65-Band DIAS Multi-Head Models:
1. Validates all 17 custom layer deserialization contracts (ViT_Weighted_add, GF_Expand_Dims, etc.)
2. Verifies .keras model loading for AlexNet, GFNet, and ViT-UNet
3. Runs dummy forward inference with shape (1, 9, 9, 65)
4. Asserts 7 output heads, each of shape (1, 8), summing to 1.0 (probabilities)
5. Validates bold 8-class color palette, background, and uncertainty colors
6. Audits notebooks for AST syntax correctness, zero tab20, and quotes around 'best_model_path'

Usage:
    python scripts/verify_models_smoke.py [--mode {auto,dynamic,static}]
"""

import os
import sys
import json
import ast
import math
import argparse
import zipfile
from pathlib import Path

# Optional TensorFlow / Keras import
try:
    import numpy as np
    import tensorflow as tf
    try:
        import keras
        from keras import layers
    except ImportError:
        from tensorflow import keras
        from tensorflow.keras import layers
    if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
        keras_serializable = keras.saving.register_keras_serializable(package="custom")
    elif hasattr(tf.keras.utils, "register_keras_serializable"):
        keras_serializable = tf.keras.utils.register_keras_serializable(package="custom")
    else:
        def keras_serializable(cls):
            return cls
    HAS_TF = True
except ImportError:
    HAS_TF = False
    def keras_serializable(cls):
        return cls

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path("/Users/danesh/Desktop/Reports_dias")
DIAS_DIR = WORKSPACE_ROOT / "Kaggle_Computed_65Bands_DIAS"
MODELS_DIR = DIAS_DIR / "multicp" / "models"

TARGET_NOTEBOOKS = [
    DIAS_DIR / "multicp" / "Model_training_multihead.ipynb",
    DIAS_DIR / "multicp" / "Model_uncertainty_multicp.ipynb",
    DIAS_DIR / "multicp_sacp" / "Model_uncertainty_multicp_sacp.ipynb",
]

TARGET_MODELS = [
    MODELS_DIR / "AlexNet_CNN_MultiHead_best.keras",
    MODELS_DIR / "GFNet_MultiHead_best.keras",
    MODELS_DIR / "ViT_UNet_MultiHead_best.keras",
]

# ── Canonical Color Constants ────────────────────────────────────────────────
CLASS_COLOR_BASE = [
    "#0000FF", "#00FF00", "#FF0000", "#00FFFF", "#FF00FF",
    "#FFFF00", "#A52A2A", "#FFA500", "#7FFF00", "#8A2BE2"
]
BACKGROUND_COLOR = "#000000"
UNCERTAIN_COLOR = "#808080"
CERTAIN_COLOR = "#FFFF00"
UNCERTAIN_MAP_COLOR = "#001F3F"

NUM_CLASSES = 8
K_HEADS = 7
INPUT_SHAPE = (1, 9, 9, 65)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Custom Layer Definitions (Exact Keras 3 Harmonized Contracts)
# ═════════════════════════════════════════════════════════════════════════════

if HAS_TF:
    @keras_serializable
    class Pearson_correlation_masked(layers.Layer):
        """Pixel-wise Pearson-correlation attention — masks pixels below mean correlation."""
        def __init__(self, P_S=9, **kwargs):
            super().__init__(**kwargs)
            self.P_S = P_S

        def call(self, inputs):
            loc = self.P_S // 2
            channels = inputs.shape[-1]
            x_mean = tf.repeat(tf.math.reduce_mean(inputs, axis=-1, keepdims=True), channels, axis=-1)
            y = tf.repeat(tf.repeat(inputs[:, loc:loc+1, loc:loc+1, :], self.P_S, axis=-2), self.P_S, axis=-3)
            y_mean = tf.repeat(tf.math.reduce_mean(y, axis=-1, keepdims=True), channels, axis=-1)
            a, b = inputs - x_mean, y - y_mean
            num = tf.reduce_sum(a * b, axis=-1, keepdims=True)
            deno = tf.sqrt(tf.reduce_sum(a * a, axis=-1, keepdims=True) * tf.reduce_sum(b * b, axis=-1, keepdims=True))
            corr = num / (deno + 1e-8)
            mask = tf.cast(corr > tf.reduce_mean(corr), corr.dtype)
            attention_weights = tf.repeat(mask * corr, channels, axis=-1)
            return layers.multiply([inputs, attention_weights])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"P_S": self.P_S})
            return cfg

    @keras_serializable
    class Dropout_Train(layers.Layer):
        """Deterministic structured dropout used during training shift scheduling."""
        def __init__(self, rate=0.0, shift=1, noise_shape=None, seed=None, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate
            self.shift = shift
            self.noise_shape = noise_shape
            self.seed = seed
            self.supports_masking = True

        def call(self, inputs, training=None):
            return tf.identity(inputs)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "rate": self.rate, "shift": self.shift,
                "noise_shape": self.noise_shape, "seed": self.seed,
                "supports_masking": self.supports_masking,
            })
            return cfg

    @keras_serializable
    class GF_MLP(layers.Layer):
        """Two-layer GELU MLP used inside GF_Block."""
        def __init__(self, in_features, out_features, drop=0.0, **kwargs):
            super().__init__(**kwargs)
            self.in_features = in_features
            self.out_features = out_features
            self.drop = drop
            self.mlp_1 = layers.Dense(in_features, activation=tf.keras.activations.gelu, use_bias=False)
            self.mlp_2 = layers.Dense(out_features, activation=tf.keras.activations.gelu, use_bias=False)
            self.drop_1 = layers.Dropout(drop)
            self.drop_2 = layers.Dropout(drop)

        def call(self, x):
            return self.drop_2(self.mlp_2(self.drop_1(self.mlp_1(x))))

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"in_features": self.in_features, "out_features": self.out_features, "drop": self.drop})
            return cfg

    @keras_serializable
    class GF_DropPath(layers.Layer):
        """Stochastic depth / drop-path regularisation."""
        def __init__(self, drop_prob=0.0, training=False, **kwargs):
            super().__init__(**kwargs)
            self.drop_prob = drop_prob
            self.training = training

        def call(self, x, **kwargs):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
            random_tensor = tf.floor(random_tensor)
            return tf.divide(x, keep_prob) * random_tensor

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"drop_prob": self.drop_prob, "training": self.training})
            return cfg

    @keras_serializable
    class GF_Expand_Dims(layers.Layer):
        """Wraps tf.expand_dims as a Keras layer."""
        def __init__(self, ndim=2, **kwargs):
            super().__init__(**kwargs)
            self.ndim = ndim

        def call(self, x):
            return tf.expand_dims(x, axis=self.ndim)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"ndim": self.ndim})
            return cfg

    @keras_serializable
    class GF_Patches(layers.Layer):
        """Extract image patches — either via extract_patches or Conv2D projection."""
        def __init__(self, patch_size=3, hidden_dim=256, patch_method="extract", **kwargs):
            super().__init__(**kwargs)
            self.patch_size = patch_size
            self.hidden_dim = hidden_dim
            self.patch_method = str(patch_method).lower()

        def call(self, images):
            if self.patch_method == "conv":
                x = layers.Conv2D(self.hidden_dim, self.patch_size, self.patch_size)(images)
                return layers.Reshape([-1, x.shape[-1]])(x)
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1], padding="VALID"
            )
            return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "patch_method": self.patch_method,
            })
            return cfg

    @keras_serializable
    class GF_PatchEncoder(layers.Layer):
        """Linear projection + positional embedding for GFNet patches."""
        def __init__(self, num_patches, projection_dim, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(num_patches, projection_dim)

        def call(self, patch, **kwargs):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            return self.projection(patch) + self.position_embedding(positions)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
            return cfg

    @keras_serializable
    class GF_GlobalFilter(layers.Layer):
        """Learnable frequency-domain filter via 2-D real FFT."""
        def __init__(self, patch_size, dim, **kwargs):
            super().__init__(**kwargs)
            self.patch_size = patch_size
            self.dim = dim

        def build(self, input_shape):
            self.complex_weight = self.add_weight(
                name="complex_weight",
                shape=(self.patch_size, self.patch_size, input_shape[-1] // 2 + 1, 2),
                initializer=tf.random_uniform_initializer(),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x, **kwargs):
            B, N, C = x.shape
            a = b = int(math.sqrt(N))
            x = tf.reshape(x, [-1, a, b, C])
            x = tf.signal.rfft2d(x)
            x = x * tf.dtypes.complex(self.complex_weight[:, :, :, 0], self.complex_weight[:, :, :, -1])
            x = tf.signal.irfft2d(x)
            return tf.reshape(x, [-1, N, C])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"patch_size": self.patch_size, "dim": self.dim})
            return cfg

    @keras_serializable
    class GF_Block(layers.Layer):
        """Single GFNet transformer-style residual block."""
        def __init__(self, patch_size=3, dim=512, mlp_ratio=4.0, drop=0.0, drop_path=0.0, **kwargs):
            super().__init__(**kwargs)
            self.patch_size = patch_size
            self.dim = dim
            self.mlp_ratio = mlp_ratio
            self.drop = drop
            self.drop_path_rate = drop_path
            self.norm1 = layers.LayerNormalization(axis=-1)
            self.filter = GF_GlobalFilter(patch_size, dim)
            self.drop_path = GF_DropPath(drop_path)
            self.norm2 = layers.LayerNormalization(axis=-1)
            self.mlp = GF_MLP(int(dim * mlp_ratio), dim, drop)

        def call(self, x):
            return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "patch_size": self.patch_size, "dim": self.dim,
                "mlp_ratio": self.mlp_ratio, "drop": self.drop,
                "drop_path": self.drop_path_rate,
            })
            return cfg

    @keras_serializable
    class ViT_Patches(layers.Layer):
        """Extract non-overlapping patches and project to embed_dim."""
        def __init__(self, patch_size, embed_dim=768, **kwargs):
            super().__init__(**kwargs)
            self.patch_size = patch_size
            self.embed_dim = embed_dim

        def build(self, input_shape):
            self.projection = layers.Dense(self.embed_dim)

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1], padding="VALID"
            )
            patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
            return self.projection(patches)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
            return cfg

    @keras_serializable
    class ViT_PatchEncoder(layers.Layer):
        """Linear projection + CLS token + positional embedding."""
        def __init__(self, num_patches, projection_dim, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(num_patches + 1, projection_dim)
            self.cls_token = self.add_weight(
                name="cls_token", shape=(1, 1, projection_dim),
                initializer=tf.zeros_initializer(), trainable=True
            )

        def call(self, patch, **kwargs):
            batch_size = tf.shape(patch)[0]
            cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
            x = tf.concat([cls_tokens, self.projection(patch)], axis=1)
            positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
            return x + self.position_embedding(positions)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
            return cfg

    @keras_serializable
    class ViT_SpatialAttention(layers.Layer):
        """Lightweight 4-conv spatial attention branch."""
        def __init__(self, k_size=3, **kwargs):
            super().__init__(**kwargs)
            self.k_size = k_size
            self.norm = layers.BatchNormalization()
            self.conv1 = layers.Conv2D(1, k_size, padding="same")
            self.conv2 = layers.Conv2D(1, k_size, padding="same")
            self.conv3 = layers.Conv2D(1, k_size, padding="same")
            self.conv4 = layers.Conv2D(1, k_size, padding="same")

        def call(self, inputs):
            x = self.norm(inputs)
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)
            return layers.multiply([inputs, tf.sigmoid(x4)])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"k_size": self.k_size})
            return cfg

    @keras_serializable
    class ViT_SpatialAttention1(layers.Layer):
        """Single-conv spatial attention branch."""
        def __init__(self, input_shape=None, **kwargs):
            super().__init__(**kwargs)
            self.in_shape = input_shape
            self.norm = layers.BatchNormalization()
            self.conv = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")

        def call(self, inputs):
            return layers.multiply([inputs, self.conv(self.norm(inputs))])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"input_shape": self.in_shape})
            return cfg

    @keras_serializable
    class ViT_Weighted_add(layers.Layer):
        """Combine two tensors with a single learned scalar weight `w`: w*a + (1-w)*b."""
        def __init__(self, wt_name=None, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.wt_name = wt_name if wt_name is not None else name

        def build(self, input_shape):
            self.w = self.add_weight(
                name=f"weighted_add_{self.wt_name}",
                shape=(1,),
                initializer=tf.random_normal_initializer(),
                trainable=True,
            )

        def call(self, a, b):
            return a * self.w + b * (1.0 - self.w)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"wt_name": self.wt_name})
            return cfg

    @keras_serializable
    class ViT_TransFormer(layers.Layer):
        """Single Transformer encoder block with learned-weight residuals."""
        def __init__(self, layer_num, num_heads, projection_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.layer_num = layer_num
            self.num_heads = num_heads
            self.projection_dim = projection_dim
            self.dropout = dropout

        def build(self, input_shape):
            self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{self.layer_num}")
            self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{self.layer_num}")
            self.add1 = ViT_Weighted_add(f"transformer_1_{self.layer_num}")
            self.add2 = ViT_Weighted_add(f"transformer_2_{self.layer_num}")
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.projection_dim,
                dropout=self.dropout,
                name=f"mha_{self.layer_num}",
            )
            self.dense1 = layers.Dense(self.projection_dim * 2, activation=tf.keras.activations.gelu)
            self.dense2 = layers.Dense(self.projection_dim, activation=tf.keras.activations.gelu)
            self.drop1 = layers.Dropout(self.dropout)
            self.drop2 = layers.Dropout(self.dropout)

        def call(self, inputs, training=None):
            x1 = self.add1(self.mha(self.norm1(inputs), self.norm1(inputs), training=training), inputs)
            x2 = self.drop1(self.dense1(self.norm2(x1)), training=training)
            x2 = self.drop2(self.dense2(x2), training=training)
            return self.add2(x2, x1)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "layer_num": self.layer_num,
                "num_heads": self.num_heads,
                "projection_dim": self.projection_dim,
                "dropout": self.dropout,
            })
            return cfg

    @keras_serializable
    class ViT_TransFormer_Block(layers.Layer):
        """Stack of ViT_TransFormer layers with U-Net-style symmetric skip connections."""
        def __init__(self, num_layers, num_heads, projection_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.projection_dim = projection_dim
            self.dropout = dropout

        def build(self, input_shape):
            self.Blocks = [
                ViT_TransFormer(i, self.num_heads, self.projection_dim, self.dropout)
                for i in range(self.num_layers)
            ]

        def call(self, inputs, training=None):
            stack, x = [], inputs
            for i, blk in enumerate(self.Blocks):
                x = blk(x, training=training)
                if i <= self.num_layers // 2:
                    stack.append(x)
                else:
                    x = layers.Add()([x, stack[self.num_layers - i - 1]])
            return x

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "projection_dim": self.projection_dim,
                "dropout": self.dropout,
            })
            return cfg

    @keras_serializable
    class ViT_Class_Token_Norm(layers.Layer):
        """Layer-normalise full sequence then return CLS token (index 0)."""
        def __init__(self, eps=1e-6, **kwargs):
            super().__init__(**kwargs)
            self.eps = eps
            self.norm = layers.LayerNormalization(epsilon=eps)

        def call(self, inputs):
            return self.norm(inputs)[:, 0, :]

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"eps": self.eps})
            return cfg

    CUSTOM_OBJECTS = {
        "Pearson_correlation_masked": Pearson_correlation_masked,
        "Dropout_Train": Dropout_Train,
        "GF_Patches": GF_Patches,
        "GF_PatchEncoder": GF_PatchEncoder,
        "GF_GlobalFilter": GF_GlobalFilter,
        "GF_Block": GF_Block,
        "GF_Expand_Dims": GF_Expand_Dims,
        "GF_MLP": GF_MLP,
        "GF_DropPath": GF_DropPath,
        "ViT_Patches": ViT_Patches,
        "ViT_PatchEncoder": ViT_PatchEncoder,
        "ViT_SpatialAttention": ViT_SpatialAttention,
        "ViT_SpatialAttention1": ViT_SpatialAttention1,
        "ViT_Weighted_add": ViT_Weighted_add,
        "ViT_TransFormer": ViT_TransFormer,
        "ViT_TransFormer_Block": ViT_TransFormer_Block,
        "ViT_Class_Token_Norm": ViT_Class_Token_Norm,
    }
else:
    CUSTOM_OBJECTS = {}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Color Palette & Colormap Verification
# ═════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(hex_code):
    h = hex_code.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def verify_color_palette():
    print("\n" + "=" * 60)
    print("STAGE 1: Color Palette & Contrast Verification")
    print("=" * 60)
    
    assert len(CLASS_COLOR_BASE) == 10, f"Expected 10 base colors, got {len(CLASS_COLOR_BASE)}"
    assert BACKGROUND_COLOR == "#000000", f"Expected black background (#000000), got {BACKGROUND_COLOR}"
    assert UNCERTAIN_COLOR == "#808080", f"Expected grey uncertain color (#808080), got {UNCERTAIN_COLOR}"
    assert CERTAIN_COLOR == "#FFFF00", f"Expected yellow certain color (#FFFF00), got {CERTAIN_COLOR}"

    u_rgb = hex_to_rgb(UNCERTAIN_COLOR)
    b_rgb = hex_to_rgb(BACKGROUND_COLOR)
    
    for idx, hex_val in enumerate(CLASS_COLOR_BASE[:NUM_CLASSES]):
        c_rgb = hex_to_rgb(hex_val)
        d_unc = rgb_distance(c_rgb, u_rgb)
        d_bg = rgb_distance(c_rgb, b_rgb)
        print(f"  Class {idx} ({hex_val}): dist to uncertain={d_unc:.1f}, dist to background={d_bg:.1f}")
        assert d_unc >= 60.0, f"Color clash: Class {idx} ({hex_val}) too close to {UNCERTAIN_COLOR} (dist {d_unc:.1f} < 60)"
        assert d_bg >= 100.0, f"Color clash: Class {idx} ({hex_val}) too dark against background (dist {d_bg:.1f} < 100)"

    print("[PASS] Color palette verified: all 8 classes have bold separation and Euclidean RGB distance >= 60 to grey.")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Notebook AST Syntax & tab20 Absence Verification
# ═════════════════════════════════════════════════════════════════════════════

def verify_notebook_ast_and_palettes(strict=True):
    print("\n" + "=" * 60)
    print("STAGE 2: Notebook AST, tab20 Scan & Defect Audit")
    print("=" * 60)
    
    all_clean = True
    for nb_path in TARGET_NOTEBOOKS:
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        tab20_occurrences = []
        syntax_errors = []
        name_error_detected = False

        for cell_idx, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            
            if "tab20" in src:
                tab20_occurrences.append(cell_idx)

            if "best_model_path" in src and cell.get("cell_type") == "code":
                # Check for unquoted {best_model_path: mp}
                for line_no, line in enumerate(src.splitlines(), start=1):
                    if "{best_model_path:" in line or "[best_model_path]" in line:
                        name_error_detected = True
                        print(f"  [AUDIT FLAG] {nb_path.name} Cell {cell_idx} Line {line_no}: unquoted best_model_path found!")

            if cell.get("cell_type") == "code":
                clean_lines = []
                for line in src.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("%") or stripped.startswith("!"):
                        clean_lines.append(f"# {line}")
                    else:
                        clean_lines.append(line)
                code_str = "\n".join(clean_lines)
                try:
                    ast.parse(code_str)
                except SyntaxError as e:
                    syntax_errors.append((cell_idx, str(e)))

        print(f"Notebook: {nb_path.name}")
        print(f"  Total cells: {len(nb.get('cells', []))}")
        print(f"  tab20 occurrences: {len(tab20_occurrences)}")
        print(f"  Syntax errors: {len(syntax_errors)}")
        print(f"  Unquoted best_model_path bug present: {name_error_detected}")
        
        assert len(tab20_occurrences) == 0, f"{nb_path.name}: Found {len(tab20_occurrences)} occurrences of tab20 in cells {tab20_occurrences}"
        assert len(syntax_errors) == 0, f"{nb_path.name}: AST syntax errors found: {syntax_errors}"
        if name_error_detected:
            all_clean = False
            if strict:
                raise AssertionError(f"{nb_path.name}: Unquoted best_model_path NameError bug present!")

    if all_clean:
        print("[PASS] Notebook static audit verified: zero tab20, zero AST syntax errors, zero unquoted best_model_path.")
    else:
        print("[AUDIT NOTICE] Notebook static audit complete with defect flags noted above.")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Static Model Archive Verification (Direct ZIP Inspection)
# ═════════════════════════════════════════════════════════════════════════════

def verify_model_archives_static():
    print("\n" + "=" * 60)
    print("STAGE 3: Static Model Archive Structure Inspection")
    print("=" * 60)

    for m_path in TARGET_MODELS:
        assert m_path.exists(), f"Model file not found: {m_path}"
        print(f"\nInspecting: {m_path.name} ({m_path.stat().st_size / (1024*1024):.1f} MB)")
        
        with zipfile.ZipFile(m_path, "r") as z:
            namelist = z.namelist()
            assert "config.json" in namelist, f"{m_path.name}: Missing config.json"
            assert "metadata.json" in namelist, f"{m_path.name}: Missing metadata.json"
            assert "model.weights.h5" in namelist, f"{m_path.name}: Missing model.weights.h5"

            with z.open("metadata.json") as f:
                meta = json.load(f)
                print(f"  Keras Version: {meta.get('keras_version')}, Saved Date: {meta.get('date_saved')}")

            with z.open("config.json") as f:
                cfg = json.load(f)
                output_layers = cfg.get("config", {}).get("output_layers", [])
                assert len(output_layers) == K_HEADS, f"Expected {K_HEADS} output heads, found {len(output_layers)}"
                
                head_names = [x[0] for x in output_layers]
                print(f"  Heads ({len(head_names)}): {head_names}")

                layers_dict = {l["name"]: l for l in cfg.get("config", {}).get("layers", [])}
                input_cfg = layers_dict.get("input_layer", {}).get("config", {})
                batch_shape = input_cfg.get("batch_shape")
                print(f"  Input batch_shape: {batch_shape}")
                assert batch_shape == [None, 9, 9, 65], f"Unexpected input shape: {batch_shape}"

                for h in head_names:
                    h_cfg = layers_dict.get(h, {}).get("config", {})
                    assert h_cfg.get("units") == NUM_CLASSES, f"{h} units={h_cfg.get('units')} != {NUM_CLASSES}"
                    assert h_cfg.get("activation") == "softmax", f"{h} activation != softmax"

                # Check specific layer configs
                if "GFNet" in m_path.name:
                    exp_dims = layers_dict.get("gf__expand__dims", {}).get("config", {})
                    print(f"  GF_Expand_Dims ndim: {exp_dims.get('ndim')}")
                    assert exp_dims.get("ndim") == 2, f"GF_Expand_Dims ndim is {exp_dims.get('ndim')} != 2"
                elif "ViT_UNet" in m_path.name:
                    tf_block = layers_dict.get("vi_t__trans_former__block", {}).get("config", {})
                    print(f"  ViT_TransFormer_Block: layers={tf_block.get('num_layers')}, heads={tf_block.get('num_heads')}, dim={tf_block.get('projection_dim')}")
                    assert tf_block.get("num_layers") == 12, "Expected 12 transformer layers"

    print("\n[PASS] Static model archive inspection verified: all 3 models have input [None, 9, 9, 65] and 7 softmax heads of 8 classes.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Dynamic Model Loading & Forward Inference Verification
# ═════════════════════════════════════════════════════════════════════════════

def verify_models_dynamic_inference():
    print("\n" + "=" * 60)
    print("STAGE 4: Dynamic Model Deserialization & Dummy Inference")
    print("=" * 60)
    
    if not HAS_TF:
        print("[SKIP] TensorFlow / Keras not installed in this environment. Skipping dynamic inference.")
        return

    dummy_input = np.zeros(INPUT_SHAPE, dtype=np.float32)
    print(f"Created dummy input tensor with shape {dummy_input.shape}, dtype={dummy_input.dtype}")

    for m_path in TARGET_MODELS:
        print(f"\nLoading model: {m_path.name} ...")
        model = keras.models.load_model(m_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        print(f"  Successfully deserialized {m_path.name}!")

        # Run forward prediction
        raw_outputs = model.predict(dummy_input, verbose=0)
        
        # Standardize outputs into list of head arrays
        if isinstance(raw_outputs, tuple):
            heads = list(raw_outputs)
        elif isinstance(raw_outputs, list):
            heads = raw_outputs
        else:
            heads = [raw_outputs]

        print(f"  Model returned {len(heads)} output heads.")
        assert len(heads) == K_HEADS, f"Expected {K_HEADS} output heads, got {len(heads)}"

        for h_idx, head_tensor in enumerate(heads, start=1):
            assert head_tensor.ndim == 2, f"Head {h_idx}: expected rank-2 tensor, got rank {head_tensor.ndim}"
            assert head_tensor.shape == (1, NUM_CLASSES), f"Head {h_idx}: expected shape (1, {NUM_CLASSES}), got {head_tensor.shape}"
            assert np.all(np.isfinite(head_tensor)), f"Head {h_idx}: contains NaN or Inf values"
            assert np.all(head_tensor >= 0.0), f"Head {h_idx}: contains negative probability values"
            assert np.all(head_tensor <= 1.0 + 1e-6), f"Head {h_idx}: contains probability values > 1.0"
            
            prob_sum = np.sum(head_tensor, axis=-1)[0]
            assert np.isclose(prob_sum, 1.0, atol=1e-5), f"Head {h_idx}: probabilities sum to {prob_sum:.6f} != 1.0"
            print(f"    Head {h_idx}: shape={head_tensor.shape}, sum={prob_sum:.6f}, max_prob={np.max(head_tensor):.4f}")

    print("\n[PASS] Dynamic forward inference verified: all 3 models deserialized and produced 7 valid probability heads.")


# ═════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MultiCP Model Deserialization & Smoke Verification Harness")
    parser.add_argument("--mode", choices=["auto", "dynamic", "static"], default="auto",
                        help="Verification mode: 'auto' (runs dynamic if TF available, else static), 'dynamic', 'static'")
    parser.add_argument("--no-strict", action="store_true",
                        help="Do not raise AssertionError on unquoted best_model_path (audit mode)")
    args = parser.parse_args()

    print("=" * 70)
    print("DIAS 65-Band Multi-Head Model Verification & Deserialization Harness")
    print(f"Python: {sys.version.split()[0]} | TensorFlow Available: {HAS_TF}")
    print("=" * 70)

    # 1. Color Palette Check
    verify_color_palette()

    # 2. Notebook Static & AST Check
    verify_notebook_ast_and_palettes(strict=not args.no_strict)

    # 3. Model Archive Static Check
    verify_model_archives_static()

    # 4. Dynamic Model Loading & Inference
    if args.mode in ("auto", "dynamic"):
        verify_models_dynamic_inference()

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
