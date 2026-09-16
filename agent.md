# Project Context & Handoff: Hyperspectral Image Classification & UQ (DIAS 65-Bands)

**Date Generated:** 2026-09-16
**Purpose:** To maintain context for future agents and projects working on this repository.

## 1. Project Overview
This project adapts the generalized Hyperspectral Classification and Uncertainty Quantification (UQ) framework for the specific **DIAS 65-bands dataset**. 
Due to heavy computational requirements, training is orchestrated autonomously on **Kaggle GPUs**, and results are downloaded, git-committed, and synced locally and to Google Drive.

## 2. Directory Structure & Syncing Rules
- **Local Workspace:** `/Users/danesh/Documents/AGY/agy/classification/` (Tracks the `classification_dias` GitHub repo).
- **Google Drive Mirror:** `/Users/danesh/Documents/AGY/classification_65bands/` (Used for cold storage backup).
- **Golden Rule for Drive:** The Drive folder acts as the ultimate backup. Local heavy caches/untracked files should be deleted from the Mac to save memory, but the Drive folder must remain untouched as cold storage.
- **Git LFS:** All `.keras` and `.h5` model weights are tracked via Git LFS.

## 3. Kaggle Obfuscation & Dataset Architecture
To prevent Kaggle automated bans/flags, all data and models are obfuscated before upload.
- **Mapping Registry:** `kaggle_name_mapping.json` (Maintains bidirectional aliases like `k001.mat`, `k016.keras`).
- **Base Dataset:** `daneshselwal/dias-dataset-k01` (Contains the raw MATLAB image/reference data).
- **Ensemble Dataset:** `daneshselwal/dias-ensemble-models` (Contains 15 trained teacher `.keras` models needed for CREDIT distillation, obfuscated as `k016` through `k030`).

## 4. Completed Pipelines & Models (Fully Synced)
1. **Baseline Pipeline (`baseline/`)**
   - Models trained: AlexNet, GFNet, ViT-UNet.
   - Status: Complete. Models and evaluations synced.
2. **Self-Adaptive Conformal Prediction (`sacp/`)**
   - Evaluated across spatial windows 3, 5, 7, and 9.
   - Status: Complete. Excel and CSV summaries synced.
3. **Multi-Head CP (`multicp/`)**
   - Models trained: 6 Multi-head variants.
   - Status: Complete.
4. **Deep Ensembles & CreDE (`ensemble/`)**
   - Models trained: 15 teachers (5 AlexNet, 5 GFNet, 5 ViT-UNet).
   - Evaluation: CreDE spatial maps and pixel-wise epistemic/aleatoric uncertainty.
   - Status: Complete. Uploaded to Kaggle dataset.

## 5. Pending / Active Pipelines
- **CREDIT Distillation (`credit/`)**
   - Architecture: Dual-head student networks learning from the 15 ensemble teachers.
   - Status: Kernel `k015` (v4) is currently executing on Kaggle. Requires downloading results, pushing to GitHub, and syncing to Drive upon completion.

## 6. Future Agent Instructions
- **Do not** blindly delete `.keras` files from tracked Git folders (like `models/`) locally unless using `git rm`, otherwise it will break the repository state.
- Always use `BypassSandbox: true` for `git push`, Kaggle CLI commands, and `rsync`.
- Rely on `kaggle_name_mapping.json` when uploading new data to Kaggle.
