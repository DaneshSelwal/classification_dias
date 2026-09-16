# MASTER CONTEXT & WORKFLOW: Hyperspectral Image Classification & UQ (DIAS 65-Bands)

**Last Updated:** 2026-09-16
**Role:** This is the single authoritative "Master File" for any agent, developer, or automated system working in this workspace. It dictates the architectural setup, pipeline execution process, storage strategy, and historical context of the DIAS 65-bands Uncertainty Quantification (UQ) project.

---

## 1. Project Domain & Objective
This project implements a generalized framework for **Hyperspectral Remote Sensing Image Classification and Uncertainty Quantification**. 
Specifically, this repository is dedicated to processing the **DIAS agricultural dataset (65 spectral bands)**. The goal is to train deep learning models (AlexNet, GFNet, ViT-UNet) and apply various UQ methodologies (Conformal Prediction, Ensembles, Distillation) to measure epistemic and aleatoric uncertainty, yielding reliable spatial maps and metrics.

---

## 2. Workspace Architecture & Storage Strategy
The environment is split across three primary locations to balance local Mac memory limitations with the need for persistent cloud backups.

1. **Active Local Workspace (`/Users/danesh/Documents/AGY/agy/classification/`)**
   - The primary staging and development ground.
   - **Git Tracking:** Tracks the remote GitHub repository `DaneshSelwal/classification_dias.git`.
   - **Constraint:** Do not store redundant `.zip` or unzipped caches here. Clean up `dl_temp/` folders after processing.

2. **Google Drive Cold Storage (`/Users/danesh/Documents/AGY/classification_65bands/`)**
   - The ultimate local mirror to the Google Drive cloud backup.
   - **Syncing Rule:** This folder must strictly mirror the finalized artifacts from the Local Workspace using `rsync`. It is never to be used as a working directory.

3. **GitHub Repository (`classification_dias`)**
   - **Git LFS:** All heavy `.keras` and `.h5` model weights MUST be tracked via Git Large File Storage (LFS). 
   - Before pushing, always run `git lfs track "folder/models/*.keras"`.

---

## 3. The Orchestration Pipeline (How Work Gets Done)
Due to computational constraints on the local Mac, **all model training is orchestrated autonomously on Kaggle GPUs**. Any new pipeline must follow this strict 9-step execution workflow:

### Step 1: Notebook Adaptation
The local Jupyter notebook must be adapted for Kaggle. File paths must point to `/kaggle/input/...` and outputs must be written to `/kaggle/working/...`.

### Step 2: Obfuscation Strategy (Crucial)
To prevent Kaggle automated bans/flags when uploading datasets:
- Every dataset file and model weight must be renamed to a generic alias (e.g., `k001.mat`, `k016.keras`).
- The mapping must be registered in `/Users/danesh/Documents/AGY/agy/classification/kaggle_name_mapping.json`.

### Step 3: Kaggle Upload
- **Data/Models:** Large files must be uploaded as a Kaggle Dataset (`kaggle datasets create`).
- **Code:** The notebook is pushed as a Kaggle Kernel (`kaggle kernels push`).

### Step 4: Execution & Monitoring
Run `kaggle kernels status <slug>` and poll until the status is `COMPLETE`.

### Step 5: Download Outputs
Use `kaggle kernels output <slug> -p <local_temp_folder> -o` to pull down the generated `.keras` weights and evaluation CSV/Excel files.

### Step 6: Local Organization
Move downloaded artifacts from the temp folder to their permanent homes (e.g., `baseline/models/` and `baseline/results/`).

### Step 7: Local Cache Cleanup
**Immediately delete** the local `dl_temp/` caches to free up Mac storage memory.

### Step 8: Git Commit & Push
Commit the models and results. **Agents must use `BypassSandbox: true`** when executing `git push` to GitHub.

### Step 9: Google Drive Sync
Run `rsync -av /Users/danesh/Documents/AGY/agy/classification/ /Users/danesh/Documents/AGY/classification_65bands/ --exclude '.git'` to update the cold storage.

---

## 4. Completed Pipelines & Methodologies
*All items listed here have been fully trained, evaluated, pushed to GitHub (via Git LFS), and synced to Google Drive.*

1. **Baseline (`baseline/`)**
   - **Method:** Softmax, MC Dropout, Temperature Scaling.
   - **Models:** AlexNet, GFNet, ViT-UNet (both `_best` and `_final`).
2. **Self-Adaptive Conformal Prediction (`sacp/`)**
   - **Method:** Post-hoc calibration evaluated across spatial windows (3, 5, 7, 9).
3. **Multi-Head CP (`multicp/`)**
   - **Method:** Architectural modification outputting multi-head predictions.
   - **Models:** 6 multi-head `.keras` variants.
4. **Deep Ensembles & CreDE (`ensemble/`)**
   - **Method:** 15 distinct teacher models trained (5 seeds per architecture).
   - **Evaluation:** CreDE spatial uncertainty maps generated.
   - **Kaggle Dependency:** These 15 models were uploaded as a dedicated Kaggle Dataset (`daneshselwal/dias-ensemble-models`, obfuscated as `k016-k030`) so other kernels could access them.

## 5. Active / Pending Pipelines
1. **CREDIT Distillation (`credit/`)**
   - **Method:** Conformalized Regularized Distillation Training (Student learning from the 15 Ensembles).
   - **Status:** Currently training on Kaggle as `k015` v4. 
   - *Note for future agents:* Kaggle kernels cannot read large `.keras` files from other kernels via `kernel_sources`. This pipeline historically failed until the ensemble teachers were mounted explicitly as a `dataset_sources` dependency.

## 6. Future Skeletons (To Be Implemented)
The original research repository contains several advanced methodologies that currently exist only as empty "skeleton" directories in this workspace. Future iterations of this project will port and execute these:
- `dapm/` (Deep Adaptive Predictive Modeling)
- `dofa/` (Dynamic Wavelength Tokenization)
- `georsclip/` (Vision-Language Model)
- `mambahsi/` (State Space Models)
- `spatialgcn/` (Graph Convolutional Networks)
- `multicp_sacp/` (Spatial Multi-Head CP)

---

## 7. Hard Rules for Autonomous Agents
1. **Never delete `.keras` files** from Git-tracked folders unless you are intentionally removing them from the GitHub repository using `git rm`.
2. **Always use `BypassSandbox: true`** when communicating with network interfaces (Git push, Kaggle CLI commands, Rsync).
3. **Protect the Google Drive:** Only write to `/Users/danesh/Documents/AGY/classification_65bands/` using `rsync` from the local workspace. Do not treat it as a scratch pad.
4. **Update this Document:** If a new pipeline is executed, a new Kaggle dataset is created, or a core architectural rule changes, this `agent.md` file MUST be updated to reflect the new state.
