<div align="center">
<h1 align="center">Mamba-FCS</h1>

<h3 align="center">
Mamba-FCS: Joint Spatio-Frequency Feature Fusion, Change-Guided Attention, and SeK Loss for Enhanced Semantic Change Detection in Remote Sensing
</h3>

<p align="center">
<a href="https://arxiv.org/abs/2508.08232"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.08232-b31b1b.svg"></a>
</p>

<p align="center">
<b>Semantic Change Detection (SCD)</b> with a Visual State Space backbone enhanced by frequency-aware fusion, change-guided cross-task attention, and an imbalance-aware SeK loss.
</p>

<p align="center">
<a href="#-overview">Overview</a> •
<a href="#-method-in-30-seconds">Method</a> •
<a href="#%EF%B8%8F-quick-start">Quick Start</a> •
<a href="#-data-preparation">Data</a> •
<a href="#-training--evaluation-yaml-driven">Training</a> •
<a href="#-results">Results</a> •
<a href="#-citation">Citation</a>
</p>
</div>

---

## 🛎️ Updates
- **2025-08-11**: Preprint available on arXiv (v1).

## 🔭 Overview
**Mamba-FCS** targets **semantic change detection** from bi-temporal remote sensing imagery, addressing:
- Long-range context modelling with **state-space models**.
- Illumination/appearance variation robustness via **joint spatio-frequency fusion (JSF)**.
- Stronger coupling between **BCD ↔ SCD** via **Change-Guided Attention (CGA)**.
- Class-imbalance aware optimisation using **Separated Kappa (SeK) Loss**.

<p align="center">
  <img src="docs/full_architecture.png" alt="Mamba-FCS Overall Architecture" width="92%">
</p>

> This repo’s training entrypoint is YAML-driven via `MambaFCS/train.py` (see below).

---

## 🧠 Method in 30 seconds
Given T1 and T2 images:
1. **Backbone**: Visual State Space encoder extracts multi-scale features.
2. **JSF block**: fuses spatial features with log-amplitude frequency cues (FFT-based).
3. **CGA module**: uses change cues to guide semantic alignment and refine change semantics.
4. **SeK Loss**: optimises class-imbalanced SCD behaviour with a separated-kappa-inspired objective.

---

## ⚙️ Quick Start

### A. Installation
Recommended: Linux

Install PyTorch first (match your CUDA version), then install repo deps.

```bash
conda create -n mambafcs python=3.10 -y
conda activate mambafcs
python -m pip install -U pip
python -m pip install -r MambaFCS/requirements.txt
python -m pip install pyyaml
```

### B. Build selective-scan CUDA op (required for VMamba-style backbones)
```bash
cd MambaFCS/kernels/selective_scan
python3 -m pip install .
cd ../../../..
```

If you hit build errors, ensure your `torch` CUDA build, `nvcc`, and GCC version are compatible.

---

## 🗂 Data Preparation

This codebase includes loaders for **SECOND** and **Landsat-SCD** in a *common “A/B + labelA/labelB”* layout.

### SECOND
Expected layout:
```text
/path/to/SECOND/
├── train/
│   ├── A/         # T1 images
│   ├── B/         # T2 images
│   ├── labelA/    # semantic labels at T1 (single-channel class IDs)
│   └── labelB/    # semantic labels at T2 (single-channel class IDs)
├── test/
│   ├── A/
│   ├── B/
│   ├── labelA/
│   └── labelB/
├── train.txt      # filenames (e.g., 0001.png), one per line
└── test.txt
```

### Landsat-SCD
Expected layout:
```text
/path/to/Landsat-SCD/
├── A/
├── B/
├── labelA/
├── labelB/
├── train_list.txt
├── val_list.txt
└── test_list.txt
```

Notes:
- `labelA/labelB` must be **integer class maps** (not RGB palettes). Update preprocessing if your dataset uses color-coded labels.
- List files contain **filenames only** (with extension), matching files inside `A/`, `B/`, `labelA/`, `labelB/`.

---

## 🚀 Training & Evaluation

Training is launched via:
- `MambaFCS/train.py` (reads a YAML config, builds args, runs `train_MambaSCD.Trainer`)
- Dataset-specific YAMLs in `MambaFCS/configs/`

### 1) (One-time) Set dataset paths in the YAML
Edit:
- `MambaFCS/configs/train_LANDSAT.yaml`
- `MambaFCS/configs/train_SECOND.yaml`

Update `dataset_root`, `train_dataset_path`, and `test_dataset_path` to match your machine.

### 2) Train on Landsat-SCD
```bash
python3 MambaFCS/train.py --config MambaFCS/configs/train_LANDSAT.yaml
```

### 3) Train on SECOND
```bash
python3 MambaFCS/train.py --config MambaFCS/configs/train_SECOND.yaml
```

### 4) Sanity-check your resolved config (recommended)
```bash
python3 MambaFCS/train.py -c MambaFCS/configs/train_SECOND.yaml --dry-run
```

### 5) Override the VMamba/VSSM yacs config at runtime (optional)
`--opts` appends to the YAML `opts` field and is passed into `yacs.merge_from_list(...)`:
```bash
python3 MambaFCS/train.py -c MambaFCS/configs/train_SECOND.yaml --opts MODEL.DROP_PATH_RATE 0.2
```

### Outputs
- Checkpoints: `model_param_path/model_saving_name/` (from the YAML)
- TensorBoard logs: `saved_models/model_saving_name/logs/`

### GPU selection
The YAMLs contain `cuda_device`. If you set `CUDA_VISIBLE_DEVICES`, make sure `cuda_device` matches the *visible* indexing (usually `0`).

### Resume training
If you set `resume`, you must also set `optim_path` and `scheduler_path` in the YAML (required by `train_MambaSCD.Trainer`).

---

## 🧪 Results
Reported metrics (from the paper; update if you retrain with different settings):

| Method        | Dataset     | OA (%) | F<sub>scd</sub> (%) | mIoU (%) | SeK (%) |
| ------------- | ----------- | -----: | ------------------: | ------: | ------: |
| **Mamba-FCS** | SECOND      |  88.62 |               65.78 |   74.07 |   25.50 |
| **Mamba-FCS** | Landsat-SCD |  96.25 |               89.27 |   88.81 |   60.26 |

---

## 📜 Citation
If you use this code in your research, please cite:

```bibtex
@misc{wijenayake2025mambafcs,
  title        = {Mamba-FCS: Joint Spatio-Frequency Feature Fusion, Change-Guided Attention, and SeK Loss for Enhanced Semantic Change Detection in Remote Sensing},
  author       = {Wijenayake, Buddhi and Ratnayake, Athulya and Sumanasekara, Praveen and Godaliyadda, Roshan and Ekanayake, Parakrama and Herath, Vijitha and Wasalathilaka, Nichula},
  year         = {2025},
  eprint       = {2508.08232},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV}
}
```
