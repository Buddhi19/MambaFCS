<div align="center">

<h1>🚀 Mamba-FCS</h1>

<h2>Joint Spatio-Frequency Feature Fusion with Change-Guided Attention and SeK Loss</h2>

<h2>🏆 Current Best-Performing Algorithm for Semantic Change Detection 🏆</h2>

<p>
  <a href="https://ieeexplore.ieee.org/document/11391528">
    <img src="https://img.shields.io/badge/IEEE%20JSTARS-Official%20Publication-00629B.svg" alt="IEEE JSTARS Paper">
  </a>
  <a href="https://arxiv.org/abs/2508.08232">
    <img src="https://img.shields.io/badge/arXiv-2508.08232-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/buddhi19/MambaFCS/tree/main">
    <img src="https://img.shields.io/badge/Hugging%20Face-Weights%20Available-FFD21E.svg" alt="Weights">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

<p>
Visual State Space backbone fused with explicit spatio–frequency cues, bidirectional change guidance, and class-imbalance-aware loss—delivering robust, precise semantic change detection under tough illumination/seasonal shifts and severe long-tail labels.
</p>

<p>
<a href="#updates">🔥 Updates</a> •
<a href="#overview">🔭 Overview</a> •
<a href="#why-spatiofrequency-matters">✨ Why Spatio–Frequency?</a> •
<a href="#method">🧠 Method</a> •
<a href="#quickstart">⚡ Quick Start</a> •
<a href="#data">🗂 Data</a> •
<a href="#train--evaluation">🚀 Train & Eval</a> •
<a href="#interactive-notebook">🧪 Interactive Notebook</a> •
<a href="#results">📊 Results</a> •
<a href="#acknowledgements">🙏 Acknowledgements</a> •
<a href="#citation">📜 Cite</a>
</p>

</div>

---

## 🔥🔥 Updates
- **Mar 2026 - Weights + Notebook Released** — Official Mamba-FCS checkpoints are now available on Hugging Face: https://huggingface.co/buddhi19/MambaFCS/tree/main, and the interactive evaluation/annotation notebook is available at `annotations/MambaFCS.ipynb`
- **Feb 2026 - Paper Published** — IEEE JSTARS (Official DOI: https://doi.org/10.1109/JSTARS.2026.3663066)
- **Jan 2026 - Accepted** — IEEE JSTARS (Camera-ready version submitted)
- **Jan 2026 - Code Released** — Full training pipeline with structured YAML configurations is now available
- **Aug 2025 - Preprint Released** — Preprint available on arXiv: https://arxiv.org/abs/2508.08232

Ready to push the boundaries of change detection? Let's go.

---

## 🔭 Overview

Semantic Change Detection in remote sensing is tough: seasonal shifts, lighting variations, and severe class imbalance constantly trip up traditional methods.

Mamba-FCS changes the game:

- **VMamba backbone** → linear-time long-range modeling (no more transformer VRAM nightmares)  
- **JSF spatio–frequency fusion** → injects FFT log-amplitude cues into spatial features for appearance invariance + sharper boundaries  
- **CGA module** → change probabilities actively guide semantic refinement (and vice versa)  
- **SeK Loss** → finally treats rare classes with the respect they deserve  

Outcome: cleaner maps, stronger rare-class recall, and real-world resilience.

<p align="center">
  <img src="docs/full_architecture.png" alt="Mamba-FCS Architecture" width="95%">
  <br><em>Spatial power + frequency smarts + change-guided attention = next-level SCD</em>
</p>

---

## ✨ Why Spatio–Frequency Matters

Remote sensing change detection suffers from **appearance shifts** (illumination, seasonal phenology, atmospheric effects).  
Purely spatial feature fusion can overfit to texture/color changes, while **frequency-domain cues** capture structure and boundaries more consistently.

**Mamba-FCS explicitly combines:**
- **Spatial modeling (VMamba / state-space)** for long-range context
- **Frequency cues (FFT log-amplitude)** for appearance robustness
- **Change-guided cross-task attention** to tighten BCD ↔ SCD synergy

This spatio–frequency + change-guided design is a key reason for strong rare-class performance and cleaner semantic boundaries.

---

## 🧠 Method in ~30 Seconds

Feed in bi-temporal images **T1** and **T2**:

1. VMamba encoder extracts rich multi-scale features from both timestamps  
2. JSF injects **frequency-domain log-amplitude (FFT)** into spatial features → stronger invariance to illumination/seasonal shifts  
3. CGA leverages change cues to tighten BCD ↔ SCD synergy  
4. Lightweight decoder predicts the final semantic change map  
5. SeK Loss drives balanced optimization, even when changed pixels are scarce  

Simple. Smart. Superior.

---

## ⚡ Quick Start

### 1. Download Released Mamba-FCS Weights

Pretrained Mamba-FCS checkpoints are now hosted on Hugging Face: [buddhi19/MambaFCS](https://huggingface.co/buddhi19/MambaFCS/tree/main).

Use these weights directly for inference and evaluation, or keep them alongside your experiment checkpoints for quick benchmarking.

### 2. Grab Pre-trained VMamba Weights

| Model         | Links                                                                                                    |
|---------------|----------------------------------------------------------------------------------------------------------|
| VMamba-Tiny   | [Zenodo](https://zenodo.org/records/14037769) • [GDrive](https://drive.google.com/file/d/160PXughGMNZ1GyByspLFS68sfUdrQE2N/view?usp=drive_link) • [BaiduYun](https://pan.baidu.com/s/1P9KRVy4lW8LaKJ898eQ_0w?pwd=7qxh) |
| VMamba-Small  | [Zenodo](https://zenodo.org/records/14037769) • [GDrive](https://drive.google.com/file/d/1dxHtFEgeJ9KL5WiLlvQOZK5jSEEd2Nmz/view?usp=drive_link) • [BaiduYun](https://pan.baidu.com/s/1RRjTA9ONhO43sBLp_a2TSw?pwd=6qk1) |
| VMamba-Base   | [Zenodo](https://zenodo.org/records/14037769) • [GDrive](https://drive.google.com/file/d/1kUHSBDoFvFG58EmwWurdSVZd8gyKWYfr/view?usp=drive_link) • [BaiduYun](https://pan.baidu.com/s/14_syzqwNnVB8rD3tejEZ4w?pwd=q825) |

Set `pretrained_weight_path` in your YAML to the downloaded `.pth`.

### 3. Install

```bash
git clone https://github.com/Buddhi19/MambaFCS.git
cd MambaFCS

conda create -n mambafcs python=3.10 -y
conda activate mambafcs

pip install --upgrade pip
pip install -r requirements.txt
pip install pyyaml
````

### 4. Build Selective Scan Kernel (Critical Step)

```bash
cd kernels/selective_scan
pip install .
cd ../../..
```

(Pro tip: match your torch CUDA version with nvcc/GCC if you hit issues.)

---

## 🗂 Data Preparation

Plug-and-play support for **SECOND** and **Landsat-SCD**.

### SECOND Layout

```
/path/to/SECOND/
├── train/
│   ├── A/          # T1 images
│   ├── B/          # T2 images
│   ├── labelA/     # T1 class IDs (single-channel)
│   └── labelB/     # T2 class IDs
├── test/
│   ├── A/
│   ├── B/
│   ├── labelA/
│   └── labelB/
├── train.txt
└── test.txt
```

### Landsat-SCD

Same idea, with `train_list.txt`, `val_list.txt`, `test_list.txt`.

**Must-do**: Use integer class maps (not RGB). Convert palettes first.

---

## 🚀 Train & Evaluation

YAML-driven — clean and flexible.

1. Edit paths in `configs/train_LANDSAT.yaml` or `configs/train_SECOND.yaml`

2. Fire it up:

```bash
# Landsat-SCD
python train.py --config configs/train_LANDSAT.yaml

# SECOND
python train.py --config configs/train_SECOND.yaml
```

Checkpoints + TensorBoard logs land in `saved_models/<your_name>/`.

Resume runs? Just flip `resume: true` and point to optimizer/scheduler states.

---

<a id="interactive-notebook"></a>
## 🧪 Interactive Evaluation & Annotation

For an interactive workflow, use the notebook [`annotations/MambaFCS.ipynb`](annotations/MambaFCS.ipynb).

It is set up for users who want to:

- run evaluations interactively
- inspect predictions and qualitative outputs
- perform annotation and review in a notebook-driven workflow

Pair it with the released checkpoints on [Hugging Face](https://huggingface.co/buddhi19/MambaFCS/tree/main) for fast experimentation without retraining.

---

## 📊 Results

Straight from the paper — reproducible out of the box:

| Method        | Dataset     |    OA (%) | F<sub>SCD</sub> (%) |  mIoU (%) |   SeK (%) |
| ------------- | ----------- | --------: | ------------------- | --------: | --------: |
| **Mamba-FCS** | SECOND      | **88.62** | **65.78**           | **74.07** | **25.50** |
| **Mamba-FCS** | Landsat-SCD | **96.25** | **89.27**           | **88.81** | **60.26** |

Visuals speak louder: expect dramatically cleaner boundaries and far better rare-class detection.

---

## 🙏 Acknowledgements

This work is strongly influenced by prior advances in state-space vision backbones and Mamba-based change detection.
In particular, we acknowledge:

* **VMamba (Visual State Space Models for Vision)** — backbone inspiration: [https://github.com/MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)
* **ChangeMamba** — Mamba-style change detection inspiration: [https://github.com/ChenHongruixuan/ChangeMamba.git](https://github.com/ChenHongruixuan/ChangeMamba.git)

---

## 📜 Citation

If Mamba-FCS fuels your research, please cite:

```bibtex
@ARTICLE{mambafcs,
  author={Wijenayake, Buddhi and Ratnayake, Athulya and Sumanasekara, Praveen and Godaliyadda, Roshan and Ekanayake, Parakrama and Herath, Vijitha and Wasalathilaka, Nichula},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Mamba-FCS: Joint Spatio-Frequency Feature Fusion, Change-Guided Attention, and Sek Inspired Loss for Enhanced Semantic Change Detection in Remote Sensing}, 
  year={2026},
  volume={},
  number={},
  pages={1-19},
  keywords={Remote sensing imagery;semantic change detection;separated kappa;spatial–frequency fusion;state-space models},
  doi={10.1109/JSTARS.2026.3663066}
}
```

You might consider citing:

```bibtex
@misc{wijenayake2025precisionspatiotemporalfeaturefusion,
      title={Precision Spatio-Temporal Feature Fusion for Robust Remote Sensing Change Detection}, 
      author={Buddhi Wijenayake and Athulya Ratnayake and Praveen Sumanasekara and Nichula Wasalathilaka and Mathivathanan Piratheepan and Roshan Godaliyadda and Mervyn Ekanayake and Vijitha Herath},
      year={2025},
      eprint={2507.11523},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2507.11523}, 
}
```

```bibtex
@INPROCEEDINGS{11217111,
  author={Ratnayake, R.M.A.M.B. and Wijenayake, W.M.B.S.K. and Sumanasekara, D.M.U.P. and Godaliyadda, G.M.R.I. and Herath, H.M.V.R. and Ekanayake, M.P.B.},
  booktitle={2025 Moratuwa Engineering Research Conference (MERCon)}, 
  title={Enhanced SCanNet with CBAM and Dice Loss for Semantic Change Detection}, 
  year={2025},
  volume={},
  number={},
  pages={84-89},
  keywords={Training;Accuracy;Attention mechanisms;Sensitivity;Semantics;Refining;Feature extraction;Transformers;Power capacitors;Remote sensing},
  doi={10.1109/MERCon67903.2025.11217111}}
```

---

## 🌍🛰️ Got inspired? Give us a STAR 
