<div align="center">

<h1>🚀 Mamba-FCS</h1>

<h2>Mamba-Powered Semantic Change Detection<br>That Cuts Through Real-World Remote Sensing Chaos</h2>

<h3>
Joint Spatio-Frequency Fusion • Change-Guided Attention • SeK Loss<br>
→ State-of-the-Art on SECOND & Landsat-SCD with Efficient Long-Range Modeling
</h3>

<p>
<a href="https://arxiv.org/abs/2508.08232">
  <img src="https://img.shields.io/badge/arXiv-2508.08232-b31b1b.svg" alt="arXiv">
</a>
<a href="#">
  <img src="https://img.shields.io/badge/IEEE%20JSTARS-paper%20coming%20soon-00629B.svg" alt="IEEE JSTARS">
</a>
<a href="#">
  <img src="https://img.shields.io/badge/Weights-coming%20soon-7B2CBF.svg" alt="Weights">
</a>
</p>

<p>
Visual State Space backbone fused with frequency-aware features, bidirectional change guidance, and class-imbalance-aware loss—delivering robust, precise semantic change detection in the toughest remote sensing scenarios.
</p>

<p>
<a href="#updates">🔥 Updates</a> •
<a href="#overview">🔭 Overview</a> •
<a href="#method">🧠 Method</a> •
<a href="#quickstart">⚡ Quick Start</a> •
<a href="#data">🗂 Data</a> •
<a href="#train">🚀 Train & Eval</a> •
<a href="#results">📊 Results</a> •
<a href="#citation">📜 Cite</a>
</p>

</div>

---

## 🔥 Updates

- **Aug 2025** — Preprint live on arXiv: [2508.08232](https://arxiv.org/abs/2508.08232)  
- **Accepted** — IEEE JSTARS (camera-ready coming soon)  
- **Code Drop** — Full training pipeline + clean YAML configs now public  

Ready to push the boundaries of change detection? Let's go.

---

## 🔭 Overview

Semantic Change Detection in remote sensing is tough: seasonal shifts, lighting variations, and severe class imbalance constantly trip up traditional methods.

Mamba-FCS changes the game:

- **VMamba backbone** → linear-time long-range modeling (no more transformer VRAM nightmares)  
- **JSF fusion** → FFT-powered frequency cues for illumination robustness and razor-sharp edges  
- **CGA module** → change probabilities actively guide semantic refinement (and vice versa)  
- **SeK Loss** → finally treats rare classes with the respect they deserve

Outcome: cleaner maps, stronger rare-class recall, and real-world resilience.

<p align="center">
  <img src="docs/full_architecture.png" alt="Mamba-FCS Architecture" width="95%">
  <br><em>Spatial power + frequency smarts + change-guided attention = next-level SCD</em>
</p>

---

## 🧠 Method in ~30 Seconds

Feed in bi-temporal images **T1** and **T2**:

1. VMamba encoder extracts rich multi-scale features from both timestamps  
2. JSF block injects log-amplitude frequency information → appearance-invariant features  
3. CGA leverages change cues to tighten BCD ↔ SCD synergy  
4. Lightweight decoder predicts the final semantic change map  
5. SeK Loss drives balanced optimization, even when changed pixels are scarce

Simple. Smart. Superior.

---

## ⚡ Quick Start

### 1. Grab Pre-trained VMamba Weights

| Model         | Links                                                                                                    |
|---------------|----------------------------------------------------------------------------------------------------------|
| VMamba-Tiny   | [Zenodo](https://zenodo.org/records/14037769) • [GDrive](https://drive.google.com/file/d/160PXughGMNZ1GyByspLFS68sfUdrQE2N/view?usp=drive_link) • [BaiduYun](https://pan.baidu.com/s/1P9KRVy4lW8LaKJ898eQ_0w?pwd=7qxh) |
| VMamba-Small  | [Zenodo](https://zenodo.org/records/14037769) • [GDrive](https://drive.google.com/file/d/1dxHtFEgeJ9KL5WiLlvQOZK5jSEEd2Nmz/view?usp=drive_link) • [BaiduYun](https://pan.baidu.com/s/1RRjTA9ONhO43sBLp_a2TSw?pwd=6qk1) |
| VMamba-Base   | [Zenodo](https://zenodo.org/records/14037769) • [GDrive](https://drive.google.com/file/d/1kUHSBDoFvFG58EmwWurdSVZd8gyKWYfr/view?usp=drive_link) • [BaiduYun](https://pan.baidu.com/s/14_syzqwNnVB8rD3tejEZ4w?pwd=q825) |

Set `pretrained_weight_path` in your YAML to the downloaded `.pth`.

### 2. Install

```bash
git clone https://github.com/Buddhi19/MambaFCS.git
cd MambaFCS

conda create -n mambafcs python=3.10 -y
conda activate mambafcs

pip install --upgrade pip
pip install -r requirements.txt
pip install pyyaml
```

### 3. Build Selective Scan Kernel (Critical Step)

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

## 📊 Results

Straight from the paper — reproducible out of the box:

| Method        | Dataset       | OA (%) | F<sub>SCD</sub> (%) | mIoU (%) | SeK (%) |
|---------------|---------------|-------:|---------------------|---------:|--------:|
| **Mamba-FCS** | SECOND        | **88.62** | **65.78**        | **74.07** | **25.50** |
| **Mamba-FCS** | Landsat-SCD   | **96.25** | **89.27**        | **88.81** | **60.26** |

Visuals speak louder: expect dramatically cleaner boundaries and far better rare-class detection.

---

## 📜 Citation

If Mamba-FCS fuels your research, please cite:

```bibtex
@misc{wijenayake2025mambafcs,
      title={Mamba-FCS: Joint Spatio- Frequency Feature Fusion, Change-Guided Attention, and SeK Loss for Enhanced Semantic Change Detection in Remote Sensing}, 
      author={Buddhi Wijenayake and Athulya Ratnayake and Praveen Sumanasekara and Roshan Godaliyadda and Parakrama Ekanayake and Vijitha Herath and Nichula Wasalathilaka},
      year={2025},
      eprint={2508.08232},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2508.08232}, 
}
```

---

## 🌍🛰️ Let's detect real change — together.

Got questions or ideas? Open an issue. Stars fuel development ⭐

Happy experimenting!
