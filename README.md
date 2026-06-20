<div align="center">

<h1>🚀 Mamba-FCS</h1>

<h4>Official implementation of Mamba-FCS: Joint Spatio- Frequency Feature Fusion, Change-Guided Attention, and SeK Loss for Enhanced Semantic Change Detection in Remote Sensing
</h4>

<h4>🏆 The Current Best-Performing Algorithm for Remote Sensing Semantic Change Detection 🏆</h4>

## 🌐 [**Project Page**](https://buddhi19.github.io/MambaFCS/)

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11391528">
    <img src="https://img.shields.io/badge/IEEE%20JSTARS-Published%20Paper-00629B?logo=ieee&logoColor=white" alt="IEEE JSTARS Published Paper">
  </a>
  <a href="https://arxiv.org/abs/2508.08232">
    <img src="https://img.shields.io/badge/arXiv-2508.08232-B31B1B?logo=arxiv&logoColor=white" alt="arXiv Preprint">
  </a>
  <a href="https://huggingface.co/buddhi19/MambaFCS/tree/main">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model%20Weights-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Model Weights">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  </a>
  <a href="#-citations">
    <img src="https://img.shields.io/badge/Citations-6-blue.svg" alt="Citations">
  </a>
</p>
<p>
Visual State Space backbone fused with explicit spatio–frequency cues, bidirectional change guidance, and class-imbalance-aware loss—delivering robust, precise semantic change detection under tough illumination/seasonal shifts and severe long-tail labels.
</p>

<!-- <p>
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
</p> -->

</div>

---

## 🔥🔥 Updates
- **Mar 2026 - Notebook Released** For an interactive workflow, use the notebook **✨✨[`annotations/MambaFCS.ipynb`](annotations/MambaFCS.ipynb)✨✨**.
- **Mar 2026 - Weights + Notebook Released** — Official Mamba-FCS checkpoints are now available on **🤗🤗[Hugging Face](https://huggingface.co/buddhi19/MambaFCS/tree/main)🤗🤗**.
- **Feb 2026 - Paper Published** — IEEE JSTARS (Official DOI: https://doi.org/10.1109/JSTARS.2026.3663066)
- **Jan 2026 - Accepted** — IEEE JSTARS (Camera-ready version submitted)
- **Jan 2026 - Code Released** — Full training pipeline with structured YAML configurations is now available
- **Aug 2025 - Preprint Released** — Preprint available on arXiv: https://arxiv.org/abs/2508.08232

Ready to push the boundaries of change detection? Let's go.

---

## 🔭 Overview

Semantic Change Detection in remote sensing is tough: seasonal shifts, lighting variations, and severe class imbalance constantly trip up traditional methods.

We try to solve this problem by,

- **VMamba backbone** → linear-time long-range modeling
- **Joint spatio–frequency fusion** → injects FFT log-amplitude cues into spatial features for appearance invariance + sharper boundaries  
- **CGA module** → change probabilities actively guide semantic refinement (and vice versa)  
- **SeK Loss** → direct optimization for evaluation metrics 

<p align="center">
  <img src="docs/full_architecture.png" alt="Mamba-FCS Architecture" width="95%">
  <br><em>Spatial power + frequency smarts + change-guided attention =Mamba-FCS</em>
</p>

---

## ✨ Why Spatio–Frequency Matters

The frequency domain is known to reveal latent structures in signals that remain obscure in the spatial domain. 

### Building on this premise, we explore whether Fourier transformation of latent representations can expose similarly discriminative hidden features.

---

## 🧠 Method

Feed in bi-temporal images **T1** and **T2**:

1. VMamba encoder extracts rich multi-scale features from both timestamps  
2. JSF injects **frequency-domain log-amplitude (FFT)** into spatial features
3. CGA leverages change cues to tighten BCD ↔ SCD synergy  
4. Lightweight decoder predicts the final semantic change map  
5. SeK Loss drives balanced optimization, even when changed pixels are scarce  

---

## ⚡ Quick Start

### 1. Download Released Mamba-FCS Weights

Pretrained Mamba-FCS checkpoints are now hosted on Hugging Face: [buddhi19/MambaFCS](https://huggingface.co/buddhi19/MambaFCS/tree/main).

Use these weights directly for inference and evaluation, or keep them alongside your experiment checkpoints for quick benchmarking.

### 2. Grab Pre-trained VMamba Weights

| Model         | Links                                                                                                    |
|---------------|----------------------------------------------------------------------------------------------------------|
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

Install a compatible ```pytorch``` version for your current CUDA setup. We installed,

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Build Selective Scan Kernel (Critical Step)

The kernel's `setup.py` imports `torch` at build time but ships no
`pyproject.toml`, so pip's default build isolation can't see your installed
torch (`ModuleNotFoundError: No module named 'torch'`). Always build with
`--no-build-isolation`:

```bash
cd kernels/selective_scan
pip install . --no-build-isolation
cd ../..
```

Your `nvcc` must match your torch CUDA **major** version — compare
`python -c "import torch; print(torch.version.cuda)"` with `nvcc --version`. If
they differ, install/activate a matching CUDA toolkit before building.

<details>
<summary><b>Building against CUDA 13 (torch <code>cu130</code>) with no matching system toolkit</b></summary>

If torch is a `cu130` build but the machine only has a CUDA 12.x system toolkit
(or none, and you lack root), install a matching CUDA 13 compiler into the venv
as pip wheels — torch already provides the cu13 runtime/headers:

```bash
cd kernels/selective_scan

# CUDA 13 compiler + CUB/Thrust (CCCL) headers + ninja
pip install "cuda-toolkit[nvcc,cccl]==13.0.2" ninja
# Pin nvvm + crt to nvcc's version (13.0.88); otherwise the newer front-end
# emits PTX the 13.0 ptxas rejects ("Unsupported .version 9.3").
pip install "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88"

# Point the build at the in-venv CUDA 13 toolkit
CU=$(python -c "import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),'cu13'))")
ln -sf libcudart.so.13 "$CU/lib/libcudart.so"   # unversioned lib for -lcudart
export CUDA_HOME="$CU" PATH="$CU/bin:$PATH" LD_LIBRARY_PATH="$CU/lib:$LD_LIBRARY_PATH" MAX_JOBS=4

pip install . --no-build-isolation
cd ../..
```

CUDA 13 also needs two source tweaks that are already applied in this repo —
dropping the removed `compute_70` (Volta) gencode in `setup.py` (now targets
`sm_80`/`sm_86`; adjust for your GPU) and a CUB 3.0 shim for the removed
`cub::LaneId()`/`cub::CTA_SYNC()`. See
[`kernels/selective_scan/README.md`](kernels/selective_scan/README.md) for the
full rationale.
</details>

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

 Use integer class maps (not RGB). Convert palettes first.

---

## 🚀 Train & Evaluation

We support ```YAML``` driven training via,

1. Edit paths in `configs/train_LANDSAT.yaml` or `configs/train_SECOND.yaml`

2. Start Training:

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

Notebook supports,

- run evaluations interactively
- inspect predictions and qualitative outputs
- perform annotations

Pair it with the released checkpoints on [Hugging Face](https://huggingface.co/buddhi19/MambaFCS/tree/main) for fast experimentation without retraining.

---

## 📊 Results

<p><strong>Straight from the paper — reproducible out of the box:</strong></p>

We choose a **```batchsize of 2```** for all of our experiments.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Dataset</th>
      <th>OA (%)</th>
      <th>F<sub>SCD</sub> (%)</th>
      <th>mIoU (%)</th>
      <th>SeK (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SCanNet</td>
      <td>SECOND</td>
      <td>87.86</td>
      <td>63.66</td>
      <td>73.42</td>
      <td>23.94</td>
    </tr>
    <tr>
      <td>ChangeMamba</td>
      <td>SECOND</td>
      <td>88.12</td>
      <td>64.03</td>
      <td>73.68</td>
      <td>24.11</td>
    </tr>
    <tr>
      <td><span style="color:red;"><strong>Mamba-FCS</strong></span></td>
      <td>SECOND</td>
      <td><span style="color:red;"><strong>88.62</strong></span></td>
      <td><span style="color:red;"><strong>65.78</strong></span></td>
      <td><span style="color:red;"><strong>74.07</strong></span></td>
      <td><span style="color:red;"><strong>25.50</strong></span></td>
    </tr>
    <tr>
      <td>SCanNet</td>
      <td>Landsat-SCD</td>
      <td>96.04</td>
      <td>85.62</td>
      <td>86.37</td>
      <td>52.63</td>
    </tr>
    <tr>
      <td>ChangeMamba</td>
      <td>Landsat-SCD</td>
      <td>96.08</td>
      <td>86.61</td>
      <td>86.91</td>
      <td>53.66</td>
    </tr>
    <tr>
      <td><span style="color:red;"><strong>Mamba-FCS</strong></span></td>
      <td>Landsat-SCD</td>
      <td><span style="color:red;"><strong>96.25</strong></span></td>
      <td><span style="color:red;"><strong>89.27</strong></span></td>
      <td><span style="color:red;"><strong>88.81</strong></span></td>
      <td><span style="color:red;"><strong>60.26</strong></span></td>
    </tr>
  </tbody>
</table>

## ```If you find errors``` when reproducing the results please raise an ``issue```. We would be happy to help 💁.


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
  title={Mamba-FCS: Joint Spatio-Frequency Feature Fusion, Change-Guided Attention, and SeK Inspired Loss for Enhanced Semantic Change Detection in Remote Sensing}, 
  year={2026},
  volume={19},
  number={},
  pages={7680-7698},
  keywords={Semantics;Feature extraction;Transformers;Remote sensing;Frequency-domain analysis;Decoding;Computational modeling;Computer architecture;Context modeling;Lighting;Remote sensing imagery;semantic change detection (CD);separated Kappa (SeK);spatial–frequency fusion;state-space models (SSMs)},
  doi={10.1109/JSTARS.2026.3663066}}

```


You might consider citing most influenced papers for our work,

```bibtex
@INPROCEEDINGS{11450773,
  author={Wijenayake, W.M.B.S.K. and Ratnayake, R.M.A.M.B. and Sumanasekara, D.M.U.P. and Wasalathilaka, N.S. and Piratheepan, M. and Godaliyadda, G.M.R.I. and Ekanayake, M.P.B. and Herath, H.M.V.R.},
  booktitle={2025 IEEE 19th International Conference on Industrial and Information Systems (ICIIS)}, 
  title={Precision Spatio-Temporal Feature Fusion for Robust Remote Sensing Change Detection}, 
  year={2026},
  volume={19},
  number={},
  pages={557-562},
  keywords={Accuracy;Computational modeling;Pipelines;Feature extraction;Transformers;Decoding;Remote sensing;Optimization;Monitoring;Context modeling;Remote Sensing;Binary Change Detection;State Space Models;Mamba},
  doi={10.1109/ICIIS69028.2026.11450773}}

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

@article{Chen_2024,
   title={ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model},
   volume={62},
   ISSN={1558-0644},
   url={http://dx.doi.org/10.1109/TGRS.2024.3417253},
   DOI={10.1109/tgrs.2024.3417253},
   journal={IEEE Transactions on Geoscience and Remote Sensing},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Chen, Hongruixuan and Song, Jian and Han, Chengxi and Xia, Junshi and Yokoya, Naoto},
   year={2024},
   pages={1–20} }

@misc{liu2024vmambavisualstatespace,
      title={VMamba: Visual State Space Model}, 
      author={Yue Liu and Yunjie Tian and Yuzhong Zhao and Hongtian Yu and Lingxi Xie and Yaowei Wang and Qixiang Ye and Jianbin Jiao and Yunfan Liu},
      year={2024},
      eprint={2401.10166},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.10166}, 
}
```

---

## 🌍🛰️ Got inspired? Give us a STAR🌟🌟
