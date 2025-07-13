# ESEG: Event-Based Segmentation Boosted by Explicit Edge‑Semantic Guidance

## Project Overview
ESEG is a novel **event‑based semantic segmentation (ESS)** framework that leverages **explicit edge‑semantic supervision** to guide dense semantic feature extraction from sparse and noisy event data. Unlike traditional methods that rely solely on complex network architectures to learn edge awareness implicitly, ESEG introduces semantic edge labels to **explicitly inform the model where valuable features reside**, enhancing segmentation reliability—especially in high‑speed or low‑light environments.

Event cameras, due to their high temporal resolution and dynamic range, provide great potential in scenes where RGB sensors often fail. However, event data is inherently sparse and lacks sufficient context for pixel‑level semantic tasks. ESEG addresses this challenge by exploiting the fact that event triggers are highly correlated with motion‑induced edges.

---

## Core Components and Technical Insights

### 1. Edge‑Semantic Guidance
ESEG introduces a branch that learns **semantic edge maps** from event data. Since standard datasets do not offer semantic edge labels, the authors propose **SELSAM** (Semantic Edge Label with SAM), a pipeline based on the Segment Anything Model (SAM) to generate clean, edge‑focused supervision.

* **SELSAM** avoids noisy internal texture edges by leveraging SAM's robust segmentation masks and localized semantic assignment.
* These edge maps serve as explicit guidance for the segmentation model, steering its attention to reliable edge‑rich regions.

### 2. Multi‑Branch Architecture

| Branch          | Goal                          | Backbone        |
| --------------- | ----------------------------- | --------------- |
| Edge‑Semantic   | Learn semantic edge features  | DFF (ResNet‑34) |
| Dense‑Semantic  | Learn full‑scene semantics    | SegFormer       |
| Fusion (D2CAF)  | Merge edge & dense semantics  | —               |

### 3. D2CAF Module (Density‑Aware Dynamic‑Window Cross‑Attention Fusion)
D2CAF adaptively fuses sparse edge features and dense semantic features:

1. **Density Indicator Matrix (DIM)** – estimates information density via entropy & distance metrics.
2. **Dynamic Window Masking** – widens/narrows the attention window based on second‑order gradients of edge density.
3. **Cross‑Attention Fusion** – blends edge and dense features while suppressing noisy regions.

---

## Experimental Results

### DSEC‑Semantic

| Model      | Accuracy | mIoU |
| ---------- | -------- | ---- |
| **ESEG‑B** | 90.22 %  | 55.93 |
| **ESEG‑L** | 91.47 %  | 57.55 |

### DDD17

| Model      | Accuracy | mIoU |
| ---------- | -------- | ---- |
| **ESEG‑B** | 89.64 %  | 57.01 |
| **ESEG‑L** | 90.68 %  | 59.97 |

ESEG consistently outperforms prior ESS baselines (Ev‑SegNet, EvSegFormer, HMNet, etc.) and even RGB‑trained hybrids such as EvDistill.

---

## Requirements

> **Recommended environment:** Ubuntu 20.04 LTS, Python ≥3.9, CUDA 11.x, cuDNN 8.x (GPU with ≥8 GB VRAM).

```text
# Core
torch>=2.1.0           # enable torch.compile
torchvision>=0.16.0
numpy>=1.23
opencv-python>=4.8
pillow>=10.0

# Training utilities
tqdm>=4.66
einops>=0.7            # used by SegFormer
timm>=0.9              # MiT backbone weights loader
scikit-image>=0.22
scipy>=1.11
```



---

## Code Structure

```
ESEG/
├── DFF/          # Lightweight DFF edge detector (ResNet-34); powers the edge-semantic branch
├── dst/                 # Dataset loaders, augmentation & IO helpers for DSEC-Semantic and DDD17
├── local_configs/       # YAML / JSON experiment configs (paths, hyper-params, model size, etc.)
├── model/               # Core network definitions: SegFormer, DFF adapters, and the D2CAF fusion block
├── trainer/             # Training / evaluation entry scripts, experiment manager & checkpoints
├── utils/               # Common utilities (logging, metric computation, visualization, seed control)
├── eseg_d^2caf_core.py  # Stand-alone D2CAF core code for quick understanding
└── README.md            
```

## Citation

If you find this repository helpful, please cite:

```bibtex
@inproceedings{zhao2025eseg,
  title     = {ESEG: Event-Based Segmentation Boosted by Explicit Edge-Semantic Guidance},
  author    = {Zhao, Yucheng and Lyu, Gengyu and Li, Ke and Wang, Zihao and Chen, Hao and Yang, Zhen and Deng, Yongjian},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```


