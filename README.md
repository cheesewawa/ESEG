# ESEG: Event-Based Segmentation Boosted by Explicit Edge-Semantic Guidance

## Project Overview

ESEG is a novel event-based semantic segmentation (ESS) framework that leverages **explicit edge-semantic supervision** to guide dense semantic feature extraction from sparse and noisy event data. Unlike traditional methods that rely solely on complex network architectures to learn edge awareness implicitly, ESEG introduces semantic edge labels to **explicitly inform the model where valuable features reside**, enhancing segmentation reliability—especially in high-speed or low-light environments.

Event cameras, due to their high temporal resolution and dynamic range, provide great potential in scenes where RGB sensors often fail. However, event data is inherently sparse and lacks sufficient context for pixel-level semantic tasks. ESEG addresses this challenge by exploiting the fact that event triggers are highly correlated with motion-induced edges.

---

## Core Components and Technical Insights

### 1. **Edge-Semantic Guidance**
ESEG introduces a branch that learns **semantic edge maps** from event data. Since standard datasets do not offer semantic edge labels, the authors propose **SELSAM** (Semantic Edge Label with SAM), a pipeline based on the Segment Anything Model (SAM) to generate clean, edge-focused supervision.

- **SELSAM** avoids noisy internal texture edges by leveraging SAM's robust segmentation masks and localized semantic assignment.
- These edge maps serve as explicit guidance for the segmentation model, steering its attention to reliable edge-rich regions.

### 2. **Multi-Branch Architecture**
The model is structured in three branches:
- **Edge-Semantic Branch**: Learns semantic edge features using DFF (Dynamic Feature Fusion) with a ResNet34 backbone.
- **Dense-Semantic Branch**: Learns full-scene semantics using SegFormer as backbone.
- **Fusion Branch**: Combines the above via the novel D2CAF module.

### 3. **D2CAF Module (Density-Aware Dynamic-Window Cross-Attention Fusion)**
This module enables adaptive fusion of sparse edge features and dense semantic features:
- **Density Indicator Matrix (DIM)**: Measures information density using entropy and distance metrics to guide attention.
- **Dynamic Window Masking**: Adjusts attention span based on second-order gradients of edge density—wider attention for edge-rich regions, narrower for smooth zones.
- **Cross-Attention Fusion**: Merges edge and dense features through dynamic masking to avoid feature interference from noisy or irrelevant regions.

---

## Experimental Results

ESEG was evaluated on two standard ESS benchmarks: **DSEC-Semantic** and **DDD17**, achieving **state-of-the-art (SOTA)** performance in both accuracy and mIoU.

### DSEC-Semantic Dataset
| Method        | Accuracy | mIoU  |
|---------------|----------|-------|
| ESEG-B (MiT-b0) | 90.22%   | 55.93 |
| ESEG-L (MiT-b1) | 91.47%   | 57.55 |

Outperforms traditional ESS methods like Ev-SegNet, EvSegFormer, HMNet, and even RGB-trained ESS models like ESS-Sup.

### DDD17 Dataset
| Method        | Accuracy | mIoU  |
|---------------|----------|-------|
| ESEG-B (MiT-b0) | 89.64%   | 57.01 |
| ESEG-L (MiT-b1) | 90.68%   | 59.97 |

Notably, ESEG outperforms RGB-event hybrid methods like EvDistill and demonstrates better edge refinement in visually challenging scenes.

---

## Citation

If you find this work useful in your research, please cite it as:

```bibtex
@inproceedings{zhao2025eseg,
  title     = {ESEG: Event-Based Segmentation Boosted by Explicit Edge-Semantic Guidance},
  author    = {Yucheng Zhao and Gengyu Lyu and Ke Li and Zihao Wang and Hao Chen and Zhen Yang and Yongjian Deng},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
