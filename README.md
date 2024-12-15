# ESEG: Event-based Segmentation Boosted by Explicit Edge-Semantic Guidance

 
*A novel framework for event-based semantic segmentation.*

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Introduction

Event-based Semantic Segmentation (ESS) is a groundbreaking approach to tackling challenges like motion blur and extreme lighting conditions, where conventional RGB cameras often fall short. **ESEG** introduces a novel method that leverages **explicit edge-semantic supervision** to enhance segmentation accuracy for sparse and noisy event data.

Our approach is based on two key innovations:
1. **Semantic Edge Labels (SELSAM)**: Automatically generated edge-semantic labels for better supervision.
2. **Density-Aware Dynamic-Window Cross-Attention Fusion (D2CAF)**: A robust feature fusion method optimized for edge-dense regions.

ESEG sets new benchmarks on the **DSEC-Semantic** and **DDD17** datasets, demonstrating significant performance improvements over state-of-the-art methods.

---

## Features

- **High Precision**: Enhanced event data segmentation by focusing on reliable edge regions.
- **Explicit Supervision**: Utilizes semantic edge information for guiding model learning.
- **Advanced Fusion Module**: Combines dense and sparse features using dynamic window masking.
- **State-of-the-Art Performance**: Outperforms existing methods on multiple datasets.

---

## Installation

### Prerequisites
- Python >= 3.7
- PyTorch >= 1.10
- CUDA (optional for GPU acceleration)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ESEG.git
   cd ESEG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare datasets:
   - Download the [DSEC-Semantic](https://link-to-dsec-dataset) and [DDD17](https://link-to-ddd17-dataset) datasets.
   - Follow instructions in `datasets/README.md` to organize the files.

---

## Usage

### Training
To train the ESEG model:
```bash
python train.py --dataset DSEC --config configs/eseg.yaml
```

### Evaluation
To evaluate the pre-trained model:
```bash
python evaluate.py --model checkpoints/eseg_best.pth --dataset DSEC
```

### Visualization
To visualize segmentation outputs:
```bash
python visualize.py --input path_to_event_data --output path_to_results
```

---

## Results

ESEG achieves state-of-the-art performance on popular ESS benchmarks:

| Dataset      | Backbone | Acc (%) | mIoU (%) |
|--------------|----------|---------|----------|
| DSEC         | MiT-b0   | 90.22   | 55.93    |
| DSEC         | MiT-b1   | 91.47   | 57.55    |
| DDD17        | MiT-b0   | 89.64   | 57.01    |
| DDD17        | MiT-b1   | 90.68   | 59.97    |

See the **Experiments** section of our [paper](link_to_paper) for more details.

---

## Citation

If you use this project, please cite:
```
@article{eseg2024,
  title={ESEG: Event-based Segmentation Boosted by Explicit Edge-Semantic Guidance},
  author={Anonymous},
  journal={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

