# Tree-Species-Classification-for-point-clouds
This repo aimed to present the Attribute-Aware Cross-Branch Transformer (AACB) for Individual Tree Species Classification and more widely used models for point clouds in tree species classification.

# Attribute-Aware Cross-Branch Transformer (AACB) for Individual Tree Species Classification

> Official implementation of:  
> **Wang, L., Lu, D., Xu, L., Robinson, D. T., Tan, W., Xie, Q., … & Li, J. (2024). Individual tree species classification using low-density airborne multispectral LiDAR data via attribute-aware cross-branch transformer. _Remote Sensing of Environment_, 315, 114456.**

---

## Overview

Low-density airborne multispectral LiDAR (MSL) provides valuable information for large-scale forest inventories but poses challenges due to sparse sampling, noisy distribution, and cross-site heterogeneity.  

We propose the **Attribute-Aware Cross-Branch Transformer (AACB)** — a dual-branch deep learning model that jointly encodes **geometric** (xyz, normals) and **spectral/radiometric** (intensity, multispectral channels) features, with cross-branch attention to strengthen feature interaction. The AACB demonstrates superior robustness under limited point densities and achieves improved generalization across forest plots.

---

## Key Features

- **Dual-branch fusion** of geometric and attribute channels  
- **Attribute-aware cross-attention** to enhance separability in low-density conditions  
- **Robust performance** under small point budgets (1k points or fewer)  
- **Reproducible training and inference pipeline** with voting-based prediction and detailed confusion matrix reporting  

---

## Repository Structure
```bash
project/
├─ train.py # training & validation (saves best_oa / best_macc / best_f1)
├─ inference.py # batch inference with --vote_times & resume support
├─ data_util
    └─ModelNetDataLoader.py # dataset loaders ()
├─ ErrorMatrix.py # confusion matrix (OA, mAcc, F1, PrettyTable output)
├─ provider.py # data augmentation utilities
├─ models/ # AACB and baseline models
    ├─
    ├─
└─ data/ # dataset root (see Dataset Layout)
```
---

## Dataset Layout
```bash
data/ModelNet40_liked_dataset/
├─ Tree_names.txt
├─ train.txt
├─ validate.txt
├─ test.txt
├─ AbiesAlba/
│  ├─ AbiesAlba_3021.txt
│  └─ ...
├─ EucalyptusMiniata/
│  └─ ...
└─ ...
```
- Tree_names.txt: species list (order defines class IDs)
- train.txt / validate.txt/ test.txt: sample filenames relative to species folders
- Each .txt contains points with multiple channels (xyz + attributes) no header
  
```
#like: x y z nx ny nz attribute1 attribute2 attribute3
-1.00830587 -0.13363360 -0.11359782 0.16809449 -0.48270576 -0.85949950 0.01413491 0.22723206 0.70960816 
0.43969413 0.34936640 -3.31659782 0.64334566 0.03696523 -0.76468290 0.00626230 0.03936303 0.09822866 
0.04369413 -0.02463360 2.92740218 0.02843276 0.80189664 0.59678585 0.00035785 0.02021829 0.07335838 
2.20269413 0.09536640 -0.92159782 0.64704469 0.06066600 -0.76003474 0.00536769 0.05671855 0.17534443 
```

---

## Installation

Requirements (Python ≥ 3.8, PyTorch ≥ 2.0):

```bash
pip install torch torchvision torchaudio
pip install numpy scipy tqdm prettytable matplotlib pyyaml scikit-learn laspy open3d
```

---

## Citation

Please cite our work if you use this code or method:

@article{Wang2024AACB,
  title   = {Individual tree species classification using low-density airborne multispectral LiDAR data via attribute-aware cross-branch transformer},
  author  = {Wang, L. and Lu, D. and Xu, L. and Robinson, D. T. and Tan, W. and Xie, Q. and Li, J.},
  journal = {Remote Sensing of Environment},
  volume  = {315},
  pages   = {114456},
  year    = {2024}
}

---
## License

This repository is released under the GPL-3.0 License (see LICENSE).
The dataset and any derived data are intended for research and academic use only.
Commercial use is prohibited.

---
## Contact

For questions or collaboration:
[Lanying Wang] – [lanying.wang@uwaterloo.ca]
