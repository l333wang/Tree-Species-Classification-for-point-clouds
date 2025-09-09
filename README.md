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


---

## Installation

Requirements (Python ≥ 3.8, PyTorch ≥ 2.0):

```bash
pip install torch torchvision torchaudio
pip install numpy scipy tqdm prettytable matplotlib pyyaml scikit-learn laspy open3d
