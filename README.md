# DOGraspNet

<p align="center">
  <b>A Multi-Modal Framework with Offset-Driven Local Feature Aggregation for 6-DoF Grasp Pose Estimation</b>
</p>

<p align="center">
  <a href="https://github.com/huamo555/DOGraspNet"><img src="https://img.shields.io/badge/Project-DOGraspNet-2f80ed.svg" alt="project"></a>
  <img src="https://img.shields.io/badge/IROS-2026-ff6f00.svg" alt="IROS 2026">
  <img src="https://img.shields.io/badge/Python-99%25-3776ab.svg" alt="Python">
  <img src="https://img.shields.io/badge/Task-6--DoF%20Grasp%20Pose%20Estimation-8a2be2.svg" alt="task">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-See%20LICENSE-green.svg" alt="license"></a>
</p>

<p align="center">
  <a href="#highlights">Highlights</a> |
  <a href="#method-overview">Method</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#training">Training</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#citation">Citation</a>
</p>

> Official implementation of **DOGraspNet**, accepted by **IROS 2026**.  
> This repository provides training, inference, visualization, and evaluation code for multi-modal 6-DoF grasp pose estimation in cluttered scenes.

## News

- **2026-06**: Repository released.
- **2026-06**: DOGraspNet accepted to **IROS 2026**.

## Highlights

- **Offset-driven local feature aggregation** for learning geometry-aware grasp features around candidate regions.
- **Multi-modal perception** that combines complementary cues for robust grasp pose estimation.
- **6-DoF grasp prediction** in cluttered scenes, targeting practical robotic manipulation.
- **Complete experimental pipeline** including training, testing, collision-aware evaluation, AP/APu computation, and grasp visualization.
- **GraspNet-compatible workflow** for benchmark evaluation and comparison with existing methods.

## Method Overview

DOGraspNet focuses on strengthening local geometric reasoning for 6-DoF grasp pose estimation. The model aggregates offset-driven neighborhood features and fuses multi-modal information to produce reliable grasp candidates in complex scenes.

```mermaid
flowchart LR
    A["RGB-D / Point Cloud Input"] --> B["Backbone Feature Extraction"]
    B --> C["Offset-Driven Local Feature Aggregation"]
    C --> D["Multi-Modal Feature Fusion"]
    D --> E["Grasp Pose Prediction"]
    E --> F["Collision Checking"]
    F --> G["6-DoF Grasp Results"]
```

## Repository Structure

```text
DOGraspNet/
|-- dataset/                         # Dataset loading and preprocessing
|-- doc/                             # Figures, documents, and supplementary assets
|-- graspnetAPI/                     # GraspNet evaluation utilities
|-- knn/                             # KNN CUDA/C++ extensions
|-- pointnet2/                       # PointNet++ operators and modules
|-- SE_resUnet.py                    # SE-ResUNet module
|-- backbone_resunet14.py            # Backbone network
|-- collision_detector.py            # Collision checking
|-- data_utils.py                    # Data processing utilities
|-- get_AP_and_APu.py                # AP/APu metric computation
|-- graspnet.py                      # Main network definition
|-- infer_vis_grasp*.py              # Inference and visualization scripts
|-- label_generation.py              # Label generation
|-- loss.py / loss_utils.py          # Training losses
|-- modules.py                       # Core network blocks
|-- train.py                         # Training entry point
|-- test.py                          # Testing entry point
`-- command_*.sh                     # Ready-to-run training/testing scripts
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/huamo555/DOGraspNet.git
cd DOGraspNet
```

### 2. Create the environment

```bash
conda create -n dograspnet python=3.8 -y
conda activate dograspnet
pip install -r requirements.txt
```

### 3. Build CUDA extensions

The repository uses custom operators under `pointnet2/` and `knn/`. Build them before training or evaluation.

```bash
cd pointnet2
