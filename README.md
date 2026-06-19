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
python setup.py install
cd ../knn
python setup.py install
cd ..
```

If your environment already includes compiled extensions, this step can be skipped.

## Dataset

DOGraspNet follows the GraspNet-1Billion benchmark protocol. Please download and organize the dataset according to the official GraspNet instructions.

Recommended layout:

```text
data/
`-- graspnet/
    |-- scenes/
    |-- models/
    |-- dex_models/
    |-- grasp_label/
    `-- collision_label/
```

After downloading the dataset, update the dataset path in your training/testing configuration or command scripts.

## Training

The repository provides shell scripts for convenient training.

```bash
bash command_train_re.sh
```

For KNN-based training:

```bash
bash command_train_kn.sh
```

You can also call the training entry point directly:

```bash
python train.py
```

Before launching training, check the following items:

- Dataset root path is correct.
- CUDA extensions are compiled.
- Batch size fits your GPU memory.
- Checkpoint and log directories are writable.

## Evaluation

Run the provided testing scripts:

```bash
bash command_test_re.sh
```

For KNN-based testing:

```bash
bash command_test_kn.sh
```

Collision-aware evaluation scripts are also provided:

```bash
bash command_test_re_pengzhuangjiance.sh
bash command_test_kn_pengzhuangjiance.sh
```

Compute AP and APu metrics:

```bash
python get_AP_and_APu.py
```

## Inference and Visualization

Visualize grasp predictions on benchmark scenes:

```bash
python infer_vis_grasp.py
```

Single-object visualization:

```bash
python infer_vis_grasp_singleObject.py
```

Additional visualization entry:

```bash
python infer_vis_grasp_wupeng.py
```

## Results

Please replace the placeholder values below with the final numbers from the accepted paper.

### GraspNet-1Billion Benchmark

| Method | Seen AP | Similar AP | Novel AP | APu | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | TBD | TBD | TBD | TBD | Reproduced baseline |
| DOGraspNet | TBD | TBD | TBD | TBD | Ours |

### Ablation Study

| Setting | Seen AP | Similar AP | Novel AP | Key Difference |
| --- | ---: | ---: | ---: | --- |
| w/o offset-driven aggregation | TBD | TBD | TBD | Removes local offset reasoning |
| w/o multi-modal fusion | TBD | TBD | TBD | Uses limited modality input |
| Full DOGraspNet | TBD | TBD | TBD | Complete model |

## Model Zoo

| Checkpoint | Dataset | Metric | Download |
| --- | --- | --- | --- |
| DOGraspNet | GraspNet-1Billion | TBD | Coming soon |

## TODO

- [ ] Release pretrained checkpoints.
- [ ] Add detailed dataset preparation instructions.
- [ ] Add final benchmark tables from the paper.
- [ ] Add qualitative visualization figures.
- [ ] Add demo video and project page link.

## Citation

If this project is helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{dograspnet2026,
  title     = {A Multi-Modal Framework with Offset-Driven Local Feature Aggregation for 6-DoF Grasp Pose Estimation},
  author    = {TODO},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## Acknowledgements

This project builds upon the GraspNet benchmark and related open-source 6-DoF grasp pose estimation frameworks. We sincerely thank the authors and contributors of these projects for advancing robotic grasping research.
