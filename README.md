# 🧠 Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
**NeurIPS 2025 • Spotlight**

📄 **Paper:** _link coming soon (arXiv / OpenReview)_  

We show that large pretrained Vision Transformers (especially self-supervised ones like DINOv2) naturally learn **object binding** — they internally represent whether two patches belong to the same object (IsSameObject) without any explicit object-level supervision.

## ⚙️ Installation

We provide a single setup script that installs all dependencies (PyTorch, mmseg, DinoV2, xformers, etc.). During setup, `libs/dinov2/requirements.txt` is replaced with `requirements_dino.txt` so that the install uses the exact dependency versions tested in this repo.

```bash
bash scripts/setup.sh
```

## 📂 Dataset Structure (ADE20K)

```
<DATASET_ROOT>/
    ADE20K_2021_17_01/
        images/ADE/training/
        images/ADE/validation/
        objectInfo150.csv   # <-- MUST BE PLACED HERE
        objects.txt
        index_ade20k.mat
        index_ade20k.pkl
```
`objectInfo150.csv` must be placed inside the dataset root.  
It maps ADE20K’s full label space into the canonical **150-class** version used for probing.


## 🔧 Config (W&B)

Edit the wandb fields in `cfgs/config.yaml`.

## 🚀 Training & Evaluation

All training/evaluation entry points are provided in the `scripts/` directory.

| Command | Description |
|--------|-------------|
| `bash train.sh` | Train a probe |
| `bash eval.sh`  | Evaluate a trained checkpoint |

### Probe types

You select which probing task to run via the `mode=` argument:

| `mode=`          | Description                     |
|------------------|----------------------------------|
| `train`          | Pairwise (IsSameObject) probing  |
| `train_class`    | Pointwise **class** probing      |
| `train_identity` | Pointwise **identity** probing   |

Choose the probe architecture via:
```
probe.mode=linear / diag_quadratic / quadratic
```

## 👀 Visualization

To visualize **layer-wise IsSameObject scores**, run:
```bash
python main.py mode=vis
```









