# APTOS 2019 Diabetic Retinopathy Classification

Deep learning pipeline for detecting and grading diabetic retinopathy (DR) from retinal fundus images, built on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) Kaggle dataset. The project covers three progressively harder tasks:

1. **Binary classification** — DR vs. no DR.
2. **Multiclass classification** — DR severity grading (0–4) with self-supervised contrastive pre-training on an auxiliary ocular disease dataset.
3. **Membership inference attack** — a privacy analysis that probes whether a sample was part of the training set using Monte Carlo augmentation features.

All notebooks are written to run on Google Colab with the dataset stored on Google Drive and a GPU runtime.

## Repository layout

```
.
├── binary_classifier.ipynb           # ResNet18 binary classifier (DR vs no DR)
├── multiclass_classifier.ipynb       # SimCLR-style pre-training + 5-class fine-tuning
├── APTOS_membership_inference.ipynb  # Monte Carlo membership inference attack
└── data/
    ├── train.csv                     # id_code, diagnosis (0–4) for 3,662 images
    └── test.csv                      # id_code for held-out Kaggle test set
```

The image folders (`train_images/`, `test_images/`) and the original Kaggle zip are gitignored — you need to download them yourself (see [Setup](#setup)).

## Dataset

- **Primary**: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection). 3,662 labeled training images graded on the international DR scale:
  - `0` — No DR
  - `1` — Mild
  - `2` — Moderate
  - `3` — Severe
  - `4` — Proliferative DR
- **Auxiliary (pre-training)**: [ODIR-5K Ocular Disease Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k). A cropped subset of ~2,000 images is used for self-supervised pre-training of the backbone.

The label distribution is heavily imbalanced toward class 0, which motivated tackling the binary task first.

## Methods

### Shared preprocessing

- `crop_retina` removes the black border around each fundus image by thresholding the grayscale intensity and cropping to the non-black bounding box (with padding) — this prevents the model from learning border artifacts.
- Images are resized to 224×224 and normalized with ImageNet statistics.
- Training-time augmentations: horizontal flip, small rotations, and color jitter.

### 1. Binary classifier (`binary_classifier.ipynb`)

- Labels remapped to `0` (no DR) and `1` (any DR), which produces a roughly balanced dataset.
- Stratified 64/16/20 train/val/test split (`random_state=42`).
- ResNet18 (ImageNet-pretrained), final FC replaced with a 2-way head.
- Trained end-to-end with Adam (`lr=1e-4`), cross-entropy loss, 20 epochs, batch size 16.
- Best checkpoint selected by validation loss; reported metrics include accuracy, ROC AUC, sensitivity, and specificity.

### 2. Multiclass classifier (`multiclass_classifier.ipynb`)

A two-stage pipeline designed to mitigate class imbalance and limited labeled data:

**Stage 1 — Self-supervised pre-training (SimCLR-style).**
- Backbone: ResNet18 with the FC layer removed, followed by a 2-layer projection head (`512 → 256 → 128`).
- Each image produces two strongly augmented views (random resized crop, flips, color jitter, grayscale, Gaussian blur). The NT-Xent contrastive loss pulls views of the same image together and pushes other views apart.
- Trained on ~2,000 cropped images from the ODIR-5K dataset for 10 epochs, Adam (`lr=1e-4`), batch size 32.

**Stage 2 — Supervised fine-tuning.**
- Pre-trained backbone is loaded, parameters frozen, and a new `Linear(512, 5)` head is trained on the APTOS labels.
- Same 64/16/20 stratified split as the binary task, but stratified on the full 5-class label.
- Evaluation uses accuracy and **Quadratic Weighted Kappa** (the official APTOS competition metric), which penalizes predictions that are further from the true severity grade.

### 3. Membership inference attack (`APTOS_membership_inference.ipynb`)

A black-box privacy analysis of the fine-tuned multiclass model:

- **Members** = the training split used to fit the model; **non-members** = the val + test splits.
- For each image, `T` augmented views are passed through the trained model and Monte Carlo statistics are computed:
  - Mean / std of the true-class softmax probability
  - Mean / std of the predictive entropy
  - Mean / std of the per-sample cross-entropy loss
- A logistic regression attack model is trained on these 6 features to predict membership. Performance is reported via accuracy, ROC AUC, and the full ROC curve, with per-feature histograms comparing member vs. non-member distributions.

The intuition: trained-on images tend to have higher confidence, lower entropy, and lower loss under augmentation than unseen images, and a simple classifier can exploit that gap.

## Setup

The notebooks were developed against Google Colab. To run them:

1. **Get the data**
   - Place a Kaggle API token at `~/.kaggle/kaggle.json` (the first cells of each notebook handle uploading and permissions in Colab).
   - Download and unzip `aptos2019-blindness-detection.zip` into `data/`.
   - For Stage 1 pre-training, also download the [ODIR-5K dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) and pre-crop a subset into `data/cropped_pretrain_images/` using `crop_retina`.

2. **Mount Drive (Colab)**
   The notebooks expect the project at `/content/drive/MyDrive/aptos_project` with a `data/` subfolder. Adjust `PROJECT_DIR` if running locally.

3. **Dependencies** (Colab has most of these pre-installed)
   - `torch`, `torchvision`
   - `numpy`, `pandas`, `scikit-learn`
   - `opencv-python`, `Pillow`
   - `matplotlib`, `tqdm`

4. **Hardware**
   A CUDA-capable GPU is strongly recommended. All notebooks auto-select `cuda` when available and fall back to `cpu`.

## Reproducing results

Run the notebooks in this order:

1. `binary_classifier.ipynb` — sanity-check that the pipeline works end-to-end on the balanced binary task. Saves `best_resnet18_binary.pth`.
2. `multiclass_classifier.ipynb` — runs Stage 1 (pre-training, saves `pretrain_model_weights.pth`) and Stage 2 (fine-tuning, saves `model_weights.pth`), then evaluates on the held-out test split.
3. `APTOS_membership_inference.ipynb` — loads `model_weights.pth` and runs the attack pipeline.

Splits use `random_state=42` throughout for reproducibility.

## Notes and limitations

- The Kaggle test set has no public labels, so all reported metrics use an internal stratified split of the labeled training set.
- The membership inference attack uses a small sample (`max_samples=100` per group by default) for speed — increase `T` and `max_samples` for tighter estimates.
- The pre-training stage assumes you have manually cropped and saved the ODIR-5K subset to `data/cropped_pretrain_images/`; the crop helper is the same `crop_retina` defined in each notebook.
