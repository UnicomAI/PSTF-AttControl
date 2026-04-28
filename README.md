# PSTF-AttControl

> **Per-subject-tuning-free personalized image generation with controllable face attributes** > Accepted at *Image and Vision Computing*

This repository contains the official implementation for the paper **"PSTF-AttControl: Per-subject-tuning-free personalized image generation with controllable face attributes"**. 

Our method enables high-fidelity, personalized image generation without the need to fine-tune the model for each specific subject, while maintaining precise and controllable manipulation over specific facial attributes.

---

## 🛠️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/UnicomAI/PSTF-AttControl.git
   cd PSTF-AttControl
   ```
2. Install the **PreciseControl**[https://github.com/rishubhpar/PreciseControl] dependency

---

## 📂 Data Preparation

Before training, you need to prepare the dataset and extract the necessary face and style features.

1. **Download the FFHQ Dataset:**
   Download the [Flickr-Faces-HQ (FFHQ) dataset](https://github.com/NVlabs/ffhq-dataset) and place the images in the appropriate directory (e.g., `./ffhq-dataset/`).

2. **Extract Features:**
   Run the data preparation script to extract face embeddings and style latent codes. This will process the images and generate the required `.npy` and `output_data.json` files.
   ```bash
   cd PSTF-AttControl
   python prepare.py
   ```

---

## 🚀 Training

To train the attention-modified InstantID model with style constraints, run the provided shell script:

```bash
sh train_instantId_sdxl_style_atten.sh
```
*(Make sure to adjust the paths to your dataset, base models, and output directories inside the `.sh` file before running).*

---

## 🎨 Inference

To generate personalized images with manipulated face attributes (e.g., smile, age, eyeglasses), run the inference script. The script automatically handles face alignment, masking, and attribute injection.

```bash
python style_instantid_mask_infer.py
```

---

## 📖 Citation

If you find this code or our paper useful for your research, please consider citing:

```bibtex
@article{liu2025pstf,
  title={PSTF-AttControl: Per-subject-tuning-free personalized image generation with controllable face attributes},
  author={Liu, Xiang and Liu, Zhaoxiang and Hu, Huan and Wang, Zipeng and Chen, Ping and Chen, Zezhou and Wang, Kai and Lian, Shiguo},
  journal={Image and Vision Computing},
  pages={105790},
  year={2025},
  publisher={Elsevier}
}
```
