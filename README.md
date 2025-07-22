# EyeCLIP: A Multimodal Visual–Language Foundation Model for Computational Ophthalmology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch\&logoColor=white)](https://pytorch.org/)

> Official repository for **EyeCLIP**, a vision-language foundation model designed specifically for **multi-modal ophthalmic image analysis**.
> 📝 **Paper**: [*A Multimodal Visual–Language Foundation Model for Computational Ophthalmology* (npj Digital Medicine, 2025)](https://www.nature.com/articles/s41746-025-01772-2.pdf)

---

## 🔍 Overview
**EyeCLIP** adapts the CLIP (Contrastive Language–Image Pretraining) architecture to address the unique challenges of ophthalmology. It incorporates self-supervised learning, multi-modal image contrastive learning, and hierarchical keyword-guided vision-language supervision. These innovations empower EyeCLIP to achieve **zero-shot disease recognition**, **cross-modal retrieval**, and **efficient fine-tuning** across a wide range of ophthalmic and systemic conditions.

---

## ✨ Key Features

* 🧠 **Multimodal Support**
  Natively pretrained on 11 ophthalmic modalities using one encoder, including:

  * Color Fundus Photography (CFP)
  * Optical Coherence Tomography (OCT)
  * Fundus Fluorescein Angiography (FFA)
  * Indocyanine Green Angiography (ICGA)
  * Fundus Autofluorescence (FAF)
  * Slit Lamp Photography
  * Ocular Ultrasound (OUS)
  * Specular Microscopy
  * External Eye Photography
  * Corneal Photography
  * RetCam Imaging

* 🔗 **CLIP-based Vision–Language Pretraining**
  Tailored adaptation of OpenAI’s CLIP for ophthalmic imaging and medical-language semantics.

* 🚀 **Zero-Shot Generalization**
  Classifies both **ophthalmic** and **systemic** diseases using natural language prompts—without task-specific fine-tuning.

* 🧩 **Versatile and Adaptable**
  Easily fine-tuned for downstream diagnostic tasks, including multi-label classification, systemic disease prediction, and rare disease diagnosis.

---

## 🗞️ News

* **2025-07**: Initial release of pre-trained EyeCLIP model weights
* **2025-06**: Paper accepted by *npj Digital Medicine*
* **2025-03**: Public release of EyeCLIP codebase

---

## ⚙️ Installation

Set up the environment using conda and pip:

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
git clone https://github.com/Michi-3000/EyeCLIP.git
cd EyeCLIP
```

---

## 🎯 Pretrained Weights

| Model Name           | Description                                                    | Download Link                      |
| -------------------- | -------------------------------------------------------------- | ---------------------------------- |
| `EyeCLIP_multimodal` | Multimodal foundation model trained on diverse ophthalmic data | [🔗 Google Drive (placeholder)](https://drive.google.com/file/d/1LS2VqYDJB8zzjkplRaWSx9v2WHguAiUg/view?usp=sharing) |

---

## 📁 Dataset Preparation

To prepare datasets for pretraining or downstream evaluation:

1. **Download** datasets referenced in the paper.
2. **Organize** them into the following format:

```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels.csv
```

* `labels.csv` should follow the format:

```
impath,class
/path/to/image1.jpg,0
/path/to/image2.jpg,1
```

---

## 🚀 Quick Start

### 🔍 Zero-Shot Evaluation

```bash
python zero_shot.py \
    --model EyeCLIP_base \
    --data_path ./your_dataset \
    --text_prompts "normal retina,diabetic retinopathy,glaucoma"
```

---

### 🩺 Fine-Tuning

#### Ophthalmic Disease Classification

```bash
bash scripts/opthalmic_loop.sh
```

Or use the Python version:

```bash
current_time=$(date +"%Y-%m-%d-%H%M")
for epoch in "eyeclip"; do
  checkpoint="checkpoint-${epoch}.pth"
  for name in 'IDRiD' 'OCTID' 'PAPILA' 'Retina' 'JSIEC' 'MESSIDOR2' 'Aptos2019' 'Glaucoma_Fundus' 'OCTDL' 'Retina Image Bank'; do
    CUDA_VISIBLE_DEVICES=2 python main_finetune_opthal.py \
      --now_epoch $epoch \
      --test_num 5 \
      --data_name $name \
      --batch_size 16 \
      --world_size 1 \
      --model vit_large_patch16 \
      --epochs 50 \
      --blr 5e-3 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.2 \
      --output_dir "output_dir_downstream/all_dataset_$current_time" \
      --data_path "" \
      --finetune "$checkpoint" \
      --input_size 224
  done
done
```

#### Systemic Disease Classification

```bash
bash scripts/cls_chro.sh
```

Or with custom parameters:

```bash
current_time=$(date +"%Y-%m-%d-%H%M")
for epoch in "eyeclip"; do
  checkpoint="checkpoint-${epoch}.pth"
  CUDA_VISIBLE_DEVICES=0 python main_finetune_chro.py \
    --now_epoch $epoch \
    --test_num 5 \
    --data_name chro \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --data_path data/public \
    --output_dir "output_dir_downstream/chronicdisease5/$current_time" \
    --finetune "$checkpoint" \
    --input_size 224
done
```

---

## 🧪 Pretraining from Scratch

To pretrain EyeCLIP on your own dataset:

```bash
python CLIP_ft_all_1enc_all.py
```

---

## 📚 Scripts and Utilities

| Script                    | Purpose                                              |
| ------------------------- | ---------------------------------------------------- |
| `main_finetune_opthal.py` | Fine-tuning on ophthalmic disease datasets           |
| `main_finetune_chro.py`   | Fine-tuning for systemic (chronic) disease detection |
| `zero_shot.py`            | Zero-shot classification using language prompts      |
| `retrieval.py`            | Cross-modal image–text retrieval                     |

---

## 📖 Citation

If you use EyeCLIP in your research, please cite:

```bibtex
@article{shi2025multimodal,
  title={A multimodal visual--language foundation model for computational ophthalmology},
  author={Shi, Danli and Zhang, Weiyi and Yang, Jiancheng and Huang, Siyu and Chen, Xiaolan and Xu, Pusheng and Jin, Kai and Lin, Shan and Wei, Jin and Yusufu, Mayinuer and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={381},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

---

## 🤝 Acknowledgements

This project builds upon prior open-source contributions, especially:

* [CLIP](https://github.com/openai/CLIP) – Contrastive Language–Image Pretraining by OpenAI
* [MAE](https://github.com/facebookresearch/mae) – Masked Autoencoders by Facebook AI Research

We thank the open-source community and the medical imaging research ecosystem for their invaluable contributions.
