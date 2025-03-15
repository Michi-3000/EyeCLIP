# EyeCLIP: A Visual-Language Foundation Model for Multi-Modal Ophthalmic Image Analysis

This repository contains the code for the article **"EyeCLIP: A Visual-Language Foundation Model for Multi-Modal Ophthalmic Image Analysis"**. The paper presents a novel approach for leveraging vision-language pretraining to enhance ophthalmic image analysis across multiple modalities. You can read the full article [here](https://arxiv.org/pdf/2409.06644).

## Overview

EyeCLIP builds upon the CLIP (Contrastive Language-Image Pretraining) framework, adapting it specifically for ophthalmic image analysis. It is designed to:
- Integrate and analyze multiple ophthalmic imaging modalities (e.g., Fundus, OCT, FA, ICGA, etc.).
- Perform zero-shot and fine-tuned classification for both ophthalmic and systemic diseases.
- Enable cross-modal retrieval between images and textual descriptions.

The code in this repository is largely based on the publicly available implementation from the original CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).

```bibtex
@misc{radford2021learningtransferablevisualmodels,
      title={Learning Transferable Visual Models From Natural Language Supervision},
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.00020},
}
```

## Installation

To set up the environment, please follow the installation instructions provided in the official CLIP repository: [https://github.com/openai/CLIP](https://github.com/openai/CLIP).

### Dependencies
Ensure that you have the following dependencies installed:
- Python 3.8+
- PyTorch (with CUDA support for GPU training)
- OpenAI CLIP package

## Dataset Preparation

To prepare the dataset for pretraining and downstream tasks, follow these steps:

1. **Download the Dataset**
   - Use the links provided in the article to download the publicly available ophthalmic imaging datasets.

2. **Organize the Data**
   - Ensure the dataset is structured as follows:
     ```
     dataset_root/
     ├── images/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   ├── ...
     ├── labels.csv
     ```
   - The `labels.csv` file should be formatted with at least the following columns:
     ```csv
     impath,class
     /path/to/image1.jpg,0
     /path/to/image2.jpg,1
     ```
     where:
     - `impath`: Absolute path to the image file.
     - `class`: Integer label representing the class of the image.

## Pretraining

To pretrain the EyeCLIP model on ophthalmic image datasets, run the following command:

```bash
python CLIP_ft_all_1enc_all.py
```

## Downstream Tasks

EyeCLIP can be used for various ophthalmic image analysis tasks. Below are the available downstream tasks with corresponding scripts to run them.

### 1. Zero-Shot Testing
To evaluate the pretrained model without fine-tuning, run:
```bash
python zero_shot.py
```

### 2. Ophthalmic Disease Classification
To fine-tune the model for ophthalmic disease classification, run:
```bash
bash scripts/cls_opthal.sh
```

### 3. Systemic Disease Classification
To fine-tune the model for systemic disease classification, run:
```bash
bash scripts/cls_chro.sh
```

### 4. Image-Text Retrieval
To perform cross-modal retrieval, run:
```bash
python retrieval.py
```

## Citation

If you use this repository or find our work helpful, please consider citing our paper:
```bibtex
@article{your_paper_citation,
  title={EyeCLIP: A Visual-Language Foundation Model for Multi-Modal Ophthalmic Image Analysis},
  author={Your Name and Co-Authors},
  journal={arXiv},
  year={2024},
  url={https://arxiv.org/pdf/2409.06644}
}
```

