# ICMA: Large Language Models are In-Context Molecule Learners

The official repository of **ICMA** - "Large Language Models are In-Context Molecule Learners".

**Paper**: [arXiv:2403.04197](https://arxiv.org/abs/2403.04197)

[IEEE TKDE Version](https://ieeexplore.ieee.org/abstract/document/10948482/)

## 📖 Documentation

- **[中文文档 / Chinese Documentation](README_CN.md)** - 完整的中文使用指南和项目介绍
- **[English Documentation](README_EN.md)** - Complete English documentation and project introduction

## 🚀 Quick Links

### Pre-trained Models
* **ICMA-Galactica-125M-M2C** (Molecule-to-Caption): [🤗 HuggingFace](https://huggingface.co/phenixace/ICMA-Galactica-125M-M2C)
* **ICMA-Galactica-125M-C2M** (Caption-to-Molecule): [🤗 HuggingFace](https://huggingface.co/phenixace/ICMA-Galactica-125M-C2M)

### Core Files
* **[icma_train.py](icma_train.py)** - Training script for ICMA
* **[inference.py](inference.py)** - Inference script for model evaluation
* **[dataset.py](dataset.py)** - Dataset loading and processing
* **[utils.py](utils.py)** - Utility functions
* **[naive_test.py](naive_test.py)** - Evaluation script

### Data & Evaluation
* **[data/](data/)** - Datasets (ChEBI-20, PubChem324k, MoleculeNet)
* **[evaluations/](evaluations/)** - Evaluation metrics and tools

## ✨ Key Features

- 🧬 **Task Support**: Mol2Cap, Cap2Mol, and molecular property prediction
- 📊 **Multiple Datasets**: ChEBI-20, PubChem324k, MoleculeNet
- 🔍 **Retrieval-Augmented Learning**: GNN, BM25, SentenceBERT, and more
- 🚀 **Efficient Training**: LoRA adapters for parameter-efficient fine-tuning

## 🏃 Quick Start

```bash
# Training Mol2Cap task
python icma_train.py \
    --base_model "facebook/galactica-125m" \
    --data_folder "./data/ChEBI-20/raw/" \
    --output_dir "./ckp/galactica-125M/mol2cap/" \
    --task "mol2cap" \
    --retrieval \
    --n_shot 1 \
    --m2c_method "gnn"

# Inference
python inference.py \
    --base_model "facebook/galactica-125m" \
    --adapter_path "./ckp/galactica-125M/mol2cap/checkpoint-8000/" \
    --data_folder "./data/ChEBI-20/raw/" \
    --task "mol2cap" \
    --retrieval \
    --n_shot 1 \
    --m2c_method "gnn" \
    --batch_infer
```

For detailed instructions, please refer to:
- **[中文文档](README_CN.md)** for Chinese users
- **[English Documentation](README_EN.md)** for detailed usage

## 📰 News
* ✅ **2026-01-21**: Complete codebase is now available!
* 🎉 **2025**: **Accepted by IEEE TKDE**
* ✅ **2024**: We release two versions of ICMA with Galactica-125M backbone
* 📝 **Preprint Paper**: [Large Language Models are In-Context Molecule Learners](https://arxiv.org/abs/2403.04197)


## 📚 Citation

If you use ICMA in your research, please cite our paper:

```bibtex
@article{li2025large,
  title={Large language models are in-context molecule learners},
  author={Li, Jiatong and Liu, Wei and Ding, Zhihao and Fan, Wenqi and Li, Yuqiang and Li, Qing},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  publisher={IEEE}
}
```

For our previous work (MolReGPT):

```bibtex
@article{li2024empowering,
  title={Empowering molecule discovery for molecule-caption translation with large language models: A chatgpt perspective},
  author={Li, Jiatong and Liu, Yunqing and Fan, Wenqi and Wei, Xiao-Yong and Liu, Hui and Tang, Jiliang and Li, Qing},
  journal={IEEE transactions on knowledge and data engineering},
  volume={36},
  number={11},
  pages={6071--6083},
  year={2024},
  publisher={IEEE}
}
```
