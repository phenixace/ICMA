# ICMA: Large Language Models are In-Context Molecule Learners

ICMA (In-Context Molecule Adaptation) is a framework for bidirectional molecule-text translation using Large Language Models (LLMs), which improves LLM performance on molecule discovery tasks through in-context molecule tuning.

**Paper**: [Large Language Models are In-Context Molecule Learners](https://ieeexplore.ieee.org/abstract/document/10948482/)

## 📢 News

* 🎉 **Accepted by IEEE TKDE**
* We release two versions of ICMA with the backbone of Galactica-125M, accessible via HuggingFace:
  * [phenixace/ICMA-Galactica-125M-M2C](https://huggingface.co/phenixace/ICMA-Galactica-125M-M2C) (Molecule-to-Caption)
  * [phenixace/ICMA-Galactica-125M-C2M](https://huggingface.co/phenixace/ICMA-Galactica-125M-C2M) (Caption-to-Molecule)
* The complete codebase is now available!

## 🎯 Overview

ICMA focuses on two critical tasks in molecule discovery:

1. **Molecule Understanding**: Given a molecule, generate text describing its structure, properties, and functions
2. **Text-conditioned Molecule Generation**: Generate corresponding molecule structures based on text descriptions

These tasks correspond to:
- **Mol2Cap (Molecule2Caption)**: Generate descriptive text for a given molecule
- **Cap2Mol (Caption2Molecule)**: Generate corresponding molecules based on text descriptions

## ✨ Key Features

### 🧬 Multi-task Support
- **Mol2Cap**: Molecule to text description
- **Cap2Mol**: Text description to molecule
- **Molecular Property Prediction**: Support for multiple MoleculeNet tasks (BACE, BBBP, ClinTox, HIV, SIDER, Tox21, ToxCast)

### 📊 Multi-dataset Support
- **ChEBI-20**: Chemical Entities of Biological Interest subset (26,407 training samples)
- **PubChem324k**: Large-scale PubChem dataset (IUPAC naming and molecular descriptions)
- **MoleculeNet**: Multiple molecular property prediction tasks

### 🔍 Retrieval-Augmented Learning
- **Multiple Retrieval Methods**:
  - **Mol2Cap**: GNN similarity, Morgan fingerprint, random retrieval
  - **Cap2Mol**: BM25, SentenceBERT, random retrieval
- **Few-shot Learning**: Context learning through retrieved similar samples
- **Automatic Example Selection**: Intelligently select the most relevant training samples as examples

### 🚀 Efficient Training
- **LoRA Adapters**: Parameter-efficient model fine-tuning
- **Multiple Model Support**: Supports both decoder-only and encoder-decoder architectures
- **Flexible Configuration**: Supports 8-bit quantization, FP16 mixed precision training

## 📁 Project Structure

```
ICMA/
├── Core Code
│   ├── icma_train.py      # Training script
│   ├── inference.py        # Inference script
│   ├── naive_test.py       # Simple test script
│   ├── dataset.py          # Dataset loading module
│   └── utils.py            # Utility functions
│
├── Data Processing (data/)
│   ├── ChEBI-20/          # ChEBI-20 dataset
│   ├── PubChem324k/       # PubChem dataset
│   └── MoleculeNet/       # MoleculeNet datasets
│
├── Evaluation Metrics (evaluations/)
│   ├── text_translation_metrics.py    # Text translation metrics
│   ├── mol_translation_metrics.py     # Molecular translation metrics
│   ├── fingerprint_metrics.py         # Fingerprint similarity metrics
│   └── ...
│
└── Scripts (run_train.bash)   # Training scripts
```

## 🚀 Quick Start

### Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- RDKit
- Other dependencies (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/phenixace/ICMA.git
cd ICMA

# Install dependencies
pip install torch transformers peft datasets rdkit sentence-transformers
```

### Training

```bash
# Train Mol2Cap task with retrieval-augmented few-shot learning
python icma_train.py \
    --base_model "facebook/galactica-125m" \
    --data_folder "./data/ChEBI-20/raw/" \
    --output_dir "./ckp/galactica-125M/mol2cap/" \
    --task "mol2cap" \
    --retrieval \
    --n_shot 1 \
    --m2c_method "gnn" \
    --micro_batch_size 4 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-5

# Train Cap2Mol task
python icma_train.py \
    --base_model "facebook/galactica-125m" \
    --data_folder "./data/ChEBI-20/raw/" \
    --output_dir "./ckp/galactica-125M/cap2mol/" \
    --task "cap2mol" \
    --retrieval \
    --n_shot 1 \
    --c2m_method "bm25" \
    --micro_batch_size 4 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-5
```

### Inference

```bash
# Inference with trained model
python inference.py \
    --base_model "facebook/galactica-125m" \
    --adapter_path "./ckp/galactica-125M/mol2cap/checkpoint-8000/" \
    --data_folder "./data/ChEBI-20/raw/" \
    --task "mol2cap" \
    --output_dir "./predictions/galactica-125M/mol2cap/" \
    --retrieval \
    --n_shot 1 \
    --m2c_method "gnn" \
    --batch_infer
```

### Evaluation

```bash
# Evaluate results using naive_test.py
python naive_test.py \
    --raw_folder "./data/ChEBI-20/raw/" \
    --target_folder "./predictions/" \
    --model "galactica-125M" \
    --ckp 8000 \
    --task "mol2cap"
```

## 🔬 Technical Details

### Retrieval-Augmented Few-shot Learning

ICMA improves performance through:

1. **Similarity Retrieval**: Retrieve the most similar samples from the training set for the query
2. **Context Construction**: Build prompts using retrieved samples as few-shot examples
3. **Parameter Tuning**: Fine-tune the model with parameter-efficient LoRA adapters

### Supported Retrieval Methods

- **Mol2Cap**:
  - `gnn`: GNN-based molecular similarity (default)
  - `morgan`: Morgan fingerprint similarity
  - `random`: Random retrieval

- **Cap2Mol**:
  - `bm25`: BM25 text retrieval (default)
  - `sentencebert`: Sentence-BERT semantic similarity
  - `random`: Random retrieval

## 📈 Experimental Results

ICMA achieves excellent performance on multiple datasets:

- On ChEBI-20 dataset, ICMA outperforms baseline methods on both Mol2Cap and Cap2Mol tasks
- On MoleculeNet property prediction tasks, ICMA demonstrates good generalization capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

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


## 📄 License

This project follows the corresponding open source license.

## 📧 Contact

For questions, please contact us via:
- Submit a GitHub Issue
- Email the project maintainers

## 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to this project.
