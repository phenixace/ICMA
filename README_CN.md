# ICMA: 大语言模型的上下文分子学习框架

ICMA (In-Context Molecule Adaptation) 是一个基于大语言模型（LLMs）的分子-文本双向翻译框架，通过上下文分子调优来提升大语言模型在分子发现任务上的性能。

**论文链接**: [Large Language Models are In-Context Molecule Learners](https://ieeexplore.ieee.org/abstract/document/10948482/)

## 📢 新闻

* 🎉 **已被 IEEE TKDE 接受**
* 我们发布了基于 Galactica-125M 的两个版本的 ICMA 模型，可通过 HuggingFace 访问：
  * [phenixace/ICMA-Galactica-125M-M2C](https://huggingface.co/phenixace/ICMA-Galactica-125M-M2C) (分子到描述)
  * [phenixace/ICMA-Galactica-125M-C2M](https://huggingface.co/phenixace/ICMA-Galactica-125M-C2M) (描述到分子)
* 完整的代码库现已发布！

## 🎯 项目简介

ICMA 专注于分子发现中的两个核心任务：

1. **分子理解 (Molecule Understanding)**: 给定一个分子，生成描述其结构、性质和功能的文本
2. **文本条件分子生成 (Text-conditioned Molecule Generation)**: 根据文本描述生成对应的分子结构

这两个任务分别对应：
- **Mol2Cap (Molecule2Caption)**: 为给定分子生成描述文本
- **Cap2Mol (Caption2Molecule)**: 根据描述文本生成对应的分子

## ✨ 主要特性

### 🧬 多任务支持
- **Mol2Cap**: 分子到文本描述
- **Cap2Mol**: 文本描述到分子
- **分子性质预测**: 支持多个 MoleculeNet 任务（BACE, BBBP, ClinTox, HIV, SIDER, Tox21, ToxCast）

### 📊 多数据集支持
- **ChEBI-20**: 化学实体生物学兴趣数据库子集（26,407 训练样本）
- **PubChem324k**: PubChem 大规模数据集（IUPAC 命名和分子描述）
- **MoleculeNet**: 多个分子性质预测任务

### 🔍 检索增强学习
- **多种检索方法**:
  - **Mol2Cap**: GNN 相似度、Morgan 指纹、随机检索
  - **Cap2Mol**: BM25、SentenceBERT、随机检索
- **Few-shot 学习**: 通过检索相似样本进行上下文学习
- **自动示例选择**: 智能选择最相关的训练样本作为示例

### 🚀 高效训练
- **LoRA 适配器**: 参数高效的模型微调
- **支持多种模型**: 支持 decoder-only 和 encoder-decoder 架构
- **灵活配置**: 支持 8-bit 量化、FP16 混合精度训练

## 📁 项目结构

```
ICMA/
├── 核心代码
│   ├── icma_train.py      # 训练主程序
│   ├── inference.py        # 推理主程序
│   ├── naive_test.py       # 简单测试脚本
│   ├── dataset.py          # 数据集加载模块
│   └── utils.py            # 工具函数模块
│
├── 数据处理 (data/)
│   ├── ChEBI-20/          # ChEBI-20数据集
│   ├── PubChem324k/       # PubChem数据集
│   └── MoleculeNet/       # MoleculeNet数据集
│
├── 评估指标 (evaluations/)
│   ├── text_translation_metrics.py    # 文本翻译指标
│   ├── mol_translation_metrics.py     # 分子翻译指标
│   ├── fingerprint_metrics.py         # 指纹相似度指标
│   └── ...
│
└── 脚本 (run_train.bash)   # 训练脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 2.0+
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- RDKit
- 其他依赖见 `requirements.txt`

### 安装

```bash
# 克隆仓库
git clone https://github.com/phenixace/ICMA.git
cd ICMA

# 安装依赖
pip install torch transformers peft datasets rdkit sentence-transformers
```

### 训练

```bash
# 使用检索增强的 Few-shot 学习训练 Mol2Cap 任务
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

# 训练 Cap2Mol 任务
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

### 推理

```bash
# 使用训练好的模型进行推理
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

### 评估

```bash
# 使用 naive_test.py 评估结果
python naive_test.py \
    --raw_folder "./data/ChEBI-20/raw/" \
    --target_folder "./predictions/" \
    --model "galactica-125M" \
    --ckp 8000 \
    --task "mol2cap"
```

## 🔬 技术细节

### 检索增强 Few-shot 学习

ICMA 通过以下方式提升性能：

1. **相似度检索**: 从训练集中检索与查询最相似的样本
2. **上下文构建**: 将检索到的样本作为 few-shot 示例构建提示
3. **参数微调**: 使用 LoRA 适配器对模型进行参数高效的微调

### 支持的检索方法

- **Mol2Cap**:
  - `gnn`: 基于 GNN 的分子相似度（默认）
  - `morgan`: Morgan 指纹相似度
  - `random`: 随机检索

- **Cap2Mol**:
  - `bm25`: BM25 文本检索（默认）
  - `sentencebert`: Sentence-BERT 语义相似度
  - `random`: 随机检索

## 📈 实验结果

ICMA 在多个数据集上取得了优异的性能：

- 在 ChEBI-20 数据集上，ICMA 在 Mol2Cap 和 Cap2Mol 任务上都超越了基线方法
- 在 MoleculeNet 性质预测任务上，ICMA 展现了良好的泛化能力

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📚 引用

如果您在研究中使用 ICMA，请引用我们的论文：

```bibtex
@article{li2025large,
  title={Large language models are in-context molecule learners},
  author={Li, Jiatong and Liu, Wei and Ding, Zhihao and Fan, Wenqi and Li, Yuqiang and Li, Qing},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  publisher={IEEE}
}
```

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

## 📄 许可证

本项目遵循相应的开源许可证。

## 📧 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

## 🙏 致谢

感谢所有为本项目做出贡献的研究者和开发者。
