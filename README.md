# TinyBERT

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.1%2B-red" alt="PyTorch">
  <!-- <img src="https://img.shields.io/badge/PRs-welcome-blue" alt="PRs Welcome"> -->
</p>

---

A minimal BERT-like transformer model for **Semantic Search** corresponding to some query, implemented from scratch in PyTorch.

---

## 🚀 Features
- Customizable transformer architecture
- Tokenizer integration
- Training and inference scripts
- Two-phase training: NLI classification and semantic search
- Used MNRL [(Multiple Negative Rankings Loss)](https://medium.com/@aisagescribe/multiple-negative-ranking-loss-mnrl-explained-5b4741e38d8f) for semantic search training
- Easy to use and extend

---

## 📊 Model Specifications
| Specification         | Value      |
|----------------------|------------|
| Encoder Layers       | 8          |
| Embedding Dimension  | 768        |
| Context Length       | 350        |
| Attention Heads      | 12         |
| Feedforward Dim      | 1600       |
| Vocabulary Size      | 30,000     |
| Special Tokens       | [UNK], [CLS], [SEP], [PAD] |
| Segments             | 2          |
| Dropout              | 0.1        |

---

## 🖼️ Model Architecture


![Model arch](model_arc.png)

<sub>
<b>Figure:</b> BERT model architecture (source: <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>, Devlin et al., 2018)
</sub>

---

## 📦 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/TinyBERT.git
   cd TinyBERT
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Download Pretrained Weights

> **Note:** The model weights (`model.pt`) are not included in this repository due to size constraints.

- Download the pretrained model weights from [Google Drive](https://drive.google.com/file/d/1ubKrR8eBtyJWYEPhhRy9_Pf4DXU_CEhw/view?usp=sharing) and place `model.pt` in the project root directory.

---

## 🛠️ Usage

- **Training:**
  ```bash
  python train.py
  ```
- **Inference:**
  ```bash
  python inference.py
  ```

---

## 📚 Datasets & Training Details
- The model is trained in two phases:
  1. **Classifier head on top of Transformer** trained on [NLI dataset](https://huggingface.co/datasets/sentence-transformers/all-nli) for 3-class classification.
  2. **Only the transformer** trained on [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions) for semantic search.
- The model was trained for only 3 epochs for phase 1 and only 1 epoch for phase 2 due to Colab's GPU limits.

---

## 💡 Notes
- If you wish to use your own data or train from scratch, see `train.py` and `dataset.py` for details.
- This repository features a simple tiny implementation of an encoder only model inspired by BERT.

---

## 🤝 Contributing
Suggestions and feedback are most welcome!

---


