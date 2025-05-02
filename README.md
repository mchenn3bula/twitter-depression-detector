# 🧠 Twitter Therapy: Depression Detection Using NLP and BERT

This project investigates how **Natural Language Processing (NLP)** can be applied to identify signs of **depression in tweets**, using BERT-based models and machine learning classifiers. Built on a 20,000-tweet dataset labeled for mental health indicators, this study demonstrates the role of AI in public health awareness and early intervention.

---

## 📘 Project Overview

- Predicts whether a tweet was written by a depressed user using **text analysis and machine learning**
- Evaluates models using **accuracy, precision, recall, and F1-score**
- Explores **bias mitigation** by avoiding keyword dependence (e.g. "depression")
- Tests both **fine-tuned BERT classifiers** and **BERT-based feature extractors + Random Forest**

---

## ✨ Key Features

- **Context-sensitive NLP modeling** using BERT
- **Data preprocessing**: URL removal, stopword filtering, normalization
- **Overfitting mitigation** via random keyword deletion
- **Model comparison**:
  - BERT Classifier (original and modified datasets)
  - BERT Tokenizer + Random Forest Classifier
- **Bayesian hyperparameter tuning** using Ray Tune

---

## 🧪 Methodology

### 1. BERT Tokenizer + Random Forest
- BERT generates tweet embeddings
- Embeddings passed to a Random Forest classifier
- Tuned with Ray Tune

### 2. Fine-tuned BERT Classifier
- BERT directly fine-tuned on tweet labels
- Tested with and without the word “depression”
- 99% probability of removing "depression" from depressed tweets

---

## 📊 Results

| Model Type                            | Accuracy | Precision | Recall | F1-score |
|--------------------------------------|----------|-----------|--------|----------|
| BERT Tokenizer + Random Forest       | 76.1%    | 75–77%    | 75–77% | 0.76     |
| BERT Classifier (w/ “depression”)    | **84.95%** | 84–86%    | 84–86% | 0.85     |
| BERT Classifier (w/o “depression”)   | 84.57%   | 84–85%    | 84–85% | 0.85     |

📉 False negatives reduced by ~44% with fine-tuned BERT compared to RF baseline.

---

## 📁 Dataset

- **20,000 tweets** from Kaggle
- Balanced classes (50% depressed, 50% non-depressed)
- Training: 14,000 tweets (with validation split)
- Testing: 6,000 tweets

---

## 🔍 OOV Word Handling

Words not seen during training are categorized:
- `<NUM>` for numbers
- `<CAP>` for capitalized words
- `<PLURAL>` for plurals
- `<PUNCT>` for punctuation
- `<UNK>` for all other unknowns

---

## 📦 Requirements

- Python 3.x
- `transformers`, `scikit-learn`, `pandas`, `numpy`, `ray[tune]`, `torch`

Install with:
```bash
pip install transformers scikit-learn pandas numpy torch ray[tune]
````

---

## 🚀 How to Run

1. Place the data files and notebooks in the same folder
2. Run the notebook `NLP_PROJECT.ipynb` to train and test models
3. Predictions and confusion matrices will be printed or saved in `output.txt`

---

## 📈 Future Work

* Expand dataset with broader and multilingual samples
* Use sequential modeling (RNN/Transformer history) for tweet timelines
* Include **multi-class sentiment detection** beyond binary depression tags
* Explore user-level classification using account history
* Collaborate with clinical professionals to enhance real-world applicability

---

## 👨‍🔬 Team

* Zhengyi Chen · [zc2018@nyu.edu](mailto:zc2018@nyu.edu)
* Andrew Qin · [aq2053@nyu.edu](mailto:aq2053@nyu.edu)
* Vanessa Chen · [vc1530@nyu.edu](mailto:vc1530@nyu.edu)
* Supervised by Prof. Adam Meyers

---

## 📜 License

This project is released under the **MIT License**.
