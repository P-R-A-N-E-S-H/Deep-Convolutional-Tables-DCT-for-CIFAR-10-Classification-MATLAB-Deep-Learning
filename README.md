# Convolution Tables: An Alternative to CNN-Based Feature Learning

This repository contains the implementation and experiments for **Convolution Tables (CT)** — an efficient alternative to traditional Convolutional Neural Networks (CNNs) for feature learning and image classification.

Instead of using heavy convolution operations, Convolution Tables use **binary decision functions and lookup tables** to extract features, making the system faster and more computationally efficient, especially on CPU and low-resource devices.

---

## 📌 Project Motivation

Convolutional Neural Networks (CNNs) rely on expensive multiply-accumulate operations in convolution layers.  
Convolution Tables (CTs) replace these operations with:

- Binary feature extraction
- Table lookups
- Summation of learned responses

This results in:

- ⚡ Faster inference  
- 💻 Lower compute cost  
- 📦 Smaller and simpler models  
- 🚀 Competitive accuracy compared to CNNs  

---

## 📂 Repository Structure

Convolution-Tables-An-Alternative-to-CNN-Based-Feature-Learning/
├── MATLAB/ # MATLAB implementation
├── Python + Marimo/ # Python implementation and experiments
├── UID part/ # UI / visualization part
├── Presentation (ppt)/ # Project presentation
├── README.md # This file


---

## 🧠 What Are Convolution Tables?

Instead of learning convolution filters like CNNs:

1. Image patches are passed through **binary tests**
2. The outputs form a **binary code**
3. This binary code is used as an **index into a lookup table**
4. The values from multiple tables are **summed to form features**
5. The final features are used for classification

So, CTs **approximate convolutional feature learning without convolutions**.

---

## ✨ Features

- ✅ Convolution-free feature learning
- ✅ Fast and lightweight inference
- ✅ Works on CPU efficiently
- ✅ MATLAB and Python implementations
- ✅ Suitable for CIFAR-10 / MNIST style datasets

---

## 🛠️ Requirements

### MATLAB
- MATLAB R2021a or later
- Deep Learning Toolbox (optional but recommended)

### Python
Install dependencies:

```bash
pip install numpy torch torchvision matplotlib
```
---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sandhiyaa031/Convolution-Tables-An-Alternative-to-CNN-Based-Feature-Learning.git
cd Convolution-Tables-An-Alternative-to-CNN-Based-Feature-Learning
```

🏁 How to Run
MATLAB Version

1. Open the MATLAB/ folder in MATLAB.
2. Load the dataset (e.g., CIFAR-10 or MNIST).
3. Run the training script (example):
```bash
train_CT_model.m
```
4. Run the testing / evaluation script:
```bash
test_CT_model.m
```

Python Version
Run :
```
python train.py
```

## 📊 Results

| Dataset  | Model | Accuracy |
|----------|--------|----------|
| CIFAR-10 | CT     | 89.0 %   |
| CIFAR-10 | CNN    | 93.0 %   |

---

## 🧪 Applications

- Embedded AI
- Low-power devices
- Edge computing
- Fast image classification systems
- Research on convolution-free neural models

---

## 📚 Reference

Inspired by the research paper:

**"Convolution Tables: An Alternative to CNN-Based Feature Learning"**

---

## 👨‍💻 Author

**Pranesh M**  
Project developed as part of academic and research work.

---

## 📄 License

This project is licensed under the **MIT License**.
