# 🚀 Convolution Tables (CTs): An Alternative to CNN-Based Feature Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MATLAB](https://img.shields.io/badge/MATLAB-R2024b-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)

</p>

---

## 📖 Overview

**Convolution Tables (CTs)** are a lightweight alternative to traditional **Convolutional Neural Networks (CNNs)** for image feature extraction.

Instead of performing computationally expensive convolution operations, CTs use:

- Binary feature extraction
- Lookup tables
- Learned response aggregation

This significantly reduces computation while maintaining competitive classification performance on image datasets.

---

## ✨ Why Convolution Tables?

Traditional CNNs rely on:

- Large convolution kernels
- Millions of floating-point operations
- Heavy GPU computation

Convolution Tables replace these with efficient binary operations and lookup tables.

### Advantages

- ⚡ Extremely fast inference
- 💻 Low computational cost
- 📦 Lightweight model
- 🧠 Memory efficient
- 🚀 CPU-friendly
- 🔋 Suitable for edge devices
- 🎯 Competitive accuracy

---

# 🏗 Project Architecture

```
Input Image
      │
      ▼
Binary Feature Extraction
      │
      ▼
Binary Code Generation
      │
      ▼
Lookup Table Indexing
      │
      ▼
Feature Aggregation
      │
      ▼
Classifier
      │
      ▼
Prediction
```

---

# 📂 Repository Structure

```
Convolution-Tables-An-Alternative-to-CNN-Based-Feature-Learning/

│
├── MATLAB/
│   ├── train_CT_model.m
│   ├── test_CT_model.m
│   └── utility_functions/
│
├── Python/
│   ├── train.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
│
├── Marimo/
│   └── experiments/
│
├── UI/
│   └── visualization/
│
├── Presentation/
│   └── Project_Presentation.pptx
│
├── Results/
│   ├── graphs/
│   └── images/
│
├── README.md
└── LICENSE
```

---

# 🧠 How Convolution Tables Work

Unlike CNNs, CTs avoid convolution filters.

The workflow consists of five simple stages:

### Step 1

Extract local image patches.

↓

### Step 2

Apply multiple binary tests.

↓

### Step 3

Generate a binary code.

↓

### Step 4

Use the binary code as an index into learned lookup tables.

↓

### Step 5

Aggregate responses from multiple tables to produce image features.

Finally, these features are passed to a classifier for prediction.

---

# 🚀 Features

- ✅ Convolution-free feature learning
- ✅ Lightweight architecture
- ✅ CPU optimized
- ✅ MATLAB implementation
- ✅ Python implementation
- ✅ Easy experimentation
- ✅ Modular project structure
- ✅ Research-friendly code

---

# 📊 Supported Datasets

- CIFAR-10
- CIFAR-100
- MNIST
- Custom image datasets

---

# 🛠 Requirements

## MATLAB

- MATLAB R2021a or newer
- Deep Learning Toolbox *(optional)*

---

## Python

Install dependencies

```bash
pip install numpy torch torchvision matplotlib
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/sandhiyaa031/Convolution-Tables-An-Alternative-to-CNN-Based-Feature-Learning.git

cd Convolution-Tables-An-Alternative-to-CNN-Based-Feature-Learning
```

---

# ▶ Running the Project

## MATLAB

Open MATLAB and execute

```matlab
train_CT_model.m
```

Evaluate the model

```matlab
test_CT_model.m
```

---

## Python

Train the model

```bash
python train.py
```

Evaluate

```bash
python test.py
```

---

# 📈 Experimental Results

| Dataset | Model | Accuracy |
|----------|--------|-----------|
| CIFAR-10 | CT | **89.0%** |
| CIFAR-10 | CNN | **93.0%** |

---

# 📊 Performance Comparison

| Metric | CNN | Convolution Tables |
|---------|-----|-------------------|
| Convolution Operations | ✓ | ✗ |
| Lookup Tables | ✗ | ✓ |
| Binary Features | ✗ | ✓ |
| CPU Efficiency | Medium | High |
| Memory Usage | High | Low |
| Inference Speed | Medium | Very Fast |

---

# 💡 Applications

- 🤖 Embedded AI
- 📱 Mobile Vision
- 🌐 Edge Computing
- 🚗 Autonomous Systems
- 🛰 IoT Devices
- 🔬 Deep Learning Research
- 🖥 Real-time Image Classification

---

# 📚 Research Inspiration

This project is inspired by the paper

> **Convolution Tables: An Alternative to CNN-Based Feature Learning**

The implementation demonstrates how lookup-table-based feature learning can replace convolution operations while maintaining strong classification performance.

---

# 🔮 Future Improvements

- Vision Transformers comparison
- GPU optimization
- Quantization support
- ONNX export
- More benchmark datasets
- Explainable AI visualization

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

**Pranesh M**

B.Tech Artificial Intelligence

Amrita Vishwa Vidyapeetham

📧 Email: your-email@example.com

🔗 GitHub: https://github.com/yourusername

---

## ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it

📢 Share it with others

---

<p align="center">

<b>Made with ❤️ for Deep Learning Research</b>

</p>
