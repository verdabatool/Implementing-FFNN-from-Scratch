# Two-Layer MLP from Scratch (NumPy)

This project implements a two-hidden-layer feedforward neural network from scratch using only NumPy (no automatic differentiation libraries). The model is trained for multi-class classification and analyzed from both optimization and interpretability perspectives.

---

## 📌 Project Overview

The implementation includes:

- Forward propagation
- Cross-entropy loss computation
- Manual backpropagation
- Batch gradient descent training
- Sigmoid and ReLU activation functions
- Gradient magnitude analysis across layers
- Gradient-based feature attribution (Aggarwal Section 2.8)
- Train / Validation / Test evaluation

---

## 🧠 Architecture

- Input dimension: 12 features  
- Hidden Layer 1: 32 neurons  
- Hidden Layer 2: 16 neurons  
- Output layer: 4 classes  

Activations tested:
- Sigmoid
- ReLU

---

## 📊 Results Summary

| Activation | Train Acc | Val Acc | Test Acc |
|------------|-----------|---------|----------|
| ReLU       | 0.7778    | 0.5688  | 0.7831   |
| Sigmoid    | 0.8298    | 0.8349  | 0.5631   |

Sigmoid demonstrated stronger optimization performance and better generalization compared to ReLU under the selected hyperparameters.

---

## 🔍 Gradient Analysis

We analyzed:

- Average gradient magnitudes across hidden layers
- Input feature gradients for feature importance ranking

Findings:
- ReLU preserved gradient flow across layers.
- Sigmoid exhibited attenuation between layers.
- Model relied heavily on `amenity_score` and room type features.
- High feature dominance contributed to generalization sensitivity.

---

## ⚙️ Running the Project

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib
