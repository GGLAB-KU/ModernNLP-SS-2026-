# Session 1: Word Embeddings, Model Training, Derivatives and Back Propagation

1. **Word Embeddings & Creation:** How continuous, low-dimensional vectors capture semantic similarity compared to sparse one-hot encoding (featuring a custom CBOW implementation).
2. **Model Training Loop:** The macro-perspective of machine learning pipelines (Forward Pass $\rightarrow$ Loss $\rightarrow$ Zero Grad $\rightarrow$ Backward Pass $\rightarrow$ Optimization).
3. **Taking Derivatives:** Grounding the calculus of optimization into an intuitive geometric rate of change.
4. **Backpropagation:** The computational implementation of the Chain Rule to distribute error blame backward through a graph.
5. **Gradient Descent:** Using numeric gradients and learning rates ($\eta$) to physically adjust weight parameters.
6. **Evaluation Metrics:** Quantifying model generalization on unseen validation data.

---

## 💻 Code walkthroughs

### 📦 1. Custom Continuous Bag of Words (CBOW)
This module builds a tiny Word2Vec architecture from scratch using PyTorch to predict a target token given its surrounding context window. It pulls the internal embeddings lookup table and projects them into 2D space using Principal Component Analysis (PCA).
* **Key Visual Takeaway:** Watch words from distinct semantic clusters (*Animals*, *Coffee/Drinks*, *Tech/Coding*) naturally group into spatial neighborhoods based entirely on contextual distribution.

### 📉 2. Live Loss Optimization & Decision Boundaries
A self-contained network designed to solve a non-linear classification baseline. It includes:
* **Interactive Notebook Wrapper:** Uses `IPython.display` to update and render the loss function curve in real-time across epochs.
* **Boundary Deformation Map:** Animates how a completely chaotic, randomly initialized boundary matrix warps into an idealized non-linear separator as backpropagation converges.
* **Terminal Progress Bar:** Employs `tqdm` to pipe real-time loss tracking directly to your console standard output without display latency.

### 🔧 3. Pure Mathematical Manual Backpropagation
A zero-dependency Python script tracking an explicit forward and backward pass using primitive arithmetic operations. 
* Removes library abstractions to expose downstream errors, input amplification gates ($x$), and exact adjustments:
  $$\frac{\partial \mathcal{L}}{\partial w} = (\hat{y} - y) \cdot x$$

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install torch matplotlib scikit-learn tqdm IPython
conda activate your_venv

# The following creates a dummy CBOW embedding model and project some sample embeddings using Principal Component Analysis (PCA),
# a method that downprojects high dimensional vectors into small dimensional vector space. Help us during visualization.
python cbow.py

# Run the following. This script features a raw, zero-dependency Python implementation of a forward and backward pass that uses primitive
# arithmetic to clearly expose downstream errors, input amplification gates (x), and exact parameter updates without library abstractions.
python backpropagation.py

# The following script implements a dynamic computation graph that uses a localized numeric finite-difference method (numeric_grads) to explicitly
# calculate partial derivatives for every single node, enabling backpropagation to update the weights of a 2-layer neural network solving the non-linear XOR problem entirely from scratch.
python backpropagation_bonus.py

# This script implements an abstract computational graph (AnalyticGraph) using topological sorting to orchestrate an analytic backpropagation pipeline,
# map out explicit local partial derivative functions (backward_func) for math operations, and update parameters via gradient descent to train a multi-layer
# network on the XOR classification task completely from scratch.
python backpropagation_bonus_2.py
```

## ✏️ Bonus Resources
- [Lecture Slides](https://docs.google.com/presentation/d/1Ju7Q-t-Xx30ku_bG9UeVQz8WELbPi1tK4C38Za7mbfg/edit?usp=sharing)
- [Recording](https://drive.google.com/drive/folders/1x5C02JgcW2u5hXrD355hFOoYEbdN-X5-?usp=sharing)
- [Complete simplified backpropagation](https://youtu.be/8gFrim7cTzI?si=xhaAVL_Ut-QO58hG)