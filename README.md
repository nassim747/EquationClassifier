# Equation Classifier with Deep Learning

This repository contains a deep learning project that predicts the results of simple equations displayed in grayscale images. Each image (50x80 pixels) contains two digits \( A \) and \( B \), and the goal is to classify the result of \( A + B \). The model is implemented using PyTorch, leveraging Convolutional Neural Networks (CNNs) and advanced techniques like residual connections and attention mechanisms.

---

## Project Overview

Mathematical equations are visually encoded as grayscale images. The objective is to process these images using a CNN-based model and predict the numerical result. The project demonstrates the application of deep learning to solve a practical OCR (Optical Character Recognition) problem with numerical computations.

---

## Key Features

- **Input**: Grayscale images (50x80 pixels) representing a simple equation \( A + B \), where \( A, B \in \{1, 2, 3, 4\} \).
- **Output**: Predicted result of the equation \( A + B \) (ranging from 2 to 8).
- **Model**: Custom CNN with residual blocks, attention mechanisms, and dropout layers for robust performance.
- **Framework**: PyTorch with data augmentation and advanced training techniques like learning rate scheduling.

---

## Technical Highlights

1. **CNN Architecture**:
   - Residual connections for better gradient flow.
   - Attention mechanism to focus on relevant features.
   - Dropout layers to prevent overfitting.

2. **Training Details**:
   - Optimizer: AdamW with weight decay.
   - Scheduler: Cosine Annealing Warm Restarts for dynamic learning rate adjustment.
   - Data augmentation applied selectively during training.

3. **Evaluation**:
   - Metrics: Accuracy and F1 score.
   - Validation set used to tune hyperparameters and monitor performance.

---


## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/equation-classifier.git
   cd equation-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   Open `project_overview.ipynb` in Jupyter Notebook or JupyterLab and execute the cells sequentially.
