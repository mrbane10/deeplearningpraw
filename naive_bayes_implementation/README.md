# Naive Bayes Spam Email Classifier

A from-scratch implementation of Naive Bayes for email spam detection, demonstrating the critical importance of numerical stability in machine learning.

## 🎯 Key Achievement

**Improved accuracy from 84.82% to 99.21%** by implementing log-space computations to handle numerical underflow.

## 📊 Results

| Implementation | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Standard NB   | 84.82%   | 59.57%    | 98.03% | 74.11%   |
| **Log-space NB** | **99.21%** | **98.42%** | 98.03% | **98.22%** |

**Dataset**: 5,728 emails (23.88% spam, 76.12% ham)

## 🔍 The Problem & Solution

**Problem**: Standard Naive Bayes suffers from numerical underflow when multiplying many small probabilities, causing them to become zero and breaking classification.

**Solution**: Log-space computations convert multiplication to addition, maintaining numerical precision:
- Standard: P(Email|Class) = ∏ P(word|Class) → underflows to 0
- Log-space: log P(Email|Class) = Σ log P(word|Class) → stays precise

## 🛠️ Features

- **From-scratch Naive Bayes** with Laplace smoothing
- **Text preprocessing** (tokenization, stopword removal)
- **Numerical stability** comparison
- **Interactive demo** with 100% accuracy on test cases
- **Comprehensive evaluation** metrics

## 🚀 Quick Start

```python
# Install dependencies
pip install numpy pandas nltk

# Run classification
email = "Congratulations! You've won $1000! Click here now!"
result = classify_email(email)  # Returns: "Spam"
