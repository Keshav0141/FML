# Foundations of Machine Learning – Assignment 3  
**Author:** Aryan Prasad  
**Roll No.:** DA25M007  

This assignment implements a complete spam–ham email classification pipeline using hand-written machine learning models, TF–IDF feature engineering, and evaluation across multiple algorithms. Models include Naive Bayes, Logistic Regression, Perceptron, k-NN, and SVMs.

---

# Dataset Description

The dataset contains raw email text labeled as spam or ham.

After preprocessing:
- Total samples: 4993  
- Ham: 4300  
- Spam: 693  
- Removed invalid/empty rows: 579  

Emails are converted into sparse TF–IDF vectors for modelling.

---

# Preprocessing and Feature Engineering

Preprocessing steps include:

### 1. Text Cleaning
- Convert to lowercase  
- Remove URLs  
- Remove punctuation and symbols  
- Remove digits  
- Normalise whitespace  

### 2. Tokenisation
Whitespace tokenisation, removing tokens of length 1 or less.

### 3. TF–IDF Feature Extraction
Each email is transformed into a TF–IDF vector and L2-normalised.

### 4. Top-K Feature Selection (5000 words)
The top 5000 most informative words are selected using log-odds scoring between spam and ham frequencies.

### 5. Full Pipeline

RAW EMAILS → CLEANING → TOKENISATION → TF–IDF → TOP-K FEATURES → TRAIN MODELS → SAVE NB MODEL → predict.py CLASSIFIES EMAILS

---

# Models Implemented

### 1. Multinomial Naive Bayes  
Uses word frequencies with Laplace smoothing and performs extremely well on sparse TF-IDF data.

### 2. Logistic Regression  
Trained using mini-batch gradient descent with L2 regularisation.

### 3. Perceptron  
A simple linear classifier updated based on prediction errors.

### 4. k-Nearest Neighbours  
Uses Euclidean distance in high-dimensional TF–IDF space.

### 5. Linear & RBF SVM  
Linear SVM performs near-optimally due to near-linear separability of TF–IDF vectors.  
RBF SVM provides no major improvement.

---

# High-Dimensional Behaviour

- Naive Bayes works well because independence assumptions fit sparse word-count data.  
- k-NN performs poorly due to distance concentration in high dimensions.  
- Linear SVM is very strong due to the geometry of TF–IDF vectors.  
- Logistic Regression and Perceptron give moderate but consistent results.

---

# Model Evaluation

### Accuracy and F1 Scores

| Model | Accuracy (%) | F1 Score |
|-------|--------------|----------|
| Naive Bayes | 97.00 | 0.9476 |
| Logistic Regression | 91.29 | 0.8676 |
| Perceptron | 92.99 | 0.8879 |
| k-NN | 84.38 | 0.6549 |
| Linear SVM | 96.10 | 0.9357 |
| RBF SVM | 94.69 | 0.9072 |

Best scratch model: **Naive Bayes**  
Best overall: **Linear SVM**

---

# Prediction System

`predict.py` loads the saved Naive Bayes model and classifies new emails using the same preprocessing pipeline.
