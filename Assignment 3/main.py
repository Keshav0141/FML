# ===============================================
#   Spam Classifier | Multiple Models
#   Naive Bayes, Logistic Regression, Perceptron,
#   kNN, Linear SVM, RBF SVM
# ===============================================

import numpy as np
from sklearn.svm import LinearSVC, SVC

from preprocess import (
    load_data,
    split_data,
    select_top_k_vocab,
    build_tfidf_matrix
)

from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegressionScratch
from perceptron import PerceptronScratch
from knn import KNN
from metrics import confusion_matrix, precision_recall_f1

from plots import (
    plot_confusion_matrix,
    compare_all_models,
    plot_vocab_sizes,
    plot_tfidf_distribution,
    plot_learning_curve
)

# ----------------------------------------------------------
# 1. Load & split dataset
# ----------------------------------------------------------
print("Loading and cleaning dataset...")
X, y = load_data("spam_ham_dataset.csv")
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"Loaded {len(X)} rows after cleaning.")

# ----------------------------------------------------------
# 2. Naive Bayes (Scratch)
# ----------------------------------------------------------

print("\nTraining Naive Bayes (Scratch)...")
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)

print("Predicting with Naive Bayes...")
y_pred_nb = nb.predict(X_test)

cm_nb = confusion_matrix(y_test, y_pred_nb)
precision_nb, recall_nb, f1_nb, acc_nb = precision_recall_f1(cm_nb)

print("==============================================================")
print("NAIVE BAYES RESULTS")
print("==============================================================")
print(f"Accuracy : {acc_nb*100:.2f}%")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall   : {recall_nb:.4f}")
print(f"F1-score : {f1_nb:.4f}")
print("Confusion Matrix:")
print(cm_nb)

# Save model for predict.py
np.savez("spam_nb_model.npz",
         alpha=np.array(nb.alpha),
         classes=np.array(nb.classes),
         vocab=np.array([nb.vocab], dtype=object),
         word_counts=np.array([nb.word_counts], dtype=object),
         class_totals=np.array([nb.class_totals], dtype=object),
         class_docs=np.array([nb.class_docs], dtype=object),
         total_docs=np.array([nb.total_docs]),
         vocab_size=np.array([nb.vocab_size]))

print("\nModel saved successfully as spam_nb_model.npz")

# ----------------------------------------------------------
# 3. Feature Extraction (Top-K + TF-IDF)
# ----------------------------------------------------------

top_k = 5000
print(f"\nSelecting top {top_k} tokens by log-odds...")
vocab_small = select_top_k_vocab(X_train, y_train, k=top_k)

print("Building TF-IDF matrices...")
X_train_tfidf, _ = build_tfidf_matrix(X_train, vocab_small)
X_test_tfidf, _ = build_tfidf_matrix(X_test, vocab_small)

# ----------------------------------------------------------
# 4. Logistic Regression (Scratch)
# ----------------------------------------------------------

print("\nTraining Logistic Regression (Scratch)...")
lr = LogisticRegressionScratch(
    lr=0.05,
    epochs=200,
    batch_size=64,
    lambda_reg=0.001
)
lr.fit(X_train_tfidf, y_train)

print("Predicting with Logistic Regression...")
y_pred_lr = lr.predict(X_test_tfidf)

cm_lr = confusion_matrix(y_test, y_pred_lr)
precision_lr, recall_lr, f1_lr, acc_lr = precision_recall_f1(cm_lr)

print("==============================================================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================================================")
print(f"Accuracy : {acc_lr*100:.2f}%")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall   : {recall_lr:.4f}")
print(f"F1-score : {f1_lr:.4f}")
print("Confusion Matrix:")
print(cm_lr)

# ----------------------------------------------------------
# 5. Perceptron (Scratch)
# ----------------------------------------------------------

print("\nTraining Perceptron (Scratch)...")
perc = PerceptronScratch(lr=0.01, epochs=50)
perc.fit(X_train_tfidf, y_train)

print("Predicting with Perceptron...")
y_pred_perc = perc.predict(X_test_tfidf)

cm_perc = confusion_matrix(y_test, y_pred_perc)
precision_perc, recall_perc, f1_perc, acc_perc = precision_recall_f1(cm_perc)

print("==============================================================")
print("PERCEPTRON RESULTS")
print("==============================================================")
print(f"Accuracy : {acc_perc*100:.2f}%")
print(f"Precision: {precision_perc:.4f}")
print(f"Recall   : {recall_perc:.4f}")
print(f"F1-score : {f1_perc:.4f}")
print("Confusion Matrix:")
print(cm_perc)

# ----------------------------------------------------------
# 6. k-NN (Scratch)
# ----------------------------------------------------------

print("\nTraining k-NN (Scratch)...")
knn = KNN(k=5)
knn.fit(X_train_tfidf, y_train)

print("Predicting with k-NN...")
y_pred_knn = knn.predict(X_test_tfidf)

cm_knn = confusion_matrix(y_test, y_pred_knn)
precision_knn, recall_knn, f1_knn, acc_knn = precision_recall_f1(cm_knn)

print("==============================================================")
print("k-NN RESULTS")
print("==============================================================")
print(f"Accuracy : {acc_knn*100:.2f}%")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall   : {recall_knn:.4f}")
print(f"F1-score : {f1_knn:.4f}")
print("Confusion Matrix:")
print(cm_knn)

# ----------------------------------------------------------
# 7. Linear SVM (Library allowed)
# ----------------------------------------------------------

print("\nTraining Linear SVM...")
linear_svm = LinearSVC()
linear_svm.fit(X_train_tfidf, y_train)

print("Predicting with Linear SVM...")
y_pred_lsvm = linear_svm.predict(X_test_tfidf)

cm_lsvm = confusion_matrix(y_test, y_pred_lsvm)
precision_lsvm, recall_lsvm, f1_lsvm, acc_lsvm = precision_recall_f1(cm_lsvm)

print("==============================================================")
print("LINEAR SVM RESULTS")
print("==============================================================")
print(f"Accuracy : {acc_lsvm*100:.2f}%")
print(f"Precision: {precision_lsvm:.4f}")
print(f"Recall   : {recall_lsvm:.4f}")
print(f"F1-score : {f1_lsvm:.4f}")
print("Confusion Matrix:")
print(cm_lsvm)

# ----------------------------------------------------------
# 8. RBF SVM
# ----------------------------------------------------------

print("\nTraining RBF SVM (may take time)...")
rbf_svm = SVC(kernel='rbf', gamma='scale', C=1.0)
rbf_svm.fit(X_train_tfidf, y_train)

print("Predicting with RBF SVM...")
y_pred_rbf = rbf_svm.predict(X_test_tfidf)

cm_rbf = confusion_matrix(y_test, y_pred_rbf)
precision_rbf, recall_rbf, f1_rbf, acc_rbf = precision_recall_f1(cm_rbf)

print("==============================================================")
print("RBF SVM RESULTS")
print("==============================================================")
print(f"Accuracy : {acc_rbf*100:.2f}%")
print(f"Precision: {precision_rbf:.4f}")
print(f"Recall   : {recall_rbf:.4f}")
print(f"F1-score : {f1_rbf:.4f}")
print("Confusion Matrix:")
print(cm_rbf)

# ----------------------------------------------------------
# 9. Plot ALL Models
# ----------------------------------------------------------

compare_all_models(
    acc_nb, acc_lr, acc_perc, acc_knn, acc_lsvm, acc_rbf,
    f1_nb, f1_lr, f1_perc, f1_knn, f1_lsvm, f1_rbf
)

plot_confusion_matrix(cm_nb, "Naive Bayes Confusion Matrix")
plot_confusion_matrix(cm_lr, "Logistic Regression Confusion Matrix")
plot_confusion_matrix(cm_perc, "Perceptron Confusion Matrix")
plot_confusion_matrix(cm_knn, "k-NN Confusion Matrix")
plot_confusion_matrix(cm_lsvm, "Linear SVM Confusion Matrix")
plot_confusion_matrix(cm_rbf, "RBF SVM Confusion Matrix")

plot_vocab_sizes(len(nb.vocab), top_k)
plot_tfidf_distribution(X_train_tfidf)
plot_learning_curve(lr.loss_history)
