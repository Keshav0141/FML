import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Confusion Matrix heatmap
# -----------------------------------------------------------
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    labels = ["Ham (0)", "Spam (+1)"]
    plt.xticks(np.arange(2), labels)
    plt.yticks(np.arange(2), labels)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Comparison of ALL  MODELS
# -----------------------------------------------------------
def compare_all_models(
    acc_nb, acc_lr, acc_perc, acc_knn, acc_lsvm, acc_rbf,
    f1_nb, f1_lr, f1_perc, f1_knn, f1_lsvm, f1_rbf
):

    models = ["NB", "LR", "Perceptron", "kNN", "Linear SVM", "RBF SVM"]
    acc = [acc_nb*100, acc_lr*100, acc_perc*100, acc_knn*100, acc_lsvm*100, acc_rbf*100]
    f1  = [f1_nb*100, f1_lr*100, f1_perc*100, f1_knn*100, f1_lsvm*100, f1_rbf*100]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, acc, width, label="Accuracy")
    plt.bar(x + width/2, f1, width, label="F1 Score")

    plt.ylabel("Percentage (%)")
    plt.title("Model Comparison Across All Algorithms")
    plt.xticks(x, models)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Vocab Size Plot
# -----------------------------------------------------------
def plot_vocab_sizes(full_size, k):
    plt.figure(figsize=(6, 4))
    sizes = [full_size, k]
    labels = ["Full Vocab", f"Top {k}"]

    plt.bar(labels, sizes, color=['grey', 'blue'])
    plt.title("Effect of Top-K Feature Selection")
    plt.ylabel("Vocabulary Size")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# TF-IDF Histogram
# -----------------------------------------------------------
def plot_tfidf_distribution(X):
    plt.figure(figsize=(6, 4))
    plt.hist(X.flatten(), bins=50, color='green', alpha=0.7)
    plt.title("TF-IDF Weight Distribution")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Logistic Regression Learning Curve
# -----------------------------------------------------------
def plot_learning_curve(loss_history):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color='purple')
    plt.title("Logistic Regression Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
