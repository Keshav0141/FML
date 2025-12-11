
import os
import numpy as np
from naive_bayes import NaiveBayes
from preprocess import clean_text 

def load_saved_model(path="spam_nb_model.npz"):
    import numpy as np, os
    from naive_bayes import NaiveBayes

    if not os.path.exists(path):
        return None

    d = np.load(path, allow_pickle=True)

    def unwrap(x):
        if isinstance(x, np.ndarray):
            if x.dtype == object and x.size == 1:
                return x[0]
        return x

    nb = NaiveBayes(alpha=float(np.squeeze(d["alpha"])))
    nb.classes = d["classes"].tolist()
    nb.vocab = unwrap(d["vocab"])
    nb.word_counts = unwrap(d["word_counts"])
    nb.class_totals = unwrap(d["class_totals"])
    nb.class_docs = unwrap(d["class_docs"])
    nb.total_docs = int(np.squeeze(d["total_docs"]))
    nb.vocab_size = int(np.squeeze(d["vocab_size"]))
    return nb


def predict_test_folder(test_folder="test", model=None):
    if model is None:
        model = load_saved_model()
        if model is None:
            raise ValueError("No trained model provided and no saved model found.")

    files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(".txt")])
    if not files:
        print("No .txt files found in", test_folder)
        return

    for fname in files:
        path = os.path.join(test_folder, fname)
        try:
            with open(path, "r", encoding="utf8", errors="ignore") as fh:
                raw = fh.read()
        except Exception as e:
            print(fname, 0)  
            continue

        tokens = clean_text(raw)  
        pred = model.predict([tokens])[0]  
        out_label = +1 if int(pred) == 1 else 0
        print(fname, out_label)

if __name__ == "__main__":
    nb = load_saved_model()
    if nb is None:
        print("No saved model found. Please run training (main.py) to create model.")
    else:
        predict_test_folder("test", model=nb)
