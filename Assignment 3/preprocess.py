import numpy as np
import csv

def clean_text(text):
    text = text.lower()

    # Remove basic HTML tags
    html_tags = ["<br>", "<br/>", "<div>", "</div>", "<p>", "</p>"]
    for tag in html_tags:
        text = text.replace(tag, " ")

    # Replace URL-like patterns
    url_triggers = ["http", "www", ".com", ".net", ".org"]
    for ut in url_triggers:
        if ut in text:
            text = text.replace(ut, " url ")

    # Replace digits with token
    cleaned_chars = []
    for ch in text:
        if ch.isdigit():
            cleaned_chars.append(" number ")
        else:
            cleaned_chars.append(ch)
    text = ''.join(cleaned_chars)

    # Keep only allowed chars
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789 "
    filtered = ""
    for ch in text:
        filtered += ch if ch in allowed else " "

    # Split and remove single-letter junk
    tokens = np.array(filtered.split())
    mask = np.char.str_len(tokens) > 1
    return tokens[mask]

import numpy as np

def load_data(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            if len(row) < 4:
                continue
            try:
                label = int(row[3].strip())
            except ValueError:
                continue
            text = str(row[2]).strip()
            if not text:
                continue
            texts.append(text)
            labels.append(label)

    print(f"Loaded {len(texts)} rows after cleaning.")

    texts = np.array(texts, dtype=str)
    labels = np.array(labels, dtype=int)

    # Remove duplicates
    unique_texts, unique_idx = np.unique(texts, return_index=True)
    texts = texts[np.sort(unique_idx)]
    labels = labels[np.sort(unique_idx)]

    # Clean and tokenize
    tokenized_docs = np.array([clean_text(t) for t in texts], dtype=object)
    return tokenized_docs, labels


def split_data(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_idx = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
def build_bow_matrix(tokenized_docs, vocab):
    vocab_list = list(vocab.keys())
    index = {w:i for i,w in enumerate(vocab_list)}

    X = np.zeros((len(tokenized_docs), len(vocab_list)), dtype=float)

    for i, tokens in enumerate(tokenized_docs):
        for t in tokens:
            if t in index:
                X[i, index[t]] += 1

        # Normalize safely
        row_sum = np.sum(X[i])
        if row_sum > 0:
            X[i] /= row_sum

    return X, vocab_list

def select_top_k_vocab(tokenized_docs, labels, k=5000, eps=1e-9):

    n = len(tokenized_docs)
    
    df_spam = {}
    df_ham = {}
    n_spam = np.sum(labels == 1)
    n_ham = np.sum(labels == 0)

    for tokens, y in zip(tokenized_docs, labels):
        seen = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            if y == 1:
                df_spam[t] = df_spam.get(t, 0) + 1
            else:
                df_ham[t] = df_ham.get(t, 0) + 1

    all_tokens = set(df_spam.keys()) | set(df_ham.keys())
    scores = {}

    for t in all_tokens:
        a = df_spam.get(t, 0)
        b = df_ham.get(t, 0)
        p_spam = (a + eps) / (n_spam + 2*eps)
        p_ham = (b + eps) / (n_ham + 2*eps)
        score = abs(np.log(p_spam / p_ham))
        scores[t] = score

    top_tokens = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    vocab = {token: idx for idx, (token, _) in enumerate(top_tokens)}

    return vocab
def build_tfidf_matrix(tokenized_docs, vocab):
    n_docs = len(tokenized_docs)
    V = len(vocab)
    X = np.zeros((n_docs, V), dtype=float)
    df = np.zeros(V, dtype=float)

    for i, tokens in enumerate(tokenized_docs):
        counts = {}
        for t in tokens:
            if t in vocab:
                idx = vocab[t]
                counts[idx] = counts.get(idx, 0) + 1

        for idx, c in counts.items():
            X[i, idx] = c
            df[idx] += 1

    idf = np.log((n_docs) / (df + 1.0))
    X = X * idf[np.newaxis, :]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    return X, list(vocab.keys())
