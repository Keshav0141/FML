import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, docs, y):
        self.classes = np.unique(y)
        self.vocab = {}
        self.word_counts = {}
        self.class_totals = {}
        self.class_docs = {}
        self.total_docs = len(docs)

        for c in self.classes:
            self.word_counts[c] = {}
            self.class_totals[c] = 0
            self.class_docs[c] = 0

        for tokens, label in zip(docs, y):
            self.class_docs[label] += 1
            for token in tokens:
                self.vocab[token] = self.vocab.get(token, 0) + 1
                self.word_counts[label][token] = self.word_counts[label].get(token, 0) + 1
                self.class_totals[label] += 1

        self.vocab_size = len(self.vocab)

    def predict_one(self, tokens):
        scores = {}
        for c in self.classes:
            prior = np.log(self.class_docs[c] / self.total_docs)
            cond_sum = 0.0
            for token in tokens:
                token_count = self.word_counts[c].get(token, 0)
                num = token_count + self.alpha
                den = self.class_totals[c] + self.alpha * self.vocab_size
                cond_sum += np.log(num / den)
            scores[c] = prior + cond_sum
        return max(scores, key=scores.get)

    def predict(self, docs):
        preds = np.zeros(len(docs), dtype=int)
        for i, tokens in enumerate(docs):
            preds[i] = self.predict_one(tokens)
        return preds

