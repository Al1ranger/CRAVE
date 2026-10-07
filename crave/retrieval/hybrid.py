from collections import Counter, defaultdict
import math
import re
import time
import numpy as np

def terms(text):
    return re.findall(r"\w+", text.lower())

class HybridRetriever:
    """BM25 + frozen dense cosine retrieval, reciprocal rank fusion, fixed top-k."""
    def __init__(self, documents, encoder):
        self.documents, self.encoder = documents, encoder
        self.counts = [Counter(terms(d["title"] + " " + d["text"])) for d in documents]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=float)
        self.average_length = float(np.mean(self.lengths)) or 1
        self.postings = defaultdict(list)
        for i, count in enumerate(self.counts):
            for term, tf in count.items(): self.postings[term].append((i, tf))
        self.embeddings = np.concatenate([encoder.encode([d["title"] + " " + d["text"] for d in documents[i:i+16]])
                                          for i in range(0, len(documents), 16)])

    def search(self, query, k=5):
        start = time.perf_counter()
        score = np.zeros(len(self.documents))
        for term in set(terms(query)):
            postings = self.postings[term]
            df = len(postings)
            idf = math.log(1 + (len(score) - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                score[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average_length))
        dense = self.embeddings @ self.encoder.encode([query])[0]
        fusion = np.zeros(len(score))
        for ranking in [np.argsort(-score, kind="stable"), np.argsort(-dense, kind="stable")]:
            for rank, idx in enumerate(ranking): fusion[idx] += 1 / (60 + rank + 1)
        order = np.argsort(-fusion, kind="stable")[:k]
        return [dict(self.documents[i], retrieval_score=float(fusion[i])) for i in order], time.perf_counter() - start
