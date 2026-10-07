import hashlib
from collections import OrderedDict
import numpy as np
from crave.agent.state import ACTIONS

class FrozenEncoder:
    def __init__(self, folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / "onnx/model.onnx"), options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        self.dim = 384
        self.cache = OrderedDict()

    def encode(self, texts):
        for text in texts:
            if text in self.cache:
                self.cache.move_to_end(text)
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            encoded = self.tokenizer.encode_batch(missing)
            data = {"input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                    "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64)}
            required = {x.name for x in self.session.get_inputs()}
            hidden = self.session.run(None, {k: v for k, v in data.items() if k in required})[0]
            mask = data["attention_mask"][..., None]
            vectors = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            self.cache.update(zip(missing, vectors.astype(np.float32)))
            while len(self.cache) > 5000:
                self.cache.popitem(last=False)
        return np.stack([self.cache[t] for t in texts])

class HashEncoder:
    """Deterministic offline unit-test fixture, forbidden in real experiments."""
    dim = 32
    def encode(self, texts):
        outputs = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                h = hashlib.sha256(word.encode()).digest()
                vector[int.from_bytes(h[:4], "little") % self.dim] += 1 if h[4] % 2 else -1
            outputs.append(vector / max(np.linalg.norm(vector), 1))
        return np.stack(outputs)

class StateFeatures:
    def __init__(self, encoder, ablation=None):
        self.encoder, self.ablation = encoder, ablation
        self.dim = encoder.dim * 4 + 13

    def __call__(self, state, utility_lambda):
        texts = [state.question, " ".join(state.history[-3:]),
                 " ".join(d["text"] for d in state.evidence), state.candidate_answer + " " + state.verification]
        embeddings = self.encoder.encode(texts).ravel()
        scalar = np.array([state.budget.tokens / 4096, state.budget.retrievals / 4,
                           state.budget.verifications / 2, state.confidence, state.convergence,
                           state.evidence_sufficiency, state.novelty, state.step / 16, utility_lambda,
                           *[float(state.previous_action == a) for a in ACTIONS]], dtype=np.float32)
        if self.ablation == "-Budget": scalar[:3] = 0
        if self.ablation == "-Confidence": scalar[3] = 0
        if self.ablation == "-Convergence": scalar[4] = 0
        if self.ablation == "-Evidence": scalar[5:7] = 0
        return np.concatenate([embeddings, scalar])
