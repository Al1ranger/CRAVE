"""Stateless local llama.cpp requests, exact tokenizer counts, bounded JSON outputs."""
import json
import time
import urllib.request
from dataclasses import dataclass
from crave.common import ROOT, digest, read_json, save_json

@dataclass
class Generation:
    data: dict
    output_tokens: int
    input_tokens: int
    latency: float
    cache_hit: bool = False
    valid_json: bool = True

class LlamaBackend:
    def __init__(self, base_url="http://127.0.0.1:8096", model_hash="unknown", cache=True):
        self.base_url = base_url.rstrip("/")
        self.model_hash = model_hash
        self.cache = cache
        self.calls = 0
        self.physical_tokens = 0
        self.physical_seconds = 0.0

    def post(self, endpoint, payload):
        req = urllib.request.Request(self.base_url + endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)

    def tokens(self, text):
        return self.post("/tokenize", {"content": text, "add_special": False})["tokens"]

    def count(self, text):
        return len(self.tokens(text))

    def truncate(self, text, limit):
        tokens = self.tokens(text)
        if len(tokens) <= limit:
            return text
        return self.post("/detokenize", {"tokens": tokens[:limit]})["content"]

    def generate(self, system, user, max_tokens, seed, schema=None):
        prompt = ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
                  + user + "\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.7, "top_p": 0.8,
                   "top_k": 20, "seed": int(seed) % (2**31 - 1), "cache_prompt": False,
                   "stop": ["<|im_end|>", "<|endoftext|>"]}
        if schema:
            payload["json_schema"] = schema
        key = digest({"model": self.model_hash, "payload": payload})
        path = ROOT / "runs" / "generation_cache" / (key + ".json")
        if self.cache and path.exists():
            result = read_json(path)
            result["cache_hit"] = True
            return Generation(**result)
        start = time.perf_counter()
        response = self.post("/completion", payload)
        latency = time.perf_counter() - start
        text = response["content"].strip()
        try:
            data = json.loads(text)
            valid = isinstance(data, dict)
        except json.JSONDecodeError:
            data, valid = {}, False
        output = int(response["tokens_predicted"]) if "tokens_predicted" in response else self.count(text)
        if output > max_tokens:
            raise RuntimeError("Backend violated token reservation")
        inputs = int(response["tokens_evaluated"]) if "tokens_evaluated" in response else self.count(prompt)
        result = Generation(data, output, inputs, latency, False, valid)
        self.calls += 1
        self.physical_tokens += output
        self.physical_seconds += latency
        if self.cache:
            save_json(path, result.__dict__)
        return result

ANSWER_SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string"}, "summary": {"type": "string"},
    "confidence": {"type": "number"}, "support": {"type": "number"}},
    "required": ["answer", "summary", "confidence", "support"], "additionalProperties": False}
ROUTER_SCHEMA = {"type": "object", "properties": {"action": {"type": "string", "enum": ["THINK", "RETRIEVE", "VERIFY", "STOP"]}}, "required": ["action"], "additionalProperties": False}
