from __future__ import annotations
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

def log(event, **fields):
    print(canonical({"event": event, "time": time.time(), **fields}), flush=True)
