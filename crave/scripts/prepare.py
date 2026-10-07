"""Download pinned public assets into the workspace; never needs API keys."""
import argparse
import json
import shutil
import urllib.request
import zipfile
import time
from pathlib import Path
from crave.common import ROOT, file_hash, save_json, log

def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CRAVE-research/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)

def download(url, path, expected_size=None, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        if expected_size is None or path.stat().st_size == expected_size:
            if expected_sha is None or file_hash(path) == expected_sha:
                return
    log("download", url=url, target=str(path))
    tmp = path.with_suffix(path.suffix + ".part")
    if path.exists() and expected_size and path.stat().st_size < expected_size and not tmp.exists():
        path.replace(tmp)
    if expected_size is None:
        request = urllib.request.Request(url, headers={"User-Agent": "CRAVE-research/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as target:
            expected_size = int(response.headers.get("Content-Length", "0")) or None
            shutil.copyfileobj(response, target, 1024 * 1024)
    else:
        attempts=0
        while not tmp.exists() or tmp.stat().st_size < expected_size:
            offset = tmp.stat().st_size if tmp.exists() else 0
            end = min(offset+16*1024*1024-1, expected_size-1)
            request = urllib.request.Request(url, headers={"User-Agent": "CRAVE-research/0.1", "Range": f"bytes={offset}-{end}"})
            try:
                with urllib.request.urlopen(request,timeout=120) as response:
                    content_range=response.headers.get("Content-Range","")
                    if offset and not content_range.startswith(f"bytes {offset}-"):
                        raise IOError("Server did not honor resume offset: " + content_range)
                    with open(tmp,"ab" if offset else "wb") as target:
                        while block := response.read(1024*1024): target.write(block)
                if tmp.stat().st_size <= offset: raise IOError("Download made no progress")
                log("download_progress",target=str(path),bytes=tmp.stat().st_size,expected=expected_size)
                attempts=0
            except Exception as error:
                attempts+=1; log("download_retry",target=str(path),attempt=attempts,error=str(error))
                if attempts>=3: raise
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise IOError(f"Incomplete file: {tmp.stat().st_size}/{expected_size} bytes; retained .part for resume")
    if expected_sha is not None and file_hash(tmp)!=expected_sha:
        raise IOError("Downloaded file SHA256 mismatch; refusing to load")
    tmp.replace(path)

def prepare_models():
    runtime = ROOT / "runtime"
    release = fetch_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
    if any(a["name"] == "nightly-tag.txt" for a in release["assets"]):
        asset = next(a for a in release["assets"] if a["name"] == "nightly-tag.txt")
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as response:
            tag = response.read().decode().strip()
        release = fetch_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/" + tag)
    assets = release["assets"]
    choices = [a for a in assets if "win" in a["name"].lower() and "vulkan" in a["name"].lower() and "x64" in a["name"].lower()]
    if not choices:
        raise RuntimeError("No official Windows x64 Vulkan asset in " + release["tag_name"] + ": " + str([a['name'] for a in assets]))
    asset = choices[0]
    archive = runtime / asset["name"]
    download(asset["browser_download_url"], archive, asset['size'], asset.get('digest','').removeprefix('sha256:') or None)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            destination = (runtime / item.filename).resolve()
            if not destination.is_relative_to(runtime.resolve()):
                raise ValueError("Unsafe runtime archive path")
        z.extractall(runtime)
    model_id = "Qwen/Qwen3-4B-GGUF"
    model = fetch_json("https://huggingface.co/api/models/" + model_id + "?blobs=true")
    model_file = "Qwen3-4B-Q4_K_M.gguf"
    info=next(f for f in model['siblings'] if f['rfilename']==model_file)
    download(f"https://huggingface.co/{model_id}/resolve/{model['sha']}/{model_file}", ROOT / "models_data" / model_file,info['size'],info['lfs']['sha256'])
    encoder_id = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = fetch_json("https://huggingface.co/api/models/" + encoder_id + "?blobs=true")
    for name in ["onnx/model.onnx", "tokenizer.json"]:
        info=next(f for f in encoder['siblings'] if f['rfilename']==name)
        download(f"https://huggingface.co/{encoder_id}/resolve/{encoder['sha']}/{name}", ROOT / "models_data" / "encoder" / name,info['size'],info.get('lfs',{}).get('sha256'))
    manifest = {"runtime": {"tag": release["tag_name"], "asset": asset["name"], "sha256": file_hash(archive)},
                "model": {"repository": model_id, "revision": model["sha"], "file": model_file, "sha256": file_hash(ROOT / "models_data" / model_file), "quantization": "Q4_K_M"},
                "encoder": {"repository": encoder_id, "revision": encoder["sha"], "sha256": file_hash(ROOT / "models_data" / "encoder" / "onnx/model.onnx")}}
    save_json(ROOT / "artifacts/model_manifest.json", manifest)
    log("models_ready", **manifest)

def prepare_hotpot(n_train=12, n_dev=40):
    # The official HotpotQA HF mirror is used; exact returned rows are preserved and checksummed.
    dataset = "hotpotqa/hotpot_qa"
    metadata = fetch_json("https://huggingface.co/api/datasets/" + dataset)
    for split, count in [("train", n_train), ("validation", n_dev)]:
        url = f"https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split={split}&offset=0&length={count}"
        result = fetch_json(url)
        save_json(ROOT / "data" / f"hotpot_{split}.json", {"repository": dataset, "revision": metadata.get("sha"), "source": url, "rows": [r["row"] for r in result["rows"]]})
    log("hotpot_ready", train=n_train, validation=n_dev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["models", "hotpot", "all"])
    args = parser.parse_args()
    if args.stage in ["models", "all"]:
        prepare_models()
    if args.stage in ["hotpot", "all"]:
        prepare_hotpot()
