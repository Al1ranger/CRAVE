"""Download official, revision-pinned GGUFs; verify every LFS SHA before use."""
import concurrent.futures, hashlib, json, os, pathlib, urllib.request, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/mive_revision/artifacts'
OUT.mkdir(parents=True, exist_ok=True)

def fetch(model):
    repo = 'Qwen/' + model + '-Instruct-GGUF'
    with urllib.request.urlopen('https://huggingface.co/api/models/' + repo + '?blobs=true', timeout=60) as f:
        metadata = json.load(f)
    files = [x for x in metadata['siblings'] if 'q4_k_m' in x['rfilename'].lower()]
    assert files
    folder = ROOT / 'crave/models_data' / model
    folder.mkdir(exist_ok=True)
    record = {'repository': repo, 'revision': metadata['sha'], 'files': files}
    (OUT / (model + '_download_manifest.json')).write_text(json.dumps(record, indent=2))
    for item in files:
        dest = folder / item['rfilename']
        expected = item['lfs']['sha256']
        if dest.exists() and hashlib.file_digest(open(dest, 'rb'), 'sha256').hexdigest() == expected:
            continue
        part = dest.with_suffix('.gguf.part')
        url = 'https://huggingface.co/' + repo + '/resolve/' + metadata['sha'] + '/' + item['rfilename']
        chunk_size=8*1024*1024
        chunks=[]
        def chunk(index):
            start=index*chunk_size;end=min(start+chunk_size,item['size'])-1
            path=folder/(item['rfilename']+'.chunk'+str(index))
            if path.exists() and path.stat().st_size==end-start+1:return path
            last=None
            for attempt in range(3):
                try:
                    request=urllib.request.Request(url+'?download=true&chunk='+str(index),headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=45) as response:
                        assert response.status==206
                        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                        with open(path,'wb') as output:
                            remaining=end-start+1
                            while remaining:
                                block=response.read(min(1024*1024,remaining))
                                if not block:raise IOError('Truncated range')
                                output.write(block);remaining-=len(block)
                    return path
                except Exception as exc:last=exc;time.sleep(1)
            raise last
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            chunks=list(pool.map(chunk,range((item['size']+chunk_size-1)//chunk_size)))
        with open(part,'wb') as output:
            for path in chunks:
                with open(path,'rb') as source:
                    while block:=source.read(1024*1024):output.write(block)
        assert part.stat().st_size == item['size']
        assert hashlib.file_digest(open(part, 'rb'), 'sha256').hexdigest() == expected
        os.replace(part, dest)
        print(json.dumps({'verified': str(dest), 'sha256': expected}), flush=True)
    record['status'] = 'verified'
    (OUT / (model + '_download_manifest.json')).write_text(json.dumps(record, indent=2))

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, ['Qwen2.5-7B', 'Qwen2.5-1.5B']))
