"""Independent ranged download parts; mandatory official whole-file SHA256."""
import concurrent.futures
import os
import time
import urllib.request
from crave.common import ROOT,file_hash,log

SIZE=2497280256
SHA='7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
URL='https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf'
TARGET=ROOT/'models_data/Qwen3-4B-Q4_K_M.gguf'
PARTS=ROOT/'models_data/qwen_download_parts'
CHUNK=8*1024*1024

def fetch_part(index):
    start=index*CHUNK;end=min(SIZE-1,start+CHUNK-1);expected=end-start+1
    path=PARTS/f'{index:04d}.part'
    if path.exists() and path.stat().st_size==expected:return path
    for attempt in range(20):
        try:
            saved=path.stat().st_size if path.exists() else 0
            if saved==expected:return path
            if saved>expected: saved=0
            offset=start+saved
            request=urllib.request.Request(URL,headers={'User-Agent':'CRAVE-research/0.1','Range':f'bytes={offset}-{end}'})
            with urllib.request.urlopen(request,timeout=120) as response:
                header=response.headers.get('Content-Range','')
                if not header.startswith(f'bytes {offset}-'):raise IOError('Incorrect ranged response offset')
                with open(path,'ab' if saved else 'wb') as target:
                    while block:=response.read(256*1024):target.write(block)
            if path.stat().st_size!=expected:raise IOError('Incomplete range response')
            log('model_part',part=index,bytes=expected)
            return path
        except Exception as error:
            log('model_part_retry',part=index,attempt=attempt+1,error=str(error))
            if attempt==19:raise
            time.sleep(min(15,2+attempt))

def main():
    if TARGET.exists() and TARGET.stat().st_size==SIZE and file_hash(TARGET)==SHA:
        log('model_verified');return
    PARTS.mkdir(exist_ok=True)
    count=(SIZE+CHUNK-1)//CHUNK
    prefix=TARGET.with_suffix(TARGET.suffix+'.part')
    if not prefix.exists() and TARGET.exists():prefix=TARGET
    if prefix.exists():
        # Reuse only complete 8 MiB chunks. Do not consume or mutate an active partial stream.
        with open(prefix,'rb') as source:
            for index in range(prefix.stat().st_size//CHUNK):
                path=PARTS/f'{index:04d}.part';block=source.read(CHUNK)
                if len(block)==CHUNK and not path.exists():path.write_bytes(block)
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        list(pool.map(fetch_part,range(count)))
    assembled=TARGET.with_suffix('.verified.tmp')
    with open(assembled,'wb') as target:
        for index in range(count):
            with open(PARTS/f'{index:04d}.part','rb') as source:
                while block:=source.read(1024*1024):target.write(block)
    if assembled.stat().st_size!=SIZE or file_hash(assembled)!=SHA:raise IOError('Assembled model failed official size/SHA256 verification')
    os.replace(assembled,TARGET)
    log('model_verified',bytes=SIZE,sha256=SHA)

if __name__=='__main__':main()
