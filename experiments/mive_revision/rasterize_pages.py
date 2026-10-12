"""Rasterize the Word-exported PDF; visual review is recorded separately."""
import hashlib,json,pathlib
from pdf2image import convert_from_path
from pypdf import PdfReader
ROOT=pathlib.Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/mive_revision';PDF=WORK/'rendered/MIVE_Verified_Manuscript.pdf'
POPPLER=pathlib.Path(r'C:/Users/ehsan/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')

def main():
    manifest=WORK/'artifacts/render_manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    prior_hashes=[hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in previous.get('pages',[])]
    images=convert_from_path(PDF,dpi=120,output_folder=PDF.parent,fmt='png',
        output_file='page',paths_only=True,thread_count=2,poppler_path=POPPLER)
    assert len(images)==len(PdfReader(PDF).pages)
    hashes=[hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in images]
    report={'page_count':len(images),'pages':images,'page_image_sha256':hashes,
        'identical_to_previous_render':[i+1 for i,h in enumerate(hashes) if i<len(prior_hashes) and h==prior_hashes[i]],
        'renderer':'Microsoft Word PDF export; Poppler at 120 dpi',
        'docx_sha256':hashlib.sha256((ROOT/'MIVE_Verified_Manuscript.docx').read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'visual_review_pending':True}
    manifest.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
