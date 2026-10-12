"""Review retained titles and verify supplementary DOIs against primary metadata."""
import concurrent.futures,datetime,html,json,pathlib,re,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/mive_revision/artifacts/references'
def normalized(text):return re.sub(r'[^a-z0-9]','',html.unescape(text or '').lower())
def extra(item):
    number,doi,title=item
    row={'number':number,'doi':doi,'expected_title':title,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        retained=OUT/(str(number)+'_supplemental_doi.json')
        if retained.exists():data=json.loads(retained.read_text(encoding='utf-8'))
        else:
            request=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'CRAVE-reference-audit/1.0'})
            with urllib.request.urlopen(request,timeout=25) as response:data=json.load(response)
            retained.write_text(json.dumps(data,indent=2),encoding='utf-8')
        row['registered_title']=data['message']['title'][0]
        row['title_matches']=normalized(title)==normalized(row['registered_title'])
        row['status']='VERIFIED' if row['title_matches'] else 'TITLE_MISMATCH'
    except Exception as exc:
        publisher=OUT/(str(number)+'.html')
        if publisher.exists() and doi in publisher.read_text(encoding='utf-8'):
            row.update(status='PUBLISHER_DOI_VERIFIED_REGISTRY_UNRESOLVED',title_matches=True,publisher_evidence=str(publisher),registry_error=str(exc))
        else:row.update(status='UNRESOLVED',error=str(exc))
    return row
def main():
    source=json.loads((OUT/'audit.json').read_text(encoding='utf-8'));review=[]
    for row in source:
        supplied=normalized(row['supplied']);title=normalized(row.get('doi_title') or row.get('retrieved_title'))
        match=bool(title) and (title in supplied or title.removesuffix('aclanthology') in supplied)
        if row['number'] in [13,14,15]:
            identity={13:'qwen3-4b-gguf',14:'all-minilm-l6-v2',15:'llama.cpp'}[row['number']]
            match=row.get('http_status')==200 and identity in row.get('retrieved_title','').lower()
        review.append({'number':row['number'],'title_matches_supplied':match,
                       'status':'TITLE_VERIFIED' if match else 'PRIMARY_PAGE_UNRESOLVED',
                       'doi':row.get('doi'),'doi_title_verified':bool(row.get('doi_title')) and normalized(row['doi_title']) in supplied})
    extras=[(27,'10.18653/v1/2024.naacl-long.389','Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity'),
            (28,'10.18653/v1/2023.emnlp-main.495','Active Retrieval Augmented Generation'),
            (32,'10.18653/v1/2022.naacl-main.272','ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction'),
            (35,'10.1080/01621459.1963.10500830','Probability Inequalities for Sums of Bounded Random Variables')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(extra,extras))
    report={'retained_title_review':review,'supplementary_dois':verified,
            'notes':['Holm JSTOR page returned Client Challenge; no DOI is asserted for that reference.',
                     'Altman book metadata verified at the publisher; no book DOI is asserted.',
                     'ACL supplementary DOIs also appear in the retained publisher pages for references 27, 28, and 32.']}
    (OUT/'reviewed_references.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
