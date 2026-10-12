"""Check supplied reference identifiers against publisher/arXiv/DOI metadata."""
import concurrent.futures, datetime, html, json, pathlib, re, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/mive_revision/artifacts/references'
OUT.mkdir(parents=True,exist_ok=True)

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CRAVE-reference-audit/1.0'}),timeout=45) as response:
        body=response.read().decode('utf-8',errors='replace')
        return response.status,body,response.url

def verify(item):
    result=dict(item);result['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        status,body,resolved=get(item['url'])
        (OUT/(str(item['number'])+'.html')).write_text(body,encoding='utf-8')
        result.update(http_status=status,resolved_url=resolved)
        titles=re.findall(r'<meta\s+name=["\']citation_title["\']\s+content=["\']([^"\']+)',body,re.I)
        if not titles: titles=re.findall(r'<title[^>]*>(.*?)</title>',body,re.I|re.S)
        result['retrieved_title']=html.unescape(re.sub('<.*?>','',titles[0])).strip() if titles else None
        result['status']='PRIMARY_PAGE_RETRIEVED_TITLE_REVIEW_REQUIRED'
        if item.get('doi'):
            doi=item['doi']
            api='https://api.datacite.org/dois/'+doi if doi.lower().startswith('10.48550/') else 'https://api.crossref.org/works/'+doi
            _,metadata,_=get(api);data=json.loads(metadata)
            (OUT/(str(item['number'])+'_doi.json')).write_text(metadata,encoding='utf-8')
            title=data['data']['attributes']['titles'][0]['title'] if 'data' in data else data['message']['title'][0]
            result['doi_title']=title;result['doi_status']='REGISTERED_TITLE_REVIEW_REQUIRED'
    except Exception as exc:result['error']=str(exc);result['status']='UNRESOLVED'
    return result

if __name__=='__main__':
    from docx import Document
    doc=Document(r'C:/Users/ehsan/Downloads/CRAVE_Revised_Manuscript (2).docx')
    rows=[]
    for p in doc.paragraphs:
        m=re.match(r'^\[(\d+)\]\s*(.*)',p.text)
        if not m:continue
        urls=re.findall(r'https://\S+',p.text)
        if not urls:continue
        url=urls[-1].rstrip('.,')
        doi=re.search(r'DOI\s+(10\.\S+)',p.text)
        if doi:doi=doi.group(1).rstrip('.,')
        else:
            aid=re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})',url)
            doi='10.48550/arXiv.'+aid.group(1) if aid else None
        rows.append({'number':int(m.group(1)),'supplied':p.text,'url':url,'doi':doi})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(verify,rows))
    (OUT/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps([{k:r.get(k) for k in ['number','status','retrieved_title','doi_title','error']} for r in results],indent=2))
