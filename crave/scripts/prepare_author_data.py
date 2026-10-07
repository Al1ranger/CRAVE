"""Small pilot subsets of public raw exports linked by the MuSiQue authors."""
import json
import urllib.request
from crave.common import ROOT,save_json,digest,log

FILES={'musique_raw_train':('1X3rJEAofrJ_tI55U2bquCZ-StVeaWnWu',12),
       'musique_raw_dev':('1TRXU68wveSehVbQrRRtWsUsFkKUF43QS',12),
       '2wiki_raw_dev':('1XtMGLeBC52PvR2VbQ1u3gkFpfLic1mwq',8)}

def main():
    source='https://raw.githubusercontent.com/StonyBrookNLP/musique/main/.all_data_information.json'
    with urllib.request.urlopen(source,timeout=90) as r: provenance=r.read()
    information=json.loads(provenance)
    for name,(file_id,n) in FILES.items():
        if file_id not in information.values():raise ValueError('File ID missing from author provenance')
        path=ROOT/'data'/(name+'.json')
        if path.exists():continue
        url=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
        request=urllib.request.Request(url,headers={'User-Agent':'CRAVE-research/0.1'})
        rows=[]
        with urllib.request.urlopen(request,timeout=120) as response:
            while len(rows)<n:
                line=response.readline(2*1024*1024)
                if not line:break
                row=json.loads(line)
                if not isinstance(row,dict) or 'contexts' not in row:raise ValueError('Invalid author raw-data record')
                rows.append(row)
        if not rows:raise ValueError('No benchmark rows obtained')
        save_json(path,{'rows':rows,'source':url,'author_provenance':source,'author_provenance_sha256':digest(json.loads(provenance)),
                       'version':'author raw export; pilot subset, not claimed identical to full v1.0 release',
                       'subset_checksum':digest(rows),'full_file_checksum':'NOT_DOWNLOADED_FULL_FILE'})
        log('author_subset',file=name,n=len(rows),first_id=rows[0]['id'],keys=list(rows[0]),question=rows[0].get('composed_question_text',rows[0].get('question_text')))

if __name__=='__main__':main()
