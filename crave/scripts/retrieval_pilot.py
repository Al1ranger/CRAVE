"""Offline support-title retrieval check; gold annotations never enter the retriever."""
import time
from argparse import Namespace
from collections import defaultdict
import numpy as np
from crave.common import ROOT,read_json,save_json,digest,log
from crave.pipeline import task_data
from crave.datasets.common import build_corpus
from crave.models.state_encoder import FrozenEncoder
from crave.retrieval.hybrid import HybridRetriever
from crave.reporting import write_csv
from crave.evaluation.statistics import paired_bootstrap

def main():
    tasks=sum(task_data(Namespace(train_tasks=2,eval_tasks=3)),[])
    gold={}
    for filename in ['hotpot_train.json','hotpot_validation.json']:
        for row in read_json(ROOT/'data'/filename)['rows']:
            facts=row['supporting_facts']
            titles=facts['title'] if isinstance(facts,dict) else [f[0] for f in facts]
            gold[('HotpotQA',row.get('id',row.get('_id')))]=set(t.casefold() for t in titles)
    for filename,dataset in [('musique_raw_train.json','MuSiQue'),('musique_raw_dev.json','MuSiQue'),('2wiki_raw_dev.json','2WikiMultiHopQA')]:
        for row in read_json(ROOT/'data'/filename)['rows']:
            gold[(dataset,row['id'])]=set(c['wikipedia_title'].casefold() for c in row['contexts'] if c.get('is_supporting'))
    documents=build_corpus(tasks)
    started=time.perf_counter()
    retriever=HybridRetriever(documents,FrozenEncoder(ROOT/'models_data/encoder'))
    setup=time.perf_counter()-started
    rows=[]
    for task in tasks:
        docs,seconds=retriever.search(task['question'],5)
        expected=gold.get((task['dataset'],task['task_id']),set())
        found=expected.intersection(d['title'].casefold() for d in docs)
        rows.append({'dataset':task['dataset'],'split':task['split'],'task_id':task['task_id'],
                     'top_k':5,'gold_support_titles':len(expected),'retrieved_support_titles':len(found),
                     'support_title_recall':len(found)/len(expected) if expected else None,
                     'all_support_titles_retrieved':int(found==expected) if expected else None,
                     'query_seconds':seconds,'retrieved_doc_ids':'|'.join(d['doc_id'] for d in docs)})
    folder=ROOT/'artifacts/retrieval_pilot';folder.mkdir(exist_ok=True)
    write_csv(folder/'per_instance.csv',rows)
    summary=[]
    for dataset in sorted(set(r['dataset'] for r in rows)):
        selected=[r for r in rows if r['dataset']==dataset and r['support_title_recall'] is not None]
        if selected:
            summary.append({'dataset':dataset,**paired_bootstrap([r['support_title_recall'] for r in selected]),
                            'all_support_title_rate':float(np.mean([r['all_support_titles_retrieved'] for r in selected]))})
    write_csv(folder/'summary.csv',summary)
    save_json(folder/'provenance.json',{'corpus_hash':digest(documents),'n_documents':len(documents),
               'tasks_hash':digest(tasks),'encoder':read_json(ROOT/'artifacts/model_manifest.json')['encoder'],
               'setup_seconds':setup,'query':'question only','top_k':5,'fusion':'BM25+dense reciprocal rank, constant 60',
               'status':'REAL_DATA_RETRIEVAL_COMPONENT_ONLY','limitations':'Pooled benchmark passages; support-title proxy, not official sentence/paragraph metric; no controller or LLM scores; no tuning.'})
    lines=['# Real-data retrieval component pilot','','Pooled benchmark corpus; fixed question-only top-5 hybrid retrieval. Gold support annotations are used only by this offline evaluator. No controller or LLM scores. No settings changed from these observations.','',
           '|Dataset|N tasks with annotations|Support-title recall|95% task-bootstrap interval|All support titles found|','|---|---:|---:|---|---:|']
    for r in summary:lines.append(f"|{r['dataset']}|{r['n_tasks']}|{r['estimate']:.4f}|[{r['ci_low']:.4f}, {r['ci_high']:.4f}]|{r['all_support_title_rate']:.4f}|")
    lines+=['','This support-title proxy does not establish the official support-fact score or CRAVE task accuracy. Raw exports and tiny sample sizes limit interpretation.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    log('retrieval_component_complete',n_tasks=len(tasks),n_documents=len(documents),summary=summary)

if __name__=='__main__':main()
