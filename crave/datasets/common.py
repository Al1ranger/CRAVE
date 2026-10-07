import csv
import json
from crave.common import ROOT, digest, read_json, save_json, file_hash

def validate_splits(tasks):
    ids,questions,sources = {},{},{}
    for task in tasks:
        key=(task["dataset"],task["task_id"])
        if key in ids: raise ValueError("Duplicate task ID across splits: " + str(key))
        ids[key]=task["split"]
        q=" ".join(task["question"].lower().split())
        if q in questions and questions[q]!=task["split"]: raise ValueError("Question leakage across splits")
        questions[q]=task["split"]
        for source in task.get('source_question_ids',[]):
            source=tuple(source)
            if source in sources and sources[source]!=task['split']:
                raise ValueError('Component question leakage across splits')
            sources[source]=task['split']

def hotpot_rows(path, split, start=0, count=None):
    rows=read_json(path)["rows"]
    rows=rows[start:None if count is None else start+count]
    return [{"task_id":r.get("id",r.get("_id")),"dataset":"HotpotQA","split":split,"question":r["question"],
             "answer":r["answer"],"aliases":[],"documents":[{"title":title,"text":" ".join(sentences)}
                 for title,sentences in (zip(r["context"]["title"],r["context"]["sentences"]) if isinstance(r["context"],dict) else r["context"])]} for r in rows]

def musique_rows(path, split):
    rows=[json.loads(line) for line in open(path,encoding="utf-8") if line.strip()]
    return [{"task_id":r["id"],"dataset":"MuSiQue","split":split,"question":r["question"],"answer":r["answer"],
             "aliases":r.get("answer_aliases",[]),"documents":[{"title":p["title"],"text":p["paragraph_text"]} for p in r["paragraphs"]]}
            for r in rows if r.get("answerable",True)]

def wiki2hop_rows(path, split="test"):
    rows=read_json(path)
    return [{"task_id":r["_id"],"dataset":"2WikiMultiHopQA","split":split,"question":r["question"],"answer":r["answer"],
             "aliases":[],"documents":[{"title":title,"text":" ".join(sentences)} for title,sentences in r["context"]]} for r in rows]

def build_corpus(tasks):
    # Pool every provided context without consulting supporting facts or answer labels.
    docs={}
    for task in tasks:
        for doc in task["documents"]:
            key=digest(doc)
            docs[key]={"doc_id":key,**doc}
    return [docs[key] for key in sorted(docs)]

def author_raw_rows(path,dataset,split,count):
    rows=read_json(path)['rows'][:count]
    tasks=[]
    for r in rows:
        if not r.get('answerable',True):continue
        question=r.get('composed_question_text') or r['question_text']
        if '[[' in question:raise ValueError('Unresolved question template in author raw export')
        tasks.append({'task_id':r['id'],'dataset':dataset,'split':split,'question':question,'answer':r['answer_text'],
                      'aliases':r.get('answer_aliases_texts',r.get('answer_aliases',[])),
                      'documents':[{'title':p['wikipedia_title'],'text':p['paragraph_text']} for p in r['contexts']],
                      'source_question_ids':[(d.get('dataset',''),str(d['id'])) for d in r.get('decomposed_instances',[]) if 'id' in d],
                      'metric_scope':'normalized answer EM/F1; released alias mapping verification pending'})
    return tasks

def manifest(tasks, source_metadata):
    rows=[]
    for dataset,split in sorted(set((t["dataset"],t["split"]) for t in tasks)):
        selected=[t for t in tasks if (t["dataset"],t["split"])==(dataset,split)]
        metadata=source_metadata.get((dataset,split),source_metadata.get(dataset,{}))
        rows.append({"dataset":dataset,"split":split,"n_examples":len(selected),
                     "source":metadata.get("source","local_authoritative_export_required"),
                     "version":metadata.get("revision") or metadata.get("version","unknown"),
                     "license":"CC BY-SA 4.0" if dataset=="HotpotQA" else ('CC BY 4.0' if dataset=='MuSiQue' else 'VERIFY_UPSTREAM'),
                     "checksum":digest(selected),"source_subset_checksum":metadata.get('subset_checksum'),
                     "used_for_training":split=="train","reserved_for_tuning":split=="dev","used_for_tuning":False,
                     "used_for_final_evaluation":False})
    with open(ROOT/"artifacts/dataset_manifest.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows
