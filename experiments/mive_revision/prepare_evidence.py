"""Recompute archival comparisons and define the prospective scaled study."""
import csv,json,pathlib,sys
from collections import defaultdict
import numpy as np
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/mive_revision/artifacts';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from crave.common import read_json,save_json,digest

def archival():
    raw=read_json(ROOT/'crave/runs/expanded_8484164aef776a12/evaluation.json')['data']
    grouped=defaultdict(list)
    for row in raw:grouped[(row['dataset'],row['method'],row['task_id'])].append(row)
    records=[]
    rng=np.random.default_rng(20261011)
    for dataset in sorted({r['dataset'] for r in raw}):
        for method in ['CRAVE','confidence','fixed','llm_router']:
            subset=[v for (d,m,q),v in sorted(grouped.items()) if d==dataset and m==method]
            ids=sorted(q for d,m,q in grouped if d==dataset and m==method)
            metrics={}
            for key in ['exact_match','f1','generated_tokens','input_tokens','retrieval_calls','verification_calls','latency']:
                if key not in subset[0][0]:continue
                values=np.array([np.mean([e[key] for e in g]) for g in subset])
                idx=rng.integers(0,len(values),(10000,len(values)))
                metrics[key]={'mean':float(values.mean()),'ci95':np.quantile(values[idx].mean(1),[.025,.975]).tolist()}
            records.append({'dataset':dataset,'method':method,'n_questions':len(ids),'n_seeds':3,'metrics':metrics})
    # Append auxiliary resources after the original bootstrap stream completes.
    # Existing metric draws therefore keep their original deterministic order.
    extra_rng=np.random.default_rng(20261011)
    for record in records:
        subset=[v for (d,m,q),v in sorted(grouped.items()) if d==record['dataset'] and m==record['method']]
        for key in ['context_tokens_added','controller_latency','routing_tokens','physical_episode_seconds','think_calls']:
            values=np.array([np.mean([sum(a['action']=='THINK' for a in e['actions']) if key=='think_calls' else e[key] for e in g]) for g in subset])
            idx=extra_rng.integers(0,len(values),(10000,len(values)))
            record['metrics'][key]={'mean':float(values.mean()),'ci95':np.quantile(values[idx].mean(1),[.025,.975]).tolist()}
    save_json(OUT/'archival_reanalysis.json',records)
    return records

def plan():
    # This is an intent specification. It explicitly cannot authorize held-out runs
    # until new data manifests, trained controllers and development choices exist.
    config={'status':'PROSPECTIVE_INCOMPLETE_NOT_A_HELDOUT_FREEZE','datasets':['HotpotQA','2WikiMultiHopQA','MuSiQue'],
            'questions_per_dataset':300,'seeds':[17,29,43],'output_token_caps':[256,512,1024],
            'retrieval_cap':1,'verification_cap':1,'methods':['legacy','confidence','fixed','MIVE'],
            'primary_episodes_required':3*300*3*3*4,'NI_EM_margin':-.01,'savings_null':'output-token savings <= 0',
            'quality_tests':9,'savings_tests':9,'multiplicity':'Holm within each nine-test family; success requires both adjusted rejections',
            'power':'compute paired question-level dev variance after averaging three seeds; conservative alpha=.05/9, power .80',
            'power_formula':'n=ceil((z_(1-alpha)+z_.80)^2*sigma_dev^2/(assumed_delta-margin)^2)',
            'baseline_conditions':{'Adaptive-RAG':'official classifier, corpus and generator matched; all classifier/retrieval tokens charged',
                                   'FLARE':'official future-sentence confidence trigger; charge discarded drafts and retrievals'},
            'required_before_heldout':['official-release identity and alias mapping','train/dev/heldout IDs with component overlap audit',
                'development reward-variance gate','K stability rule','dev baseline quality gate','dev-trained MIVE checkpoints',
                'frozen thresholds, penalties, prices, K and model-specific prompts','dev-only power analysis'],
            'heldout_inference_permitted':False}
    save_json(OUT/'prospective_scaled_plan.json',{'config':config,'sha256':digest(config)})

if __name__=='__main__':archival();plan()
