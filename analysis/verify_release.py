"""Numerical publication checks; raw benchmark texts are not redistributed."""
from pathlib import Path
from collections import defaultdict
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from crave.pipeline import code_hash
from crave.common import digest,file_hash
pilot=ROOT/'crave/runs/bdd2b8a990571bb1';expanded=ROOT/'crave/runs/expanded_8484164aef776a12'
config=json.loads((expanded/'FROZEN_EXPANDED_CONFIG.json').read_text())
assert code_hash()==config['config']['framework_source_hash']
assert file_hash(ROOT/'tmp/expanded_evaluation.py')==config['config']['extension_script_hash']
for seed,expected in config['config']['controller_checkpoints'].items():assert file_hash(pilot/f'CRAVE_{seed}.npz')==expected
metrics=['exact_match','f1','generated_tokens','input_tokens','total_tokens','retrieval_calls','verification_calls','latency','controller_latency','normalized_cost']
checked=0
for run,n,frozen_name in [(pilot,162,'FROZEN_PILOT_CONFIG.json'),(expanded,480,'FROZEN_EXPANDED_CONFIG.json')]:
    frozen=json.loads((run/frozen_name).read_text());assert digest(frozen['config'])==frozen['sha256']
    release=json.loads((run/'evaluation.json').read_text());rows=release['data'];assert len(rows)==n and digest(rows)==release['released_data_hash']
    assert len({(r['dataset'],r['task_id'],r['method'],r['seed']) for r in rows})==n
    for row in rows:
        assert 'final_answer' not in row
        assert row['generated_tokens']<=row['budget_tokens'] and row['retrieval_calls']<=row['budget_retrieval'] and row['verification_calls']<=row['budget_verification']
        assert row['budget_violation']==0
        assert sum(a['token_cost'] for a in row['actions'])+row['routing_tokens']==row['generated_tokens']
        assert sum(a['retrieval_cost'] for a in row['actions'])==row['retrieval_calls']
        assert sum(a['verification_cost'] for a in row['actions'])==row['verification_calls']
        assert abs(row['generated_tokens']/4096+row['retrieval_calls']/4+row['verification_calls']/2+row['latency']/60-row['normalized_cost'])<1e-10
    with open(run/'results/summary.csv',encoding='utf-8') as f:summary=list(csv.DictReader(f))
    for r in summary:
        selected=[v for v in rows if (v['dataset'],v['method'])==(r['dataset'],r['method'])];groups=defaultdict(list)
        for v in selected:groups[v['task_id']].append(v)
        assert all(len(group)==3 for group in groups.values())
        for metric in metrics:
            if metric in r:assert abs(np.mean([np.mean([v[metric] for v in group]) for group in groups.values()])-float(r[metric]))<1e-9,(run.name,r['method'],metric)
    checked+=n
manifest=json.loads((ROOT/'analysis/release_manifest.json').read_text())
for name,expected in manifest['files'].items():assert file_hash(ROOT/name)==expected,name
print(json.dumps({'passed':True,'numeric_episodes':checked,'framework_source_hash':code_hash(),'frozen_checkpoints':3,'release_files':len(manifest['files']),'scope':'Numerical release integrity; raw answer recomputation requires retained local records'}))
