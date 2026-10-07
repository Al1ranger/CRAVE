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
metrics=['exact_match','f1','generated_tokens','input_tokens','total_tokens','retrieval_calls','verification_calls','latency','controller_latency','normalized_cost','task_score','budget_violation']
checked=0
intervals=0;comparisons=0
def bootstrap(values,seed):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return [float(values.mean()),*np.quantile(samples,[.025,.975])]
def permutation(values,seed):
    values=np.asarray(values,float);threshold=abs(values.mean())-1e-12
    if len(values)<=16:
        signs=((np.arange(2**len(values),dtype=np.uint32)[:,None]>>np.arange(len(values)))&1)*2.-1.
        return float((abs((signs*values).mean(1))>=threshold).mean())
    signs=np.random.default_rng(seed).choice([-1,1],size=(10000,len(values)))
    return (int((abs((signs*values).mean(1))>=threshold).sum())+1)/10001
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
    seed=0 if run==pilot else 20261005
    grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:grouped[(row['dataset'],row['method'])][row['task_id']].append(row)
    for r in summary:
        selected=[v for v in rows if (v['dataset'],v['method'])==(r['dataset'],r['method'])];groups=defaultdict(list)
        for v in selected:groups[v['task_id']].append(v)
        assert all(len(group)==3 for group in groups.values())
        for metric in metrics:
            if metric in r:
                values=[np.mean([v[metric] for v in group]) for group in groups.values()]
                assert np.allclose(bootstrap(values,seed),[float(r[k]) for k in [metric,metric+'_ci_low',metric+'_ci_high']],rtol=0,atol=1e-9),(run.name,r['method'],metric)
                intervals+=1
    with open(run/'results/statistics.csv',encoding='utf-8') as f:stats=list(csv.DictReader(f))
    pvalues=[]
    for stat in stats:
        other=stat['comparison'].removeprefix('CRAVE minus ');metric=stat['metric']
        left=grouped[(stat['dataset'],'CRAVE')];right=grouped[(stat['dataset'],other)];assert set(left)==set(right)
        taskids=list(left) if run==pilot else sorted(left)
        delta=[np.mean([r[metric] for r in left[t]])-np.mean([r[metric] for r in right[t]]) for t in taskids]
        assert np.allclose(bootstrap(delta,seed),[float(stat[k]) for k in ['estimate','ci_low','ci_high']],rtol=0,atol=1e-9)
        value=permutation(delta,seed);assert abs(value-float(stat['raw_p']))<1e-12;pvalues.append(value);comparisons+=1
    order=np.argsort(pvalues);adjusted=np.zeros(len(pvalues));running=0
    for rank,index in enumerate(order):
        running=max(running,(len(pvalues)-rank)*pvalues[index]);adjusted[index]=min(running,1)
    assert np.allclose(adjusted,[float(r['holm_p']) for r in stats],rtol=0,atol=1e-12)
    checked+=n
with open(ROOT/'tmp/full_manuscript_assets/question_influence.csv',encoding='utf-8') as f:influence=list(csv.DictReader(f))
assert len(influence)==80
for record in influence:
    left=grouped[(record['dataset'],'CRAVE')];right=grouped[(record['dataset'],'confidence')]
    scale=100 if record['metric']=='exact_match' else 1
    delta={t:scale*(np.mean([r[record['metric']] for r in left[t]])-np.mean([r[record['metric']] for r in right[t]])) for t in left}
    assert np.allclose([np.mean(list(delta.values())),np.mean([v for t,v in delta.items() if t!=record['omitted_task_id']])],[float(record['full_estimate']),float(record['leave_one_out_estimate'])],rtol=0,atol=1e-9)
manifest=json.loads((ROOT/'analysis/release_manifest.json').read_text())
for name,expected in manifest['files'].items():assert file_hash(ROOT/name)==expected,name
print(json.dumps({'passed':True,'numeric_episodes':checked,'summary_metrics_with_intervals':intervals,'paired_comparisons':comparisons,'question_deletion_estimates':len(influence),'framework_source_hash':code_hash(),'frozen_checkpoints':3,'release_files':len(manifest['files']),'scope':'Numerical release integrity; raw answer recomputation requires retained local records'}))
