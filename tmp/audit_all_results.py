"""Recompute saved research outputs without language-model inference or edits."""
from pathlib import Path
from collections import defaultdict, Counter
from argparse import Namespace
import csv, hashlib, json, re, string, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from crave.pipeline import task_data, code_hash
from crave.datasets.common import build_corpus
from crave.agent.state import State, ACTIONS
from crave.models.state_encoder import FrozenEncoder, StateFeatures
from crave.models.value_model import ValueModel

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with open(path,encoding='utf-8') as f:return list(csv.DictReader(f))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def close(a,b,label,tolerance=1e-9):assert np.allclose(a,b,rtol=0,atol=tolerance),(label,a,b)
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
def holm(values):
    order=np.argsort(values);adjusted=np.zeros(len(values));running=0
    for rank,i in enumerate(order):
        running=max(running,(len(values)-rank)*values[i]);adjusted[i]=min(1,running)
    return adjusted
def normalize(value):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value.lower().translate(str.maketrans('','',string.punctuation))).split())
def scores(prediction,task):
    pred=normalize(prediction);best=[0.,0.,0.]
    for gold in [task['answer'],*task.get('aliases',[])]:
        target=normalize(gold);em=float(pred==target);common=sum((Counter(pred.split())&Counter(target.split())).values())
        f1=em if pred in ['yes','no','noanswer'] or target in ['yes','no','noanswer'] or not common else 2*common/(len(pred.split())+len(target.split()))
        best=[max(best[0],em),max(best[1],f1),max(best[2],em)]
    return best
pilot=ROOT/'crave/runs/bdd2b8a990571bb1';expanded=ROOT/'crave/runs/expanded_8484164aef776a12'
train,dev,test=task_data(Namespace(train_tasks=2,eval_tasks=3))
pf=read(pilot/'FROZEN_PILOT_CONFIG.json');cfg=pf['config']
assert digest(cfg)==pf['sha256'] and code_hash()==cfg['code_hash']
for label,value in [('train',train),('dev',dev),('pilot_eval',test)]:assert digest(value)==cfg['task_hashes'][label]
documents=build_corpus(train+dev+test);assert digest(documents)==cfg['corpus_hash']
ef=read(expanded/'FROZEN_EXPANDED_CONFIG.json');prepared=read(expanded/'task_manifest.json')
assert digest(ef['config'])==ef['sha256'] and digest(prepared)==ef['config']['prepared_hash']
tasklookup={(t['dataset'],t['task_id']):t for t in train+dev+test+prepared['tasks']}
counts=Counter();metrics=['task_score','exact_match','f1','generated_tokens','input_tokens','total_tokens','retrieval_calls','verification_calls','latency','controller_latency','normalized_cost','budget_violation']
for run,n,seed,frozen in [(pilot,162,0,pf),(expanded,480,20261005,ef)]:
    payload=read(run/'evaluation.json');episodes=payload['data']
    assert digest(episodes)==payload['data_hash'] and payload['config_hash']==frozen['sha256'] and len(episodes)==n
    assert len({(r['dataset'],r['task_id'],r['method'],r['seed']) for r in episodes})==n
    per=rows(run/'results/per_instance.csv');assert len(per)==n
    for r,saved in zip(episodes,per):
        close([r[k] for k in ['exact_match','f1','task_score']],scores(r['final_answer'],tasklookup[(r['dataset'],r['task_id'])]),'answer score')
        for k,value in saved.items():
            if isinstance(r[k],(float,int)):close(float(value),r[k],('per_instance',k))
            else:assert value==('' if r[k] is None else str(r[k])),('per_instance',k)
        close(r['generated_tokens'],sum(a['token_cost'] for a in r['actions'])+r['routing_tokens'],'output counter')
        close(r['total_tokens'],r['input_tokens']+r['generated_tokens'],'total tokens')
        for counter,key in [('retrieval_calls','retrieval_cost'),('verification_calls','verification_cost')]:close(r[counter],sum(a[key] for a in r['actions']),counter)
        close(r['normalized_cost'],r['generated_tokens']/4096+r['retrieval_calls']/4+r['verification_calls']/2+r['latency']/60,'cost')
        assert r['generated_tokens']<=r['budget_tokens'] and r['retrieval_calls']<=r['budget_retrieval'] and r['verification_calls']<=r['budget_verification'] and r['budget_violation']==0
        assert r['actions'][-1]['action']=='STOP' and r['actions'][-1]['candidate_answer']==r['final_answer']
        counts['answer_scores_and_episode_counters']+=1
    bycell=defaultdict(lambda:defaultdict(list))
    for r in episodes:bycell[(r['dataset'],r['method'])][r['task_id']].append(r)
    for s in rows(run/'results/summary.csv'):
        groups=bycell[(s['dataset'],s['method'])];assert all({r['seed'] for r in g}=={17,29,43} for g in groups.values())
        for metric in metrics:
            if metric not in s:continue
            values=[np.mean([r[metric] for r in g]) for g in groups.values()]
            close(bootstrap(values,seed),[float(s[k]) for k in [metric,metric+'_ci_low',metric+'_ci_high']],('summary',run.name,s['method'],metric))
            counts['summary_metrics_with_intervals']+=1
    stats=rows(run/'results/statistics.csv');ps=[]
    for s in stats:
        other=s['comparison'].removeprefix('CRAVE minus ');left=bycell[(s['dataset'],'CRAVE')];right=bycell[(s['dataset'],other)]
        ids=list(left) if seed==0 else sorted(left);metric=s['metric'];assert set(left)==set(right)
        delta=[np.mean([r[metric] for r in left[t]])-np.mean([r[metric] for r in right[t]]) for t in ids]
        close(bootstrap(delta,seed),[float(s[k]) for k in ['estimate','ci_low','ci_high']],('paired bootstrap',run.name,s['comparison'],metric))
        p=permutation(delta,seed);close(p,float(s['raw_p']),'raw p');ps.append(p);counts['paired_comparisons']+=1
    close(holm(ps),[float(s['holm_p']) for s in stats],'Holm correction')
    compute=read(run/'compute_accounting.json')
    outkey='logical_inference_tokens' if run==pilot else 'logical_evaluation_output_tokens'
    inkey='logical_input_tokens' if run==pilot else 'logical_evaluation_input_tokens'
    assert sum(r['generated_tokens'] for r in episodes)==compute[outkey]
    assert sum(r['input_tokens'] for r in episodes)==compute[inkey]
oracle=[];matched=[]
for name,expected_split,collection_seed in [('round1','train',17),('round2','train',29),('oracle','pilot_eval',31415)]:
    stage=read(pilot/(name+'.json'));assert stage['config_hash']==pf['sha256'] and digest(stage['data'])==stage['data_hash']
    branchrows=stage['data'] if name=='oracle' else stage['data']['records']
    if name=='oracle':oracle=branchrows
    else:matched+=branchrows
    for r in branchrows:
        assert r['split']==expected_split and r['continuation_policy']=='mu0-v1'
        assert digest(r['state_before'])==r['state_hash'] and set(r['state_before'])==set(State.__dataclass_fields__)
        assert r['branch_seed']==int(digest({'checkpoint':r['state_hash'],'seed':collection_seed,'replicate':r['replicate']})[:8],16)%(2**31-1)
        close([r[k] for k in ['exact_match','f1','reward']],scores(r['final_answer'],tasklookup[(r['dataset'],r['task_id'])]),'branch score')
        close(r['future_cost'][:3],[sum(a[k] for a in r['action_log']) for k in ['token_cost','retrieval_cost','verification_cost']],'branch costs')
        counts['branch_payloads_scores_and_seeds']+=1
assert len(matched)==156
assert sum(r['future_cost'][0] for r in matched)==read(pilot/'compute_accounting.json')['logical_offline_branch_generated_tokens']
states={r['state_hash']:State.from_dict(r['state_before']) for r in oracle};prediction_errors=[]
encoder=FrozenEncoder(ROOT/'crave/models_data/encoder');models={};features={};diagnostics=rows(pilot/'results/counterfactual_diagnostics.csv')
for r in diagnostics:
    method,seed,weight=r['method'],int(r['seed']),float(r['lambda']);key=(method,seed)
    if key not in models:models[key]=ValueModel.load(pilot/f'{method}_{seed}.npz')
    fk=(r['state_hash'],method,weight)
    if fk not in features:features[fk]=StateFeatures(encoder,method)(states[r['state_hash']],weight)
    heads=models[key].predict(features[fk][None])[0,:24].reshape(4,6);index=ACTIONS.index(r['action'])
    prediction_errors.append(abs(float(r['predicted_q'])-float(heads[index,0])))
    close(float(r['predicted_q']),heads[index,0],'diagnostic model prediction',2e-6)
    close(float(r['predicted_success']),1/(1+np.exp(-heads[index,1])),'success prediction',2e-6)
    selected=[b for b in oracle if b['state_hash']==r['state_hash'] and b['forced_action']==r['action']]
    utility=[b['reward']-weight*sum(c/ref for c,ref in zip(b['future_cost'],[4096,4,2,60])) for b in selected]
    close(float(r['empirical_q']),np.mean(utility),'empirical utility');close(float(r['empirical_success']),np.mean([b['reward'] for b in selected]),'branch success')
    close(float(r['empirical_q_standard_error']),np.std(utility,ddof=1)/np.sqrt(len(utility)),'utility standard error')
    assert int(r['k'])==len(selected);counts['oracle_action_predictions']+=1
for s in rows(pilot/'results/value_diagnostic_summary.csv'):
    cell=[r for r in diagnostics if all(r[k]==s[k] for k in ['method','seed','lambda'])];unique={r['state_hash']:r for r in cell}
    for state_hash in unique:
        values=[r for r in cell if r['state_hash']==state_hash];chosen=unique[state_hash]['chosen_action'];oracle_action=unique[state_hash]['oracle_action']
        lookup={r['action']:r for r in values};assert float(lookup[chosen]['predicted_q'])==max(float(r['predicted_q']) for r in values)
        assert float(lookup[oracle_action]['empirical_q'])==max(float(r['empirical_q']) for r in values)
        close(float(unique[state_hash]['regret']),float(lookup[oracle_action]['empirical_q'])-float(lookup[chosen]['empirical_q']),'regret')
        assert int(unique[state_hash]['top1'])==int(chosen==oracle_action)
    probability=np.array([float(r['predicted_success']) for r in cell]);observed=np.array([float(r['empirical_success']) for r in cell]);bins=np.minimum((probability*10).astype(int),9);ece=0
    for bin_id in np.unique(bins):
        selected=bins==bin_id;ece+=selected.mean()*abs(probability[selected].mean()-observed[selected].mean())
        saved=next(r for r in rows(pilot/'results/success_reliability_bins.csv') if all(r[k]==s[k] for k in ['method','seed','lambda']) and int(r['bin'])==bin_id)
        close([probability[selected].mean(),observed[selected].mean(),selected.sum()],[float(saved[k]) for k in ['confidence','observed','n']],'reliability bin')
    expected=[np.mean([float(r['regret']) for r in unique.values()]),np.mean([float(r['top1']) for r in unique.values()]),np.sqrt(np.mean([(float(r['predicted_q'])-float(r['empirical_q']))**2 for r in cell])),np.mean((probability-observed)**2),ece]
    close(expected,[float(s[k]) for k in ['mean_regret','top1_agreement','q_rmse','brier_to_finite_k_success_mean','ece_to_finite_k_success_mean']],'diagnostic summary');counts['diagnostic_summary_cells']+=1
gold={}
for name in ['hotpot_train.json','hotpot_validation.json']:
    for r in read(ROOT/'crave/data'/name)['rows']:
        facts=r['supporting_facts'];titles=facts['title'] if isinstance(facts,dict) else [f[0] for f in facts]
        gold[('HotpotQA',r.get('id',r.get('_id')))]=set(t.casefold() for t in titles)
for name,dataset in [('musique_raw_train.json','MuSiQue'),('musique_raw_dev.json','MuSiQue'),('2wiki_raw_dev.json','2WikiMultiHopQA')]:
    for r in read(ROOT/'crave/data'/name)['rows']:gold[(dataset,r['id'])]=set(c['wikipedia_title'].casefold() for c in r['contexts'] if c.get('is_supporting'))
docs={d['doc_id']:d for d in documents};retrieval=rows(ROOT/'crave/artifacts/retrieval_pilot/per_instance.csv')
for r in retrieval:
    expected=gold[(r['dataset'],r['task_id'])];found=expected&{docs[d]['title'].casefold() for d in r['retrieved_doc_ids'].split('|')}
    close([len(expected),len(found),len(found)/len(expected),int(found==expected)],[float(r[k]) for k in ['gold_support_titles','retrieved_support_titles','support_title_recall','all_support_titles_retrieved']],'retrieval support');counts['retrieval_records']+=1
for s in rows(ROOT/'crave/artifacts/retrieval_pilot/summary.csv'):
    cell=[r for r in retrieval if r['dataset']==s['dataset']]
    close(bootstrap([float(r['support_title_recall']) for r in cell],0),[float(s[k]) for k in ['estimate','ci_low','ci_high']],'retrieval interval')
    close(float(s['all_support_title_rate']),np.mean([float(r['all_support_titles_retrieved']) for r in cell]),'all titles')
report={'passed':True,'counts':dict(counts),'budget_violations':0,'framework_source_hash':code_hash(),'limits':['Recomputes retained results; does not rerun stochastic language-model generation.','Normalized answer scoring verified against the implemented definition, not official benchmark evaluator equivalence.','Physical request counts and timings are retained invocation measurements; logical token totals were recomputed.','No population-level superiority, external validity, or journal acceptance is certified.']}
report['maximum_absolute_q_prediction_difference']=max(prediction_errors)
report['float32_prediction_tolerance']=2e-6
out=ROOT/'tmp/full_manuscript_assets/comprehensive_result_audit.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
