"""Reconstruct continuation statistics without importing either analysis script."""
import collections,itertools,json,math,pathlib
import numpy as np
from scipy.stats import t
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def mean(xs):return sum(xs)/len(xs)
def var(xs):
    m=mean(xs)
    return sum((x-m)**2 for x in xs)/(len(xs)-1)
def tau(x,y):
    products=[np.sign(x[a]-x[b])*np.sign(y[a]-y[b]) for a,b in itertools.combinations(x,2)]
    nx=sum(x[a]!=x[b] for a,b in itertools.combinations(x,2))
    ny=sum(y[a]!=y[b] for a,b in itertools.combinations(y,2))
    return sum(products)/math.sqrt(nx*ny) if nx*ny else None
def close(a,b):
    if a is None or b is None:assert a is None and b is None
    else:np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)

def main():
    assert read('independent_verification.json')['passed']
    raw_path=OUT/'branches.jsonl'
    raw=[json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else json.loads((OUT.parent/'sanitized_branch_outcomes.json').read_text())
    indexed=collections.defaultdict(list)
    for row in raw:indexed[(row['checkpoint'],row['group'],row['action'])].append(row)
    for records in indexed.values():records.sort(key=lambda r:r['replicate'])
    checkpoints=read('checkpoints.json' if (OUT/'checkpoints.json').exists() else 'sanitized_checkpoints.json')
    chosen_k=read('summary.json')['K_selection']['reported_K']
    expected_stability=[];checks=0;all_primary=read('checkpoint_metrics_K3_K5.json')
    for utility,k_values in [(False,[3,5]),(True,[chosen_k])]:
        for k in k_values:
            expected=[]
            for cp in checkpoints:
                actions=cp['feasible'];vectors={};averages={}
                for group in ['ind_mu_A','ind_mu_B','ind_pi_A','ind_pi_B','matched_mu','matched_pi']:
                    vectors[group]={}
                    for action in actions:
                        records=indexed[(cp['id'],group,action)][:k]
                        vectors[group][action]=[int(r['exact_match']*40960-r['cost'][0]-1024*r['cost'][1]-2048*r['cost'][2]) if utility else r['exact_match'] for r in records]
                    averages[group]={a:sum(v)/(len(v)*(40960 if utility else 1)) for a,v in vectors[group].items()}
                mu={a:sum(vectors['ind_mu_A'][a]+vectors['ind_mu_B'][a])/(2*k*(40960 if utility else 1)) for a in actions}
                pi={a:sum(vectors['ind_pi_A'][a]+vectors['ind_pi_B'][a])/(2*k*(40960 if utility else 1)) for a in actions}
                best=lambda values:max(actions,key=values.get)
                think='THINK' in actions
                result={'checkpoint':cp['id'],'task_id':cp['task_id'],
                        'signed_THINK_bias':mu['THINK']-pi['THINK'] if think else None,
                        'absolute_discrepancy':mean([abs(mu[a]-pi[a]) for a in actions]),
                        'argmax_change':float(best(mu)!=best(pi)),
                        'kendall_tau_b':tau(mu,pi),
                        'THINK_STOP_rank_reversal':float((mu['THINK']-mu['STOP'])*(pi['THINK']-pi['STOP'])<0) if think else None,
                        'observed_squared_discrepancy':mean([(mu[a]-pi[a])**2 for a in actions]),
                        'estimated_noise':mean([(averages['ind_mu_A'][a]-averages['ind_mu_B'][a])**2/4+(averages['ind_pi_A'][a]-averages['ind_pi_B'][a])**2/4 for a in actions]),
                        'estimated_squared_shift':mean([(averages['ind_mu_A'][a]-averages['ind_pi_A'][a])*(averages['ind_mu_B'][a]-averages['ind_pi_B'][a]) for a in actions])}
                iv=[];mv=[]
                for continuation in ['mu','pi']:
                    independent=vectors['ind_'+continuation+'_A'];matched=vectors['matched_'+continuation]
                    if think:
                        for action in actions:
                            if action=='THINK':continue
                            iv.append((var(independent['THINK'])+var(independent[action]))/(k*(40960**2 if utility else 1)))
                            mv.append(var([a-b for a,b in zip(matched['THINK'],matched[action])])/(k*(40960**2 if utility else 1)))
                    if not utility:
                        a,b=averages['ind_'+continuation+'_A'],averages['ind_'+continuation+'_B'];widths=[]
                        for group in ['ind_'+continuation+'_A','ind_'+continuation+'_B']:
                            order=sorted(actions,key=lambda a:-averages[group][a]);v1,v2=[var(vectors[group][x])/k for x in order[:2]]
                            df=(v1+v2)**2/((v1*v1+v2*v2)/(k-1)) if v1+v2 else math.inf
                            widths.append(float(t.ppf(.975,df)*math.sqrt(v1+v2)))
                        expected_stability.append({'checkpoint':cp['id'],'continuation':continuation,'K':k,
                            'argmax_agreement':int(best(a)==best(b)),'precision_pass':int(max(widths)<=.1),
                            'halfwidth_A':widths[0],'halfwidth_B':widths[1]})
                result.update(independent_gap_variance=mean(iv) if iv else None,matched_gap_variance=mean(mv) if mv else None)
                if utility:
                    result['THINK_any_action_rank_reversal']=float(any((mu['THINK']-mu[a])*(pi['THINK']-pi[a])<0 for a in actions if a!='THINK')) if think else None
                else:
                    candidate_nonempty=cp.get('candidate_answer_nonempty') if 'candidate_answer_nonempty' in cp else bool(cp['saved']['payload']['candidate_answer'])
                    weights={'THINK':.35,'RETRIEVE':.35,'VERIFY':.2,'STOP':.1 if candidate_nonempty else .02}
                    chosen=indexed[(cp['id'],'ind_pi_A',actions[0])][0]['chosen_at_checkpoint']
                    result['behavioral_disagreement']=1-weights[chosen]/sum(weights[a] for a in actions)
                    assert result['THINK_STOP_rank_reversal'] in [0,None]
                saved=(read('utility_checkpoint_metrics.json') if utility else all_primary[str(k)])
                recorded=next(r for r in saved if r['checkpoint']==cp['id'])
                for key,value in result.items():
                    if key in ['checkpoint','task_id']:continue
                    try:close(value,recorded[key])
                    except AssertionError as error:raise AssertionError(f'{cp["id"]} K={k} utility={utility} metric={key}: {error}') from error
                    checks+=1
                for key,values in [('action_values_mu',mu),('action_values_pi',pi)]:
                    for action,value in values.items():close(value,recorded[key][action]);checks+=1
                expected.append(result)
            if utility or k==chosen_k:
                summary=read('utility_summary.json' if utility else 'summary.json')
                # Reproduce exact RNG allocation in the declared metric order.
                metric_order=list((read('utility_checkpoint_metrics.json') if utility else read('checkpoint_metrics.json'))[0])
                rng=np.random.default_rng(20261011)
                for metric in metric_order:
                    if metric not in expected[0] or metric in ['checkpoint','task_id']:continue
                    q_values=[]
                    for question in sorted({r['task_id'] for r in expected}):
                        xs=[r[metric] for r in expected if r['task_id']==question and r[metric] is not None]
                        if xs:q_values.append(mean(xs))
                    if not q_values:close(None,summary[metric]['estimate']);continue
                    values=np.array(q_values);sample=rng.integers(0,len(values),(10000,len(values)))
                    bootstrap=values[sample].mean(axis=1)
                    close(mean(q_values),summary[metric]['estimate'])
                    close(np.percentile(bootstrap,[2.5,97.5]),summary[metric]['ci95']);checks+=3
    recorded_stability=read('stability.json')
    for row in expected_stability:
        saved=next(r for r in recorded_stability if all(r[k]==row[k] for k in ['checkpoint','continuation','K']))
        for key in ['argmax_agreement','precision_pass','halfwidth_A','halfwidth_B']:close(row[key],saved[key]);checks+=1
    for candidate in read('summary.json')['K_selection']['candidates']:
        for policy in ['mu','pi']:
            rows=[r for r in expected_stability if r['K']==candidate['K'] and r['continuation']==policy]
            close(mean([r['argmax_agreement'] for r in rows]),candidate['by_continuation'][policy]['argmax_agreement'])
            close(mean([r['precision_pass'] for r in rows]),candidate['by_continuation'][policy]['precision_fraction'])
    report={'passed':True,'numeric_checks':checks,'resamples':10000,
            'input_mode':'full local branch journal' if raw_path.exists() else 'sanitized numerical release; raw answer rescoring requires local originals',
            'checks':['both K candidates','independent scalar reconstruction','manual Kendall tau-b pair counts',
                      'Welch stability','question-cluster bootstrap with declared RNG order','primary EM reversal endpoint constraint'],
            'scope':'reconstruction from retained branch observations; no new inference or score edits'}
    if raw_path.exists():(OUT/'statistics_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
