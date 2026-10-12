"""Secondary priced-value sensitivity, distinct from primary EM analysis.

Declared before completion of the first branch matrix, after execution began.
Prices reuse legacy lambda=.1, with latency omitted to obtain bounded returns.
No raw outcome or journal is modified. This is not a new trained controller.
"""
import datetime,json,pathlib,sys
import numpy as np
from scipy.stats import kendalltau
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crave.common import read_json,save_json,file_hash,digest
OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'

def declare():
    path=OUT/'secondary_analysis_declaration.json'
    if not path.exists():
        config={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'stage':'analysis amendment during development collection; no held-out inference',
                'objective':'EM - .1*(future_output_tokens/4096 + future_retrievals/4 + future_verifications/2)',
                'prices_fixed_from':'legacy FROZEN_PILOT_CONFIG.json lambda .1 and references (4096,4,2)',
                'latency_excluded':True,'return_range_bound':1+.1*(512/4096+1/4+1/2),
                'bootstrap_resamples':10000,'bootstrap_seed':20261011,'script_sha256':file_hash(__file__)}
        save_json(path,{'config':config,'sha256':digest(config)})

def value(row):
    c=row['cost']
    # Exact integer units prevent floating-point ties from creating reversals.
    assert row['exact_match'] in [0,1]
    assert all(int(x)==x for x in c[:3])
    return int(row['exact_match']*40960-c[0]-1024*c[1]-2048*c[2])

def analyze():
    assert (OUT/'completion.json').exists()
    rows=[json.loads(x) for x in (OUT/'branches.jsonl').read_text().splitlines()]
    save_json(OUT/'utility_analysis_implementation.json',{
        'source_sha256':file_hash(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'change':'Exact integer units of 1/40960 for unchanged secondary objective; means use integer sums before division',
        'reason':'Avoid arithmetic-order artifacts in tied action ranks; raw outcomes and prices unchanged',
        'scope':'analysis corrections after collection; raw outcomes unchanged',
        'replicate_pairing':'Sort by replicate ID before matched-action contrasts; parallel journal order is not a pairing key'})
    k=read_json(OUT/'summary.json')['K_selection']['reported_K']
    per=[]
    for cp in read_json(OUT/'checkpoints.json'):
        actions=cp['feasible']
        data={g:np.array([[value(r) for r in sorted(rows,key=lambda r:r['replicate']) if r['checkpoint']==cp['id'] and r['group']==g and r['action']==a and r['replicate']<k] for a in actions]) for g in ['ind_mu_A','ind_mu_B','ind_pi_A','ind_pi_B','matched_mu','matched_pi']}
        means={g:x.sum(1)/(40960*k) for g,x in data.items()}
        ma,mb,pa,pb=[means[g] for g in ['ind_mu_A','ind_mu_B','ind_pi_A','ind_pi_B']]
        mu=(data['ind_mu_A'].sum(1)+data['ind_mu_B'].sum(1))/(81920*k)
        pi=(data['ind_pi_A'].sum(1)+data['ind_pi_B'].sum(1))/(81920*k)
        ti=actions.index('THINK') if 'THINK' in actions else None;si=actions.index('STOP')
        tau=kendalltau(mu,pi,variant='b').statistic
        vi=[];vm=[]
        if ti is not None:
            for name in ['mu','pi']:
                x=data['ind_'+name+'_A'];z=data['matched_'+name]
                for j in range(len(actions)):
                    if j==ti:continue
                    vi.append((x[ti].var(ddof=1)+x[j].var(ddof=1))/(k*40960**2))
                    vm.append((z[ti]-z[j]).var(ddof=1)/(k*40960**2))
        per.append({'checkpoint':cp['id'],'task_id':cp['task_id'],'dataset':cp['dataset'],
                    'signed_THINK_bias':float(mu[ti]-pi[ti]) if ti is not None else None,
                    'absolute_discrepancy':float(np.abs(mu-pi).mean()),
                    'argmax_change':float(np.argmax(mu)!=np.argmax(pi)),
                    'kendall_tau_b':float(tau) if np.isfinite(tau) else None,
                    'THINK_STOP_rank_reversal':float((mu[ti]-mu[si])*(pi[ti]-pi[si])<0) if ti is not None else None,
                    'THINK_any_action_rank_reversal':float(any((mu[ti]-mu[j])*(pi[ti]-pi[j])<0 for j in range(len(actions)) if j!=ti)) if ti is not None else None,
                    'observed_squared_discrepancy':float(np.mean((mu-pi)**2)),
                    'estimated_noise':float(np.mean((ma-mb)**2/4+(pa-pb)**2/4)),
                    'estimated_squared_shift':float(np.mean((ma-pa)*(mb-pb))),
                    'independent_gap_variance':float(np.mean(vi)) if vi else None,
                    'matched_gap_variance':float(np.mean(vm)) if vm else None,
                    'action_values_mu':dict(zip(actions,mu.tolist())),'action_values_pi':dict(zip(actions,pi.tolist()))})
    rng=np.random.default_rng(20261011);summary={}
    for metric in [k for k in per[0] if k not in ['checkpoint','task_id','dataset','action_values_mu','action_values_pi']]:
        values=[]
        for q in sorted({r['task_id'] for r in per}):
            cell=[r[metric] for r in per if r['task_id']==q and r[metric] is not None]
            if cell:values.append(np.mean(cell))
        values=np.array(values)
        if len(values):
            boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            summary[metric]={'estimate':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n_questions':len(values)}
        else:summary[metric]={'estimate':None,'ci95':None,'n_questions':0}
    save_json(OUT/'utility_checkpoint_metrics.json',per)
    save_json(OUT/'utility_summary.json',summary)

if __name__=='__main__':
    declare()
    if '--declare-only' not in sys.argv:analyze()
