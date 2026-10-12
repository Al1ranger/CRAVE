"""Fresh development-only continuation audit. Journals commit after every branch.

No held-out inference. Independent primary A/B sets and a separate CRN audit.
The original executor and learned legacy checkpoint are kept unchanged.
"""
import argparse, copy, datetime, json, pathlib, platform, sys, time
import numpy as np
from scipy.stats import kendalltau, t
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crave.common import digest, file_hash, read_json, save_json
from crave.agent.budget import Budget
from crave.agent.checkpoint import checkpoint, restore
from crave.agent.environment import Environment
from crave.agent.state import ACTIONS
from crave.baselines.controllers import MixedPolicy, FixedPolicy, ConfidencePolicy, ValuePolicy
from crave.counterfactual.branch import rollout
from crave.datasets.common import hotpot_rows, author_raw_rows, build_corpus
from crave.evaluation.task_metrics import answer_metrics, future_cost
from crave.models.backend import LlamaBackend
from crave.models.state_encoder import FrozenEncoder, StateFeatures
from crave.models.value_model import ValueModel

OUT = ROOT / 'experiments/mive_revision/artifacts/dev4b'
BASE = ROOT / 'crave'

def seed_for(*parts):
    return int(digest(list(parts))[:8], 16) % (2**31 - 1)

class TapedBackend(LlamaBackend):
    """CRN couples native sampler seeds by generation-call position in a branch.

    This is a seed tape, not a guarantee of identical token streams or RNG draws
    across prompts, paths, backends, or hardware. Primary sets do not use it.
    """
    tape=None
    tape_position=0
    def generate(self,system,user,max_tokens,seed,schema=None):
        if self.tape is not None:
            seed=seed_for('generation-tape',self.tape,self.tape_position)
            self.tape_position+=1
        return super().generate(system,user,max_tokens,seed,schema)

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'freeze.json').exists():
        return read_json(OUT / 'tasks.json'), read_json(OUT / 'freeze.json')
    previous = read_json(BASE / 'runs/expanded_8484164aef776a12/task_manifest.json')['tasks']
    old = read_json(BASE / 'runs/bdd2b8a990571bb1/evaluation.json')['data']
    excluded = {(r['dataset'], r['task_id']) for r in previous + old if r.get('split') != 'dev'}
    candidates = hotpot_rows(BASE/'data/hotpot_validation.json', 'dev', count=6)
    candidates += author_raw_rows(BASE/'data/musique_raw_dev.json', 'MuSiQue', 'dev', 6)
    candidates += author_raw_rows(BASE/'data/2wiki_raw_dev.json', '2WikiMultiHopQA', 'dev', 10000)
    candidates = [r for r in candidates if (r['dataset'], r['task_id']) not in excluded]
    tasks = []
    for dataset in ['HotpotQA', '2WikiMultiHopQA', 'MuSiQue']:
        rows = sorted((r for r in candidates if r['dataset'] == dataset), key=lambda r:digest([20261011, r['task_id']]))
        assert len(rows) >= 3, (dataset, len(rows))
        tasks += rows[:3]
    source_paths = [BASE/'agent/environment.py', BASE/'models/backend.py', BASE/'baselines/controllers.py',
                    BASE/'evaluation/task_metrics.py', BASE/'agent/checkpoint.py', pathlib.Path(__file__)]
    config = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'exploratory development measurement; not a held-out or baseline eligibility certification',
              'model':'Qwen3-4B-Q4_K_M', 'model_sha256':file_hash(BASE/'models_data/Qwen3-4B-Q4_K_M.gguf'),
              'controller_sha256':file_hash(BASE/'runs/bdd2b8a990571bb1/CRAVE_17.npz'),
              'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},
              'tasks_sha256':digest(tasks), 'corpus_sha256':digest(build_corpus(tasks)),
              'n_questions':len(tasks),'question_selection_seed':20261011,'prefix_seed':17,
              'budget':{'tokens':512,'retrievals':1,'verifications':1},
              'sampling':{'temperature':0.7,'top_p':0.8,'top_k':20},
              'think_reserve':256,'verify_reserve':192,'context_tokens':1400,'max_steps':12,
              'label_continuation':'mu0-v1','deployment_continuation':'legacy-CRAVE-17-lambda0.1',
              'checkpoint_stages':['after THINK without retrieval','if variance gate fails also after RETRIEVE THINK'],
              'gate':{'maximum_fraction_zero_action_reward_variance':0.90,'reward':'normalized answer EM'},
              'replicate_sets':['ind_mu_A','ind_mu_B','ind_pi_A','ind_pi_B','matched_mu','matched_pi'],
              'K_candidates':[3,5],'stability':{'argmax_agreement_min':0.90,'margin_halfwidth_max':0.10,'precision_fraction_min':0.90},
              'bootstrap':{'unit':'question; equal checkpoint weight within question','resamples':10000,'seed':20261011},
              'matched_tape':'shared native sampler seed by generation-call position; no cross-hardware equality claim',
              'cache':False,'hardware':platform.platform(),'heldout_inference_authorized_by_this_config':False}
    freeze={'config':config,'sha256':digest(config)}
    save_json(OUT/'tasks.json',tasks)
    save_json(OUT/'freeze.json',freeze)
    return tasks, freeze

def make_environment(tasks):
    encoder=FrozenEncoder(BASE/'models_data/encoder')
    from crave.retrieval.hybrid import HybridRetriever
    backend=TapedBackend(model_hash=file_hash(BASE/'models_data/Qwen3-4B-Q4_K_M.gguf'),cache=False)
    env=Environment(backend,HybridRetriever(build_corpus(tasks),encoder),encoder,max_steps=12)
    policy=ValuePolicy(ValueModel.load(BASE/'runs/bdd2b8a990571bb1/CRAVE_17.npz'),StateFeatures(encoder),0.1)
    return env, {'mu':MixedPolicy(),'pi':policy}

def journal(path, row):
    with open(path,'a',encoding='utf-8') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()

def run():
    tasks,freeze=prepare()
    assert digest(tasks)==freeze['config']['tasks_sha256']
    env,policies=make_environment(tasks)
    path=OUT/'branches.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    completed={r['key'] for r in rows}
    cp_path=OUT/'checkpoints.json'
    checkpoints=read_json(cp_path) if cp_path.exists() else []
    baseline_path=OUT/'baseline.jsonl'
    baselines=[json.loads(x) for x in baseline_path.read_text().splitlines()] if baseline_path.exists() else []
    for task in tasks:
        if task['task_id'] not in {r['task_id'] for r in baselines}:
            start=time.perf_counter()
            final=env.initial(task,freeze['config']['budget'],17)
            states=rollout(env,final,FixedPolicy(),keep_states=True)
            row={'task_id':task['task_id'],'dataset':task['dataset'],**answer_metrics(final.candidate_answer,task['answer'],task.get('aliases',[])),
                 'final_state':final.as_dict(),'elapsed_seconds':time.perf_counter()-start}
            journal(baseline_path,row); baselines.append(row)
            save_json(OUT/'progress.json',{'phase':'baseline','completed':len(baselines),'questions':len(tasks)})
    for stage in [0,1]:
        if stage==1:
            gate=variance_gate(rows)
            save_json(OUT/'initial_variance_gate.json',gate)
            if gate['passed']: break
        for task in tasks:
            cp_id=task['task_id']+':'+str(stage)
            existing=next((r for r in checkpoints if r['id']==cp_id),None)
            if existing is None:
                env.backend.tape=None
                state=env.initial(task,freeze['config']['budget'],17)
                if stage==1: env.action(state,'RETRIEVE')
                env.action(state,'THINK')
                existing={'id':cp_id,'task_id':task['task_id'],'dataset':task['dataset'],'stage':stage,'saved':checkpoint(state),'feasible':env.feasible(state)}
                checkpoints.append(existing); save_json(cp_path,checkpoints)
            saved=existing['saved']
            for group in freeze['config']['replicate_sets']:
                policy=policies['mu' if 'mu' in group else 'pi']
                for rep in range(5):
                    for action in existing['feasible']:
                        key=digest([cp_id,group,rep,action])
                        if key in completed:continue
                        # Primary seeds are disjoint by action, continuation, A/B, and checkpoint.
                        # CRN group shares a seed across actions and continuations in a separate audit.
                        seed=seed_for('matched' if group.startswith('matched') else group,cp_id,rep,
                                      'common' if group.startswith('matched') else action)
                        initial=restore(saved); state=restore(saved); state.seed=seed
                        env.backend.tape=seed if group.startswith('matched') else None
                        env.backend.tape_position=0
                        decision_state=restore(saved);decision_state.seed=seed
                        chosen=policy(decision_state,existing['feasible'])
                        env.action(state,action)
                        if not state.terminal:rollout(env,state,policy)
                        scores=answer_metrics(state.candidate_answer,task['answer'],task.get('aliases',[]))
                        row={'key':key,'checkpoint':cp_id,'state_hash':saved['hash'],'task_id':task['task_id'],
                             'dataset':task['dataset'],'group':group,'replicate':rep,'action':action,'branch_seed':seed,
                             'continuation': 'mu0-v1' if 'mu' in group else 'legacy-CRAVE-17-lambda0.1',
                             'chosen_at_checkpoint':chosen,**scores,'cost':future_cost(initial,state),'final_state':state.as_dict()}
                        journal(path,row);rows.append(row);completed.add(key)
                        save_json(OUT/'progress.json',{'phase':'branches','stage':stage,'completed':len(rows),'checkpoint':cp_id,
                                  'physical_calls_this_invocation':env.backend.calls,'output_tokens_this_invocation':env.backend.physical_tokens})
        # Run all prespecified checkpoints before examining the reward gate.
    save_json(OUT/'variance_gate.json',variance_gate(rows))
    save_json(OUT/'completion.json',{'status':'complete','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'branches':len(rows),'checkpoints':len(checkpoints),'physical_calls_this_invocation':env.backend.calls,
              'physical_output_tokens_this_invocation':env.backend.physical_tokens,'physical_generation_seconds_this_invocation':env.backend.physical_seconds})
    analyze()

def variance_gate(rows):
    ids=sorted({r['checkpoint'] for r in rows})
    constant=[]
    for cp in ids:
        selected=[r for r in rows if r['checkpoint']==cp and r['group']=='ind_mu_A']
        means=[np.mean([r['exact_match'] for r in selected if r['action']==a]) for a in ACTIONS if any(r['action']==a for r in selected)]
        constant.append(len(set(means))<=1)
    fraction=float(np.mean(constant)) if ids else 1.0
    return {'n_checkpoints':len(ids),'n_identical_action_means':sum(constant),'fraction':fraction,'passed':fraction<=0.90,
            'definition':'K=5 action mean normalized EM under independent mu A; development only'}

def analyze():
    rows=[json.loads(x) for x in (OUT/'branches.jsonl').read_text().splitlines()]
    checkpoints=read_json(OUT/'checkpoints.json')
    per_all={3:[],5:[]}; stability=[]
    for k in [3,5]:
        for cp in checkpoints:
            actions=cp['feasible']
            data={g:np.array([[r['exact_match'] for r in rows if r['checkpoint']==cp['id'] and r['group']==g and r['action']==a and r['replicate']<k] for a in actions]) for g in ['ind_mu_A','ind_mu_B','ind_pi_A','ind_pi_B','matched_mu','matched_pi']}
            if any(x.shape!=(len(actions),k) for x in data.values()):continue
            means={g:x.mean(1) for g,x in data.items()}
            for name in ['mu','pi']:
                a,b=means['ind_'+name+'_A'],means['ind_'+name+'_B']
                hws=[]
                for group in ['ind_'+name+'_A','ind_'+name+'_B']:
                    best,second=np.argsort(-means[group],kind='stable')[:2]
                    x,y=data[group][best],data[group][second]
                    vx,vy=x.var(ddof=1)/k,y.var(ddof=1)/k
                    df=(vx+vy)**2/(vx*vx/(k-1)+vy*vy/(k-1)) if vx+vy else float('inf')
                    hws.append(float(t.ppf(.975,df)*np.sqrt(vx+vy)))
                stability.append({'checkpoint':cp['id'],'continuation':name,'K':k,'argmax_agreement':int(np.argmax(a)==np.argmax(b)),
                                  'precision_pass':int(max(hws)<=.1),'halfwidth_A':hws[0],'halfwidth_B':hws[1],
                                  'note':'plug-in zero-variance intervals can be zero and do not certify rare-event precision'})
            ma,mb,pa,pb=[means[g] for g in ['ind_mu_A','ind_mu_B','ind_pi_A','ind_pi_B']]
            mu=(ma+mb)/2; pi=(pa+pb)/2
            ti=actions.index('THINK') if 'THINK' in actions else None
            si=actions.index('STOP')
            tau=kendalltau(mu,pi,variant='b').statistic
            gap=mu-pi
            # Unbiased estimate of squared shift using independent cross-set contrasts.
            noise=(ma-mb)**2/4+(pa-pb)**2/4
            signal=(ma-pa)*(mb-pb)
            ind_var=[]; matched_var=[]
            if ti is not None:
                for name in ['mu','pi']:
                    x=data['ind_'+name+'_A']; z=data['matched_'+name]
                    for j in range(len(actions)):
                        if j==ti:continue
                        ind_var.append(float((x[ti].var(ddof=1)+x[j].var(ddof=1))/k))
                        matched_var.append(float((z[ti]-z[j]).var(ddof=1)/k))
            chosen_pi=next(r['chosen_at_checkpoint'] for r in rows if r['checkpoint']==cp['id'] and r['group']=='ind_pi_A')
            weights={'THINK':.35,'RETRIEVE':.35,'VERIFY':.20,'STOP':.10 if cp['saved']['payload']['candidate_answer'] else .02}
            disagreement=1-weights[chosen_pi]/sum(weights[a] for a in actions)
            per_all[k].append({'checkpoint':cp['id'],'task_id':cp['task_id'],'dataset':cp['dataset'],'K':k,
                        'signed_THINK_bias':float(gap[ti]) if ti is not None else None,
                        'absolute_discrepancy':float(np.abs(gap).mean()),
                        'argmax_change':float(np.argmax(mu)!=np.argmax(pi)),
                        'kendall_tau_b':float(tau) if np.isfinite(tau) else None,
                        'THINK_STOP_rank_reversal':float((mu[ti]-mu[si])*(pi[ti]-pi[si])<0) if ti is not None else None,
                        'behavioral_disagreement':float(disagreement),
                        'observed_squared_discrepancy':float((gap*gap).mean()),
                        'estimated_noise':float(noise.mean()),'estimated_squared_shift':float(signal.mean()),
                        'independent_gap_variance':float(np.mean(ind_var)) if ind_var else None,
                        'matched_gap_variance':float(np.mean(matched_var)) if matched_var else None,
                        'action_values_mu':dict(zip(actions,mu.tolist())),'action_values_pi':dict(zip(actions,pi.tolist()))})
    selection=[]
    for k in [3,5]:
        s=[r for r in stability if r['K']==k]
        by_policy={}
        for name in ['mu','pi']:
            subset=[r for r in s if r['continuation']==name]
            agreement=float(np.mean([r['argmax_agreement'] for r in subset])) if subset else 0
            precision=float(np.mean([r['precision_pass'] for r in subset])) if subset else 0
            by_policy[name]={'argmax_agreement':agreement,'precision_fraction':precision}
        selection.append({'K':k,'by_continuation':by_policy,'passed':all(x['argmax_agreement']>=.9 and x['precision_fraction']>=.9 for x in by_policy.values())})
    selected=next((r['K'] for r in selection if r['passed']),None)
    report_k=selected or 5
    per=per_all[report_k]
    save_json(OUT/'checkpoint_metrics.json',per)
    save_json(OUT/'checkpoint_metrics_K3_K5.json',per_all)
    save_json(OUT/'stability.json',stability)
    rng=np.random.default_rng(20261011); summary={}
    metrics=[x for x in per[0] if x not in ['checkpoint','task_id','dataset','K','action_values_mu','action_values_pi']] if per else []
    for metric in metrics:
        grouped={q:[r[metric] for r in per if r['task_id']==q and r[metric] is not None] for q in sorted({r['task_id'] for r in per})}
        values=np.array([np.mean(v) for v in grouped.values() if v])
        if not len(values):summary[metric]={'estimate':None,'ci95':None,'n_questions':0};continue
        boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
        summary[metric]={'estimate':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n_questions':len(values)}
    summary['K_selection']={'candidates':selection,'selected':selected,
                           'reported_K':report_k,'exploratory_if_failed':True}
    summary['counts']={'branches':len(rows),'checkpoints':len(per),'questions':len({r['task_id'] for r in per})}
    summary['variance_gate']=variance_gate(rows)
    save_json(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','analyze']);args=p.parse_args()
    {'prepare':prepare,'run':run,'analyze':analyze}[args.mode]()
