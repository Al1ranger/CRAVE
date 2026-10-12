"""Independent reconstruction of saved branches; no generation or data changes."""
import json,pathlib,sys
from collections import Counter
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crave.common import digest,read_json,save_json,file_hash
from crave.agent.checkpoint import restore
from crave.evaluation.task_metrics import answer_metrics,future_cost
from crave.agent.state import State
from crave.agent.budget import Cost
OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'

def verify():
    tasks={r['task_id']:r for r in read_json(OUT/'tasks.json')}
    freeze=read_json(OUT/'freeze.json')
    assert digest(list(tasks.values()))==freeze['config']['tasks_sha256']
    for relative,expected in freeze['config']['source_sha256'].items():
        assert file_hash(ROOT/relative.replace('\\','/'))==expected
    assert file_hash(ROOT/'crave/models_data/Qwen3-4B-Q4_K_M.gguf')==freeze['config']['model_sha256']
    assert file_hash(ROOT/'crave/runs/bdd2b8a990571bb1/CRAVE_17.npz')==freeze['config']['controller_sha256']
    cps={r['id']:r for r in read_json(OUT/'checkpoints.json')}
    rows=[json.loads(x) for x in (OUT/'branches.jsonl').read_text().splitlines()]
    from crave.datasets.common import hotpot_rows,author_raw_rows
    base=ROOT/'crave'
    historical=read_json(base/'runs/expanded_8484164aef776a12/task_manifest.json')['tasks']
    historical+=hotpot_rows(base/'data/hotpot_validation.json','pilot',start=6,count=3)
    historical+=author_raw_rows(base/'data/2wiki_raw_dev.json','2WikiMultiHopQA','pilot',3)
    question_key=lambda r:' '.join(r['question'].lower().split())
    old_questions={question_key(r) for r in historical}
    old_ids={(r['dataset'],r['task_id']) for r in historical}
    assert all(question_key(r) not in old_questions and (r['dataset'],r['task_id']) not in old_ids for r in tasks.values())
    keys=set();primary_seeds=set();primary_generation_seeds=set();shared={};counts=Counter()
    for row in rows:
        assert row['key'] not in keys;keys.add(row['key'])
        cp=cps[row['checkpoint']];before=restore(cp['saved']);after=State.from_dict(row['final_state'])
        assert row['key']==digest([row['checkpoint'],row['group'],row['replicate'],row['action']])
        parts=['matched' if row['group'].startswith('matched') else row['group'],row['checkpoint'],row['replicate'],
               'common' if row['group'].startswith('matched') else row['action']]
        assert row['branch_seed']==int(digest(parts)[:8],16)%(2**31-1)
        assert after.seed==row['branch_seed']
        assert row['state_hash']==cp['saved']['hash']
        assert row['action'] in cp['feasible']
        assert after.terminal
        np.testing.assert_allclose(future_cost(before,after),row['cost'],atol=1e-12,rtol=0)
        task=tasks[row['task_id']]
        scores=answer_metrics(after.candidate_answer,task['answer'],task.get('aliases',[]))
        assert all(row[k]==v for k,v in scores.items())
        future=after.action_log[len(before.action_log):]
        assert future[0]['action']==row['action']
        assert sum(r['token_cost'] for r in future)==row['cost'][0]
        assert sum(r['retrieval_cost'] for r in future)==row['cost'][1]
        assert sum(r['verification_cost'] for r in future)==row['cost'][2]
        assert min(after.budget.tokens,after.budget.retrievals,after.budget.verifications)>=0
        assert row['cost'][0]<=before.budget.tokens
        if row['action']=='STOP':
            assert after.candidate_answer==before.candidate_answer
            assert row['cost'][:3]==[0,0,0]
            assert len(future)==1
        if row['group'].startswith('ind'):
            assert row['branch_seed'] not in primary_seeds;primary_seeds.add(row['branch_seed'])
            for operation in future:
                seed=operation.get('metadata',{}).get('generation_seed')
                if seed is not None:
                    seed=int(seed)%(2**31-1)
                    assert seed not in primary_generation_seeds,'Colliding independent native generation seeds'
                    primary_generation_seeds.add(seed)
        else:
            pair=(row['checkpoint'],row['replicate'])
            assert shared.setdefault(pair,row['branch_seed'])==row['branch_seed']
        counts[(row['checkpoint'],row['group'],row['action'])]+=1
    assert all(v==5 for v in counts.values())
    assert len(rows)==sum(len(cp['feasible'])*6*5 for cp in cps.values())
    # Bound prerequisites checked on saved outcomes, separately from theory.
    report={'passed':True,'branches':len(rows),'checkpoints':len(cps),'questions':len(tasks),
            'budget_violations':0,'duplicate_primary_seeds':0,'scores_recomputed':len(rows),
            'distinct_primary_generation_seeds':len(primary_generation_seeds),
            'checks':['frozen task, source, model and controller hashes','question text and ID disjoint from both legacy evaluations','checkpoint restore digest','terminal outcome','future resources','answer EM/F1','deterministic STOP with zero constrained-resource charge',
                      'action feasibility at intervention','complete matrix','disjoint primary seeds','shared CRN seeds'],
            'limits':['Independent seed namespaces do not prove independence of backend pseudorandom streams.',
                      'No held-out efficacy or uniform policy-shift bound is certified.',
                      'Question-cluster intervals condition on observed finite-K estimates.']}
    save_json(OUT/'independent_verification.json',report);print(json.dumps(report))

if __name__=='__main__':verify()
