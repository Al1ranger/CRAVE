"""Resume the unchanged development matrix with four independent server slots.

Only execution concurrency changes. Preserved rows are never overwritten.
Record the runtime amendment before generating any additional branch.
"""
import concurrent.futures,datetime,json,pathlib,threading,time,re
import numpy as np
import audit
from crave.agent.environment import Environment
from crave.agent.checkpoint import checkpoint,restore
from crave.baselines.controllers import FixedPolicy,ValuePolicy
from crave.common import digest,file_hash,read_json,save_json
from crave.counterfactual.branch import rollout
from crave.evaluation.task_metrics import answer_metrics,future_cost

OUT=audit.OUT;LOCAL=threading.local();ENCODER_LOCK=threading.RLock()

def main():
    tasks,freeze=audit.prepare()
    declaration={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'reason':'serial throughput degraded by a concurrent independent p5 CPU job; no result-dependent change',
                 'previous_freeze':freeze['sha256'],'source_sha256':file_hash(__file__),
                 'changes':{'server_parallel_slots':4,'server_total_context':16384,'per_slot_context':4096,'client_workers':4},
                 'unchanged':['tasks','budgets','checkpoint rules','models','policies','seeds','K candidates','gate thresholds','outcomes already journaled'],
                 'record_policy':'append missing branch keys only; initial serial records retained',
                 'heldout_inference':False}
    save_json(OUT/'runtime_amendment.json',declaration)
    env,policies=audit.make_environment(tasks)
    original_encode=env.encoder.encode
    def locked_encode(texts):
        with ENCODER_LOCK:return original_encode(texts)
    env.encoder.encode=locked_encode
    backend_instances=[]
    def worker():
        if not hasattr(LOCAL,'env'):
            backend=audit.TapedBackend(model_hash=freeze['config']['model_sha256'],cache=False)
            LOCAL.env=Environment(backend,env.retriever,env.encoder,max_steps=12)
            LOCAL.policy=ValuePolicy(policies['pi'].model,policies['pi'].features,.1)
            backend_instances.append(backend)
        return LOCAL.env,{'mu':policies['mu'],'pi':LOCAL.policy}
    path=OUT/'branches.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    completed={r['key'] for r in rows}
    cps=read_json(OUT/'checkpoints.json') if (OUT/'checkpoints.json').exists() else []
    def branch(job):
        cp,task,group,rep,action,key=job
        wenv,wpolicies=worker();policy=wpolicies['mu' if 'mu' in group else 'pi']
        seed=audit.seed_for('matched' if group.startswith('matched') else group,cp['id'],rep,
                            'common' if group.startswith('matched') else action)
        before=restore(cp['saved']);state=restore(cp['saved']);state.seed=seed
        wenv.backend.tape=seed if group.startswith('matched') else None;wenv.backend.tape_position=0
        decision=restore(cp['saved']);decision.seed=seed
        chosen=policy(decision,cp['feasible'])
        wenv.action(state,action)
        if not state.terminal:rollout(wenv,state,policy)
        return {'key':key,'checkpoint':cp['id'],'state_hash':cp['saved']['hash'],'task_id':task['task_id'],
                'dataset':task['dataset'],'group':group,'replicate':rep,'action':action,'branch_seed':seed,
                'continuation':'mu0-v1' if 'mu' in group else 'legacy-CRAVE-17-lambda0.1',
                'chosen_at_checkpoint':chosen,**answer_metrics(state.candidate_answer,task['answer'],task.get('aliases',[])),
                'cost':future_cost(before,state),'final_state':state.as_dict(),'runtime_version':'four-slot-amendment'}
    for stage in [0,1]:
        if stage==1:
            gate=audit.variance_gate(rows);save_json(OUT/'initial_variance_gate.json',gate)
            if gate['passed']:break
        stage_cps=[]
        for task in tasks:
            cp_id=task['task_id']+':'+str(stage)
            cp=next((r for r in cps if r['id']==cp_id),None)
            if cp is None:
                env.backend.tape=None
                state=env.initial(task,freeze['config']['budget'],17)
                if stage==1:env.action(state,'RETRIEVE')
                env.action(state,'THINK')
                cp={'id':cp_id,'task_id':task['task_id'],'dataset':task['dataset'],'stage':stage,
                    'saved':checkpoint(state),'feasible':env.feasible(state),'runtime_version':'four-slot-amendment'}
                cps.append(cp);save_json(OUT/'checkpoints.json',cps)
            stage_cps.append((cp,task))
        jobs=[]
        for cp,task in stage_cps:
            for group in freeze['config']['replicate_sets']:
                for rep in range(5):
                    for action in cp['feasible']:
                        key=digest([cp['id'],group,rep,action])
                        if key not in completed:jobs.append((cp,task,group,rep,action,key))
        jobs.sort(key=lambda job:digest(['execution-order-20261011',job[-1]]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(branch,jobs):
                audit.journal(path,row);rows.append(row);completed.add(row['key'])
                save_json(OUT/'progress.json',{'phase':'parallel branches','stage':stage,'completed':len(rows),
                    'planned_this_stage':sum(len(cp['feasible'])*30 for cp,task in stage_cps),'checkpoint':row['checkpoint'],
                    'physical_calls_this_invocation':sum(b.calls for b in backend_instances)+env.backend.calls,
                    'output_tokens_this_invocation':sum(b.physical_tokens for b in backend_instances)+env.backend.physical_tokens})
    all_backends=[env.backend]+backend_instances
    completion={'status':'complete','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'branches':len(rows),'checkpoints':len(cps),
        'physical_calls_this_invocation':sum(b.calls for b in all_backends),
        'physical_output_tokens_this_invocation':sum(b.physical_tokens for b in all_backends),
        'physical_generation_seconds_this_invocation':sum(b.physical_seconds for b in all_backends),
        'runtime_amendment':str(OUT/'runtime_amendment.json')}
    # Server timings include discarded setup work and any interrupted branch.
    logs=[OUT/'initial_server.stderr.log',OUT/'parallel_server.stderr.log']
    text='\n'.join(p.read_text(errors='replace') for p in logs if p.exists())
    totals=re.findall(r'total time\s*=\s*([\d.]+) ms',text)
    outputs=re.findall(r'(?<!prompt )eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens',text)
    # Negative lookbehind excludes the prompt-evaluation line.
    completion['server_recorded_generation_calls_all_attempts']=len(totals)
    completion['server_recorded_output_tokens_all_attempts']=sum(map(int,outputs))
    completion['server_recorded_generation_seconds_all_attempts']=sum(map(float,totals))/1000
    completion['timing_scope']='sum of server prompt+decode durations; includes failed setup and interrupted branch; excludes transport, startup, and non-generation work'
    save_json(OUT/'completion.json',completion);save_json(OUT/'variance_gate.json',audit.variance_gate(rows))
    audit.analyze()

if __name__=='__main__':main()
