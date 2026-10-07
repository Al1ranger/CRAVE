"""Post-pilot held-out evaluation; frozen learners and settings, fresh questions."""
import argparse,csv,json,sys,time,urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
BASE=Path(r'D:/Study/Papers/p6')
sys.path.insert(0,str(BASE))
from crave.common import ROOT,digest,file_hash,read_json,save_json,log
from crave.pipeline import code_hash,task_data
from crave.datasets.common import hotpot_rows,author_raw_rows,build_corpus,validate_splits
from crave.models.backend import LlamaBackend
from crave.models.state_encoder import FrozenEncoder,StateFeatures
from crave.models.value_model import ValueModel
from crave.retrieval.hybrid import HybridRetriever
from crave.agent.environment import Environment
from crave.baselines.controllers import ValuePolicy,ConfidencePolicy,FixedPolicy,LLMRouter
from crave.counterfactual.branch import rollout
from crave.evaluation.task_metrics import answer_metrics,future_cost
from crave.evaluation.statistics import paired_bootstrap,paired_permutation,holm
from crave.reporting import write_csv
ORIGINAL=ROOT/'runs/bdd2b8a990571bb1'
INPUTS=ROOT/'artifacts/expanded_test/inputs'
SELECTION_SEED=20261005
METHODS=['CRAVE','confidence','fixed','llm_router']

def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'CRAVE-research/0.1'})
    for attempt in range(3):
        try:return urllib.request.urlopen(request,timeout=120)
        except (urllib.error.URLError,TimeoutError) as error:
            if attempt==2:raise
            log('public_source_retry',attempt=attempt+1,error=str(error))
            time.sleep(2)

def prepare():
    INPUTS.mkdir(parents=True,exist_ok=True)
    target=INPUTS/'hotpot_validation_extra.json'
    if not target.exists():
        url='https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset=40&length=100'
        if (INPUTS/'hotpot_response.json').exists():payload=read_json(INPUTS/'hotpot_response.json')
        else:
            with fetch(url) as response:payload=json.load(response)
        rows=[r['row'] for r in payload['rows']]
        assert len(rows)==100
        save_json(target,{'rows':rows,'source':url,'offset':40,'subset_checksum':digest(rows),'snapshot_to_repository_linkage':'NOT_CERTIFIED'})
    target=INPUTS/'2wiki_raw_dev_pool.json'
    if not target.exists():
        provenance_url='https://raw.githubusercontent.com/StonyBrookNLP/musique/main/.all_data_information.json'
        if (INPUTS/'author_provenance.json').exists():provenance=read_json(INPUTS/'author_provenance.json')
        else:
            with fetch(provenance_url) as response:provenance=json.load(response)
        file_id='1XtMGLeBC52PvR2VbQ1u3gkFpfLic1mwq'
        assert file_id in provenance.values()
        url=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
        rows=[]
        source=open(INPUTS/'2wiki_pool_raw.jsonl',encoding='utf-8') if (INPUTS/'2wiki_pool_raw.jsonl').exists() else fetch(url)
        with source as response:
            while len(rows)<128:
                line=response.readline(2*1024*1024)
                if not line:break
                item=json.loads(line)
                if not isinstance(item,dict) or 'contexts' not in item:raise ValueError('Invalid public raw record')
                rows.append(item)
        assert len(rows)==128
        save_json(target,{'rows':rows,'source':url,'author_provenance':provenance_url,'author_provenance_sha256':digest(provenance),'subset_checksum':digest(rows),'full_file_checksum':'NOT_DOWNLOADED_FULL_FILE','release_equivalence':'NOT_CERTIFIED'})
    frozen=read_json(ORIGINAL/'FROZEN_PILOT_CONFIG.json')
    args=argparse.Namespace(**frozen['config']['pilot_overrides'])
    train,dev,old_test=task_data(args)
    old=train+dev+old_test
    hotpot=hotpot_rows(ROOT/'data/hotpot_validation.json','expanded_eval')+hotpot_rows(INPUTS/'hotpot_validation_extra.json','expanded_eval')
    wiki=author_raw_rows(INPUTS/'2wiki_raw_dev_pool.json','2WikiMultiHopQA','expanded_eval',128)
    taken_ids={(t['dataset'],t['task_id']) for t in old}
    taken_questions={' '.join(t['question'].lower().split()) for t in old}
    taken_components={tuple(x) for t in old for x in t.get('source_question_ids',[])}
    rng=np.random.default_rng(SELECTION_SEED)
    selected=[];pool_counts={};excluded={}
    for dataset,pool in [('HotpotQA',hotpot),('2WikiMultiHopQA',wiki)]:
        valid=[]
        for t in pool:
            if (t['dataset'],t['task_id']) in taken_ids or ' '.join(t['question'].lower().split()) in taken_questions or any(tuple(x) in taken_components for x in t.get('source_question_ids',[])):continue
            valid.append(t)
        pool_counts[dataset]=len(valid);chosen=[]
        for i in rng.permutation(len(valid)):
            t=valid[int(i)];key=(t['dataset'],t['task_id']);question=' '.join(t['question'].lower().split());components={tuple(x) for x in t.get('source_question_ids',[])}
            if key in taken_ids or question in taken_questions or components&taken_components:continue
            chosen.append(t);taken_ids.add(key);taken_questions.add(question);taken_components.update(components)
            if len(chosen)==20:break
        assert len(chosen)==20,(dataset,len(chosen))
        selected+=chosen
    validate_splits(old+selected)
    documents=build_corpus(old+selected)
    payload={'tasks':selected,'original_partitions':{'train':train,'dev':dev,'pilot_eval':old_test},'documents':documents,'selection_seed':SELECTION_SEED,'eligible_pool_counts':pool_counts,'scope':'40 fresh questions; random draw from bounded public record pools, not a complete benchmark sample','source_files':{p.name:file_hash(p) for p in INPUTS.glob('*.json') if p.name!='prepared.json'}}
    save_json(INPUTS/'prepared.json',payload)
    log('expanded_data_ready',n_tasks=len(selected),n_documents=len(documents),eligible_pools=pool_counts)

def evaluate():
    prepared=read_json(INPUTS/'prepared.json');frozen=read_json(ORIGINAL/'FROZEN_PILOT_CONFIG.json');config=frozen['config']
    assert code_hash()==config['code_hash'],'Original implementation changed'
    manifest=read_json(ROOT/'artifacts/model_manifest.json')
    assert manifest==config['model_manifest']
    assert file_hash(ROOT/'models_data'/manifest['model']['file'])==manifest['model']['sha256']
    learners={str(seed):file_hash(ORIGINAL/f'CRAVE_{seed}.npz') for seed in config['training']['seeds']}
    for seed,sha in learners.items():assert sha==read_json(ORIGINAL/f'CRAVE_{seed}_training.json')['sha256']
    setup={'status':'POST_PILOT_EXPANDED_HELDOUT_NOT_FINAL_STUDY','parent_config_hash':frozen['sha256'],'framework_source_hash':config['code_hash'],'extension_script_hash':file_hash(__file__),'prepared_hash':digest(prepared),'tasks_hash':digest(prepared['tasks']),'corpus_hash':digest(prepared['documents']),'n_tasks_per_dataset':20,'n_documents':len(prepared['documents']),'selection_seed':SELECTION_SEED,'seeds':config['training']['seeds'],'methods':METHODS,'budget':config['budgets']['pilot_budgets'][0],'utility_lambda':.1,'confidence_threshold':config['experiments']['dev_threshold'],'controller_checkpoints':learners,'retraining':False,'development_tuning':False,'method_change_from_pilot':False,'corpus_change':'Expanded common transductive passage pool','bootstrap_resamples':10000,'paired_permutation_draws':10000,'statistical_family':'2 datasets x 3 comparators x (EM, output tokens, normalized cost); Holm 18 tests','cache_rule':'Cached requests retain observed generation latency; no cold latency claim'}
    identity=digest(setup);run=ROOT/'runs'/('expanded_'+identity[:16]);run.mkdir(parents=True,exist_ok=True)
    save_json(run/'FROZEN_EXPANDED_CONFIG.json',{'config':setup,'sha256':identity})
    save_json(run/'task_manifest.json',prepared)
    encoder=FrozenEncoder(ROOT/'models_data/encoder');retriever=HybridRetriever(prepared['documents'],encoder)
    backend=LlamaBackend(config['model']['base_url'],digest({'model':manifest['model']['sha256'],'runtime':manifest['runtime']['sha256']}))
    env=Environment(backend,retriever,encoder,max_steps=config['experiments']['max_steps'])
    features=StateFeatures(encoder);models={seed:ValueModel.load(ORIGINAL/f'CRAVE_{seed}.npz') for seed in setup['seeds']}
    rows=[];started=time.perf_counter();total=len(prepared['tasks'])*len(setup['seeds'])*len(METHODS)
    for ti,task in enumerate(prepared['tasks']):
        for seed in setup['seeds']:
            for method in METHODS:
                key=digest({'dataset':task['dataset'],'task_id':task['task_id'],'seed':seed,'method':method,'config':identity})
                path=run/'episodes'/(key+'.json')
                if path.exists():rows.append(read_json(path));continue
                policy={'CRAVE':lambda:ValuePolicy(models[seed],features,.1),'confidence':lambda:ConfidencePolicy(setup['confidence_threshold']),'fixed':FixedPolicy,'llm_router':lambda:LLMRouter(env)}[method]()
                initial=env.initial(task,setup['budget'],seed);state=env.initial(task,setup['budget'],seed);episode_start=time.perf_counter()
                rollout(env,state,policy)
                metrics=answer_metrics(state.candidate_answer,task['answer'],task.get('aliases',[]));consumed=future_cost(initial,state)
                row={'config_hash':identity,'dataset':task['dataset'],'task_id':task['task_id'],'split':'expanded_eval','method':method,'seed':seed,'budget_tokens':setup['budget']['tokens'],'budget_retrieval':setup['budget']['retrievals'],'budget_verification':setup['budget']['verifications'],'utility_lambda':.1,'final_answer':state.candidate_answer,**metrics,'generated_tokens':consumed[0],'input_tokens':state.input_tokens,'total_tokens':state.input_tokens+consumed[0],'retrieval_calls':consumed[1],'verification_calls':consumed[2],'latency':state.latency,'controller_latency':state.controller_latency,'normalized_cost':sum(consumed[j]/config['utility']['references'][j] for j in range(4)),'routing_tokens':state.routing_tokens,'context_tokens_added':state.context_tokens_added,'physical_episode_seconds':time.perf_counter()-episode_start,'budget_violation':int(min(state.budget.tokens,state.budget.retrievals,state.budget.verifications)<0),'actions':state.action_log,'status':setup['status']}
                save_json(path,row);rows.append(row)
                save_json(run/'progress.json',{'completed':len(rows),'total':total,'task':ti+1,'tasks':40,'physical_model_requests':backend.calls,'physical_output_tokens':backend.physical_tokens,'elapsed_seconds':time.perf_counter()-started})
                log('expanded_episode',completed=len(rows),total=total,dataset=task['dataset'],method=method,seed=seed,em=metrics['exact_match'],output_tokens=consumed[0])
    save_json(run/'evaluation.json',{'config_hash':identity,'data_hash':digest(rows),'data':rows})
    save_json(run/'compute_accounting.json',{'physical_requests_this_invocation':backend.calls,'physical_output_tokens_this_invocation':backend.physical_tokens,'physical_generation_seconds_this_invocation':backend.physical_seconds,'invocation_seconds':time.perf_counter()-started,'logical_evaluation_output_tokens':sum(r['generated_tokens'] for r in rows),'logical_evaluation_input_tokens':sum(r['input_tokens'] for r in rows),'cache_rule':setup['cache_rule']})
    summarize(run,rows,prepared,setup,identity)
    save_json(ROOT/'artifacts/expanded_test/LATEST_RUN.json',{'run_directory':str(run),'config_hash':identity})
    log('expanded_complete',run=str(run),episodes=len(rows))

def summarize(run,rows,prepared,setup,identity):
    output=run/'results';output.mkdir(exist_ok=True)
    write_csv(output/'per_instance.csv',[{k:v for k,v in r.items() if k!='actions'} for r in rows])
    tasks={(t['dataset'],t['task_id']):t for t in prepared['tasks']};errors=[];seen=set()
    for r in rows:
        key=(r['dataset'],r['task_id'],r['method'],r['seed'])
        if key in seen:errors.append('duplicate episode')
        seen.add(key);t=tasks[(r['dataset'],r['task_id'])]
        score=answer_metrics(r['final_answer'],t['answer'],t.get('aliases',[]))
        for metric in ['exact_match','f1','task_score']:
            if abs(r[metric]-score[metric])>1e-12:errors.append('metric mismatch')
        action_tokens=sum(a['token_cost'] for a in r['actions'])+r['routing_tokens']
        if action_tokens!=r['generated_tokens']:errors.append('output counter mismatch')
        if r['total_tokens']!=r['input_tokens']+r['generated_tokens']:errors.append('total counter mismatch')
        if not 0<=r['generated_tokens']<=setup['budget']['tokens'] or not 0<=r['retrieval_calls']<=setup['budget']['retrievals'] or not 0<=r['verification_calls']<=setup['budget']['verifications']:errors.append('budget violation')
        if not all(np.isfinite(v) for v in r.values() if isinstance(v,(float,int))):errors.append('nonfinite metric')
    assert len(rows)==480 and len(seen)==480
    assert code_hash()==setup['framework_source_hash'] and file_hash(__file__)==setup['extension_script_hash']
    validate_splits(sum(prepared['original_partitions'].values(),[])+prepared['tasks'])
    assert digest(prepared['documents'])==setup['corpus_hash']
    summary=[];task_means=[]
    metrics=['exact_match','f1','generated_tokens','input_tokens','total_tokens','retrieval_calls','verification_calls','latency','controller_latency','normalized_cost']
    for dataset in ['HotpotQA','2WikiMultiHopQA']:
        for method in METHODS:
            grouped=defaultdict(list)
            for r in rows:
                if (r['dataset'],r['method'])==(dataset,method):grouped[r['task_id']].append(r)
            assert len(grouped)==20 and all(len(v)==3 and set(r['seed'] for r in v)==set(setup['seeds']) for v in grouped.values())
            means=[{'dataset':dataset,'method':method,'task_id':key,**{m:float(np.mean([r[m] for r in group])) for m in metrics}} for key,group in grouped.items()]
            task_means+=means;row={'dataset':dataset,'method':method,'n_tasks':20,'n_seeds':3}
            for metric in metrics:
                result=paired_bootstrap([r[metric] for r in means],10000,SELECTION_SEED)
                row[metric]=result['estimate'];row[metric+'_ci_low']=result['ci_low'];row[metric+'_ci_high']=result['ci_high']
            summary.append(row)
    write_csv(output/'summary.csv',summary);write_csv(output/'task_means.csv',task_means)
    comparisons=[];lookup={(r['dataset'],r['task_id'],r['method']):r for r in task_means}
    for dataset in ['HotpotQA','2WikiMultiHopQA']:
        ids=sorted(t['task_id'] for t in prepared['tasks'] if t['dataset']==dataset)
        for other in METHODS[1:]:
            for metric in ['exact_match','generated_tokens','normalized_cost']:
                differences=[lookup[(dataset,t,'CRAVE')][metric]-lookup[(dataset,t,other)][metric] for t in ids]
                result=paired_bootstrap(differences,10000,SELECTION_SEED)
                comparisons.append({'dataset':dataset,'comparison':'CRAVE minus '+other,'metric':metric,**result,'raw_p':paired_permutation(differences,10000,SELECTION_SEED)})
    for row,p in zip(comparisons,holm([r['raw_p'] for r in comparisons])):row['holm_p']=p
    write_csv(output/'statistics.csv',comparisons)
    validation={'passed':not errors,'critical_errors':errors,'episodes':480,'new_tasks':40,'seeds':3,'methods':4,'budget_violations':sum(r['budget_violation'] for r in rows),'answer_scores_recomputed':True,'counter_accounting_checked':True,'original_framework_hash_match':True,'training_and_prior_evaluation_disjoint':True,'controller_retrained':False,'status':'POST_PILOT_EXPANDED_HELDOUT_NOT_FINAL_STUDY'}
    save_json(output/'validation.json',validation)
    if errors:raise ValueError(str(errors))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);args=p.parse_args()
    prepare() if args.stage=='prepare' else evaluate()
