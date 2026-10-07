"""Resumable real-data pilot. Refuses synthetic inference and final-paper claims."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import yaml
from crave.common import ROOT, digest, file_hash, read_json, save_json, seed_all, log
from crave.agent.checkpoint import checkpoint, restore
from crave.agent.state import ACTIONS
from crave.agent.environment import Environment
from crave.baselines.controllers import MixedPolicy, FixedPolicy, ConfidencePolicy, LLMRouter, ValuePolicy, AggregatePolicy
from crave.counterfactual.branch import branches, rollout
from crave.datasets.common import hotpot_rows, musique_rows, wiki2hop_rows, author_raw_rows, validate_splits, build_corpus, manifest
from crave.evaluation.task_metrics import answer_metrics, future_cost, utility
from crave.models.backend import LlamaBackend
from crave.models.state_encoder import FrozenEncoder, StateFeatures
from crave.models.value_model import ValueModel, sigmoid
from crave.training.train_value import arrays

def code_hash():
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in sorted(ROOT.rglob('*.py')) if '.venv' not in p.parts and 'runtime' not in p.parts})

def load_config():
    return {p.stem:yaml.safe_load(p.read_text()) for p in sorted((ROOT/'configs').glob('*.yaml'))}

def task_data(args):
    train=hotpot_rows(ROOT/'data/hotpot_train.json','train',count=args.train_tasks)
    dev=hotpot_rows(ROOT/'data/hotpot_validation.json','dev',count=6)
    test=hotpot_rows(ROOT/'data/hotpot_validation.json','pilot_eval',start=6,count=args.eval_tasks)
    if (ROOT/'data/musique_ans_v1.0_train.jsonl').exists():
        train+=musique_rows(ROOT/'data/musique_ans_v1.0_train.jsonl','train')[:args.train_tasks]
    elif (ROOT/'data/musique_raw_train.json').exists():
        train+=author_raw_rows(ROOT/'data/musique_raw_train.json','MuSiQue','train',args.train_tasks)
        if (ROOT/'data/musique_raw_dev.json').exists():dev+=author_raw_rows(ROOT/'data/musique_raw_dev.json','MuSiQue','dev',6)
    if (ROOT/'data/2wiki_dev.json').exists(): test+=wiki2hop_rows(ROOT/'data/2wiki_dev.json','pilot_eval')[:args.eval_tasks]
    elif (ROOT/'data/2wiki_raw_dev.json').exists():test+=author_raw_rows(ROOT/'data/2wiki_raw_dev.json','2WikiMultiHopQA','pilot_eval',args.eval_tasks)
    validate_splits(train+dev+test)
    return train,dev,test

class Runner:
    def __init__(self,args):
        self.args=args; self.config=load_config(); self.model_manifest=read_json(ROOT/'artifacts/model_manifest.json')
        model_path=ROOT/'models_data'/self.model_manifest['model']['file']
        expected_sha='7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
        if model_path.stat().st_size!=2497280256 or file_hash(model_path)!=expected_sha:
            raise ValueError('Unverified or incomplete Qwen3-4B checkpoint')
        if self.model_manifest['model']['sha256']!=expected_sha:
            raise ValueError('Model manifest does not match official SHA256')
        self.train,self.dev,self.test=task_data(args)
        self.documents=build_corpus(self.train+self.dev+self.test)
        self.config['pilot_overrides']=vars(args)
        self.config['model_manifest']=self.model_manifest
        self.config['code_hash']=code_hash()
        self.config['task_hashes']={k:digest(v) for k,v in [('train',self.train),('dev',self.dev),('pilot_eval',self.test)]}
        self.config['corpus_hash']=digest(self.documents)
        self.hash=digest(self.config); self.run=ROOT/'runs'/self.hash[:16]; self.run.mkdir(parents=True,exist_ok=True)
        save_json(self.run/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in
                        ['numpy','onnxruntime','tokenizers','scipy','matplotlib','PyYAML','pypdf','python-docx']},
            'hardware_setup_record':read_json(ROOT/'artifacts/environment.json')})
        save_json(self.run/'FROZEN_PILOT_CONFIG.json',{'config':self.config,'sha256':self.hash,'status':'PILOT_ONLY_NOT_FINAL_FREEZE'})
        sources={('HotpotQA',split):read_json(ROOT/'data'/filename) for split,filename in
                 [('train','hotpot_train.json'),('dev','hotpot_validation.json'),('pilot_eval','hotpot_validation.json')]}
        for dataset,split,filename in [('MuSiQue','train','musique_raw_train.json'),('MuSiQue','dev','musique_raw_dev.json'),
                                       ('2WikiMultiHopQA','pilot_eval','2wiki_raw_dev.json')]:
            if (ROOT/'data'/filename).exists():sources[(dataset,split)]=read_json(ROOT/'data'/filename)
        manifest(self.train+self.dev+self.test,sources)
        save_json(self.run/'corpus_manifest.json',{'sha256':digest(self.documents),'n_documents':len(self.documents),'scope':'pooled benchmark contexts, not full Wikipedia','uses_answer_labels':False})
        self.encoder=FrozenEncoder(ROOT/'models_data/encoder')
        from crave.retrieval.hybrid import HybridRetriever
        self.retriever=HybridRetriever(self.documents,self.encoder)
        backend_id=digest({'model':self.model_manifest['model']['sha256'],'runtime':self.model_manifest['runtime']['sha256']})
        self.backend=LlamaBackend(self.config['model']['base_url'],backend_id)
        self.env=Environment(self.backend,self.retriever,self.encoder,max_steps=self.config['experiments']['max_steps'])
        self.policy=MixedPolicy()
        self.lambdas=self.config['utility']['lambdas']; self.references=self.config['utility']['references']
        self.budgets=self.config['budgets']['pilot_budgets'][:args.budgets]
        self.seeds=self.config['training']['seeds']
        self.features=StateFeatures(self.encoder)

    def stage(self,name,callback):
        path=self.run/(name+'.json')
        if path.exists():
            payload=read_json(path)
            if payload['config_hash']!=self.hash: raise ValueError('Stage config mismatch')
            if digest(payload['data'])!=payload['data_hash']: raise ValueError('Stage artifact checksum mismatch')
            log('stage_resume',stage=name)
            return payload['data']
        log('stage_start',stage=name)
        start=time.perf_counter(); data=callback()
        save_json(path,{'config_hash':self.hash,'data_hash':digest(data),'elapsed_seconds':time.perf_counter()-start,'data':data})
        log('stage_complete',stage=name,elapsed=time.perf_counter()-start)
        return data

    def collect(self,round_number,learned=None):
        records=[]; observations=[]; pairs_done=0
        collector=self.policy if learned is None else AggregatePolicy(learned)
        for ti,task in enumerate(self.train):
            budget=self.budgets[ti%len(self.budgets)]
            visitation=self.run/f'round{round_number}_visitation_{digest([task["dataset"],task["task_id"]])[:16]}.json'
            if visitation.exists():
                saved_states=read_json(visitation)
                for saved in saved_states:restore(saved)
            else:
                state=self.env.initial(task,budget,self.seeds[round_number-1])
                saved_states=[checkpoint(state)]
                if learned is None:
                    self.env.action(state,'THINK')
                    saved_states.append(checkpoint(state))
                else:
                    while not state.terminal and len(saved_states)<self.args.states:
                        self.env.action(state,collector(state,self.env.feasible(state)))
                        if not state.terminal: saved_states.append(checkpoint(state))
                save_json(visitation,saved_states)
            for saved in saved_states[:self.args.states]:
                state_id=saved['hash'][:16]; chunk=self.run/f'round{round_number}_{state_id}.json'
                if chunk.exists():
                    cached=read_json(chunk); records.extend(cached['records']); observations.extend(cached['observations']); pairs_done+=cached['interaction_state']; continue
                log('branch_checkpoint',round=round_number,task=task['task_id'],state=state_id)
                sequences=[(a,) for a in self.env.feasible(restore(saved))]
                interaction_state=int(pairs_done<self.args.interaction_states)
                if interaction_state:
                    sequences+=[(),('RETRIEVE','THINK'),('THINK','VERIFY')]; pairs_done+=1
                branch_rows=branches(self.env,saved,task,self.policy,k=self.config['training']['k'],seed=self.seeds[round_number-1],forced_sequences=sequences)
                # -CF uses only actions naturally selected in an on-policy trajectory, no alternative-action labels.
                observational_state=restore(saved); observational_state.seed+=9000
                trace=rollout(self.env,observational_state,self.policy,keep_states=True)
                observed=[]
                for step in trace:
                    before=restore(step['checkpoint']); result=step['result']
                    immediate_state=restore(step['checkpoint'])
                    immediate_reward=answer_metrics(result['candidate_answer'],task['answer'],task.get('aliases',[]))['task_score']
                    observed.append({'task_id':task['task_id'],'dataset':task['dataset'],'split':'train','state_hash':step['checkpoint']['hash'],
                       'state_before':before.as_dict(),'forced_sequence':[step['action']],'reward':answer_metrics(observational_state.candidate_answer,task['answer'],task.get('aliases',[]))['task_score'],
                       'future_cost':future_cost(before,observational_state),'immediate':{'reward':immediate_reward,'cost':[result['token_cost'],result['retrieval_cost'],result['verification_cost'],result['latency']]},
                       'replicate':0,'continuation_policy':self.policy.version,'target_source':'observational_only'})
                save_json(chunk,{'records':branch_rows,'observations':observed,'interaction_state':interaction_state})
                records.extend(branch_rows); observations.extend(observed)
        return {'records':records,'observations':observations}

    def train_model(self,records,seed,variant='CRAVE'):
        features=StateFeatures(self.encoder,variant if variant.startswith('-') else None)
        training_variant='-RTG' if variant=='myopic' else variant
        data=arrays(records,features,self.lambdas,self.references,training_variant)
        model=ValueModel(features.dim,tuple(self.config['controller']['hidden']),seed)
        weights=dict(self.config['training']['weights'])
        if variant in ['-Interaction','-CF','-RTG','myopic']: weights['interaction']=0
        if variant=='-Pairwise': weights['rank']=0
        if variant in ['-Calibration','-Confidence']: weights['cal']=0
        losses=model.fit(*data,weights=weights,epochs=self.config['training']['epochs'],lr=self.config['training']['learning_rate'],seed=seed,batch_size=self.config['training']['batch_size'])
        path=self.run/f'{variant}_{seed}.npz'; model.save(path)
        save_json(self.run/f'{variant}_{seed}_training.json',{'losses':losses,'n_training_states_lambda_rows':len(data[0]),'variant':variant,'seed':seed,'sha256':file_hash(path)})
        log('model_trained',variant=variant,seed=seed,initial_loss=losses[0],final_loss=losses[-1])
        return model,features

    def evaluate(self,rounds):
        combined=sum([r['records'] for r in rounds],[]); observed=sum([r['observations'] for r in rounds],[])
        methods=['CRAVE','-CF','-RTG','-Interaction','-Budget','myopic']
        trained={}
        for seed in self.seeds:
            for method in methods:
                path=self.run/f'{method}_{seed}.npz'
                if path.exists(): trained[(method,seed)]=(ValueModel.load(path),StateFeatures(self.encoder,method if method.startswith('-') else None))
                else: trained[(method,seed)]=self.train_model(observed if method=='-CF' else combined,seed,method)
        rows=[]
        all_methods=['fixed','confidence','llm_router',*methods]
        eval_lambdas=self.lambdas if self.args.full_lambda_sweep else [0.1]
        for task in self.test:
            for seed in self.seeds:
                for bi,budget in enumerate(self.budgets):
                    for weight in eval_lambdas:
                        for method in all_methods:
                            key=digest({'task':task['task_id'],'dataset':task['dataset'],'seed':seed,'budget':budget,'lambda':weight,'method':method})
                            path=self.run/'episodes'/(key+'.json')
                            if path.exists(): rows.append(read_json(path)); continue
                            if method=='fixed': policy=FixedPolicy()
                            elif method=='confidence': policy=ConfidencePolicy(self.config['experiments']['dev_threshold'])
                            elif method=='llm_router': policy=LLMRouter(self.env)
                            else:
                                model,features=trained[(method,seed)]; policy=ValuePolicy(model,features,weight)
                            state=self.env.initial(task,budget,seed); start=time.perf_counter()
                            rollout(self.env,state,policy)
                            metrics=answer_metrics(state.candidate_answer,task['answer'],task.get('aliases',[]))
                            consumed=future_cost(self.env.initial(task,budget,seed),state)
                            row={'run_id':self.run.name,'config_hash':self.hash,'git_hash':None,'seed':seed,'dataset':task['dataset'],'split':'pilot_eval',
                                 'task_id':task['task_id'],'method':method,'model':self.model_manifest['model']['repository'],'budget_tokens':budget['tokens'],
                                 'budget_retrieval':budget['retrievals'],'budget_verification':budget['verifications'],'utility_lambda':weight,
                                 'final_answer':state.candidate_answer,**metrics,'reasoning_tokens':sum(a['token_cost'] for a in state.action_log if a['action']=='THINK'),'generated_tokens':consumed[0],
                                 'verification_tokens':sum(a['token_cost'] for a in state.action_log if a['action']=='VERIFY'),'routing_tokens':state.routing_tokens,
                                 'input_tokens':state.input_tokens,'total_tokens':state.input_tokens+consumed[0],
                                 'retrieval_calls':consumed[1],'verification_calls':consumed[2],'latency':state.latency,
                                 'context_tokens_added':state.context_tokens_added,'retrieved_documents':state.retrieved_documents,
                                 'controller_latency':state.controller_latency,'physical_episode_seconds':time.perf_counter()-start,
                                 'normalized_cost':sum(consumed[j]/self.references[j] for j in range(4)),
                                 'budget_violation':int(min(state.budget.tokens,state.budget.retrievals,state.budget.verifications)<0),
                                 'actions':state.action_log,'confidence':state.confidence,'status':'PILOT'}
                            save_json(path,row); rows.append(row)
                            log('episode',method=method,task=task['task_id'],seed=seed,budget=bi,EM=metrics['exact_match'],tokens=consumed[0])
        return rows

    def oracle(self):
        rows=[]
        for task in self.test[:min(3,len(self.test))]:
            state=self.env.initial(task,self.budgets[-1],self.seeds[0]); self.env.action(state,'THINK')
            saved=checkpoint(state)
            rows+=branches(self.env,saved,task,self.policy,k=self.config['training']['k'],seed=31415)
        return rows

    def execute(self):
        started=time.perf_counter()
        round1=self.stage('round1',lambda:self.collect(1))
        first_path=self.run/f'round1_{self.seeds[0]}.npz'
        if first_path.exists(): first,features=ValueModel.load(first_path),self.features
        else: first,features=self.train_model(round1['records'],self.seeds[0],'round1')
        round2=self.stage('round2',lambda:self.collect(2,ValuePolicy(first,features,0.1)))
        rows=self.stage('evaluation',lambda:self.evaluate([round1,round2]))
        oracle=self.stage('oracle',self.oracle)
        from crave.reporting import report
        report(self,rows,oracle)
        save_json(self.run/'compute_accounting.json',{'physical_model_requests_this_invocation':self.backend.calls,'physical_generated_tokens_this_invocation':self.backend.physical_tokens,
                   'physical_generation_seconds_this_invocation':self.backend.physical_seconds,'total_invocation_seconds':time.perf_counter()-started,
                   'logical_inference_tokens':sum(r['generated_tokens'] for r in rows),'logical_input_tokens':sum(r['input_tokens'] for r in rows),
                   'logical_offline_branch_generated_tokens':sum(r['future_cost'][0] for rd in [round1,round2] for r in rd['records']),
                   'cache_latency_rule':'Cached requests retain original observed generation latency; cold end-to-end latency benchmark NOT RUN.',
                   'monetary_cost':None})
        save_json(ROOT/'artifacts/LATEST_RUN.json',{'run_directory':str(self.run),'config_hash':self.hash,'status':'PILOT_COMPLETE_FULL_DESIGN_INCOMPLETE'})
        log('pilot_complete',run=str(self.run))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-tasks',type=int,default=4); p.add_argument('--eval-tasks',type=int,default=25)
    p.add_argument('--states',type=int,default=2); p.add_argument('--interaction-states',type=int,default=2)
    p.add_argument('--budgets',type=int,choices=[1,2],default=2); p.add_argument('--full-lambda-sweep',action='store_true')
    args=p.parse_args()
    if min(args.train_tasks,args.eval_tasks,args.states)<1: p.error('Task and state counts must be positive')
    Runner(args).execute()

if __name__=='__main__': main()
