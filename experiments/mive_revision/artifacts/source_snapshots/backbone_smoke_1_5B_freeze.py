"""Frozen development execution checks, not a held-out efficacy comparison."""
import argparse,datetime,json,pathlib,platform,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crave.common import read_json,save_json,digest,file_hash
from crave.models.backend import LlamaBackend,Generation
from crave.models.state_encoder import FrozenEncoder
from crave.agent.environment import Environment
from crave.agent.state import State
from crave.baselines.controllers import FixedPolicy,ConfidencePolicy
from crave.counterfactual.branch import rollout
from crave.datasets.common import build_corpus
from crave.retrieval.hybrid import HybridRetriever
from crave.evaluation.task_metrics import answer_metrics
BASE=ROOT/'crave';ART=ROOT/'experiments/mive_revision/artifacts'

class NativeChatMLBackend(LlamaBackend):
    def generate(self,system,user,max_tokens,seed,schema=None):
        prompt='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'
        payload={'prompt':prompt,'n_predict':max_tokens,'temperature':.7,'top_p':.8,'top_k':20,
                 'seed':int(seed)%(2**31-1),'cache_prompt':False,'stop':['<|im_end|>','<|endoftext|>']}
        if schema:payload['json_schema']=schema
        started=time.perf_counter();response=self.post('/completion',payload)
        elapsed=time.perf_counter()-started
        try:data=json.loads(response['content']);valid=isinstance(data,dict)
        except json.JSONDecodeError:data={};valid=False
        output=int(response['tokens_predicted']);inputs=int(response['tokens_evaluated'])
        assert output<=max_tokens
        self.calls+=1;self.physical_tokens+=output;self.physical_seconds+=elapsed
        return Generation(data,output,inputs,elapsed,False,valid)

def main(model,mode):
    out=ART/('smoke_'+model);out.mkdir(exist_ok=True)
    manifest=read_json(ART/(model+'_download_manifest.json'))
    assert manifest.get('status')=='verified','No inference before complete checksum verification'
    for item in manifest['files']:
        assert file_hash(BASE/'models_data'/model/item['rfilename'])==item['lfs']['sha256']
    full=read_json(ART/'dev4b/tasks.json')
    tasks=[next(r for r in full if r['dataset']==dataset) for dataset in ['HotpotQA','2WikiMultiHopQA','MuSiQue']]
    path=out/'freeze.json'
    if not path.exists():
        config={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'model':model,'model_manifest_sha256':file_hash(ART/(model+'_download_manifest.json')),
                'tasks_sha256':digest(tasks),'corpus_sha256':digest(build_corpus(full)),
                'task_ids':[r['task_id'] for r in tasks],'question_selection':'first previously frozen development question per dataset',
                'seed':17,'budget':{'tokens':512,'retrievals':1,'verifications':1},
                'policies':['fixed','confidence'],'confidence_threshold':.8,'think_reserve':256,'verify_reserve':192,
                'context_tokens':1400,'max_steps':12,'sampling':{'temperature':.7,'top_p':.8,'top_k':20},
                'prompt':'native Qwen2.5 ChatML without Qwen3 thinking directives','cache':False,
                'hardware':'CPU two threads, GPU layers zero' if model.endswith('1.5B') else 'RTX 3060 Laptop 6144 MiB, Vulkan1, four CPU threads',
                'server_slots':1,'server_context':4096,'platform':platform.platform(),
                'scope':'development execution and accounting check; one question per dataset, one seed; no held-out efficacy or model-size inference',
                'source_sha256':file_hash(__file__),'heldout_inference':False}
        save_json(path,{'config':config,'sha256':digest(config)})
    frozen=read_json(path);assert digest(tasks)==frozen['config']['tasks_sha256']
    if mode=='prepare':print(json.dumps(frozen));return
    port=8097 if model.endswith('1.5B') else 8098
    backend=NativeChatMLBackend(base_url='http://127.0.0.1:'+str(port),model_hash=manifest['revision'],cache=False)
    encoder=FrozenEncoder(BASE/'models_data/encoder')
    env=Environment(backend,HybridRetriever(build_corpus(full),encoder),encoder,max_steps=12)
    journal=out/'episodes.jsonl';rows=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    for task in tasks:
        for name,policy in [('fixed',FixedPolicy()),('confidence',ConfidencePolicy(.8))]:
            if any(r['task_id']==task['task_id'] and r['method']==name for r in rows):continue
            state=env.initial(task,frozen['config']['budget'],17);started=time.perf_counter()
            rollout(env,state,policy)
            row={'task_id':task['task_id'],'dataset':task['dataset'],'method':name,'seed':17,
                 **answer_metrics(state.candidate_answer,task['answer'],task.get('aliases',[])),
                 'elapsed_seconds':time.perf_counter()-started,'final_state':state.as_dict()}
            with open(journal,'a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
            rows.append(row);save_json(out/'progress.json',{'completed':len(rows),'planned':6,'physical_calls':backend.calls})
    results=[]
    for row in rows:
        state=State.from_dict(row['final_state']);task=next(r for r in tasks if r['task_id']==row['task_id'])
        assert state.terminal and min(state.budget.tokens,state.budget.retrievals,state.budget.verifications)>=0
        scores=answer_metrics(state.candidate_answer,task['answer'],task.get('aliases',[]))
        assert all(row[k]==v for k,v in scores.items())
        actions=state.action_log
        results.append({k:v for k,v in row.items() if k!='final_state'}|{
            'output_tokens':sum(a['token_cost'] for a in actions),
            'retrievals':sum(a['retrieval_cost'] for a in actions),
            'verifications':sum(a['verification_cost'] for a in actions),
            'action_counts':{a:sum(x['action']==a for x in actions) for a in ['THINK','RETRIEVE','VERIFY','STOP']}})
    assert len(rows)==6
    save_json(out/'verified_results.json',{'passed':True,'episodes':results,'budget_violations':0,
        'physical_calls_this_invocation':backend.calls,'physical_output_tokens_this_invocation':backend.physical_tokens,
        'request_seconds_this_invocation':backend.physical_seconds,'scope':frozen['config']['scope']})
    print(json.dumps({'model':model,'episodes':len(rows),'passed':True}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['Qwen2.5-1.5B','Qwen2.5-7B']);parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    main(args.model,args.mode)
