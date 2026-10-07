import copy
import unittest
import numpy as np
from crave.agent.budget import Budget,Cost
from crave.agent.checkpoint import checkpoint,restore
from crave.agent.environment import Environment
from crave.baselines.controllers import MixedPolicy
from crave.counterfactual.branch import branches,rollout,interaction_targets
from crave.datasets.common import validate_splits
from crave.evaluation.task_metrics import answer_metrics,utility
from crave.models.backend import Generation
from crave.models.state_encoder import HashEncoder,StateFeatures
from crave.models.value_model import ValueModel

class FixtureBackend:
    def count(self,text): return len(text.split())
    def truncate(self,text,n): return ' '.join(text.split()[:n])
    def generate(self,system,user,max_tokens,seed,schema=None):
        # Deliberate fixed unit-test fixture; never used by the pilot runner.
        return Generation({'answer':'fixture','summary':'a public summary','confidence':.6,'support':.5},min(max_tokens,12),20,.001)

class FixtureRetriever:
    def search(self,query,k):
        return [{'doc_id':str(i),'title':'title','text':'some source text'} for i in range(k)],.001

def fixture():
    encoder=HashEncoder(); env=Environment(FixtureBackend(),FixtureRetriever(),encoder,max_steps=6)
    task={'task_id':'fixture-1','dataset':'fixture','split':'train','question':'a question','answer':'fixture'}
    state=env.initial(task,{'tokens':512,'retrievals':2,'verifications':1,'latency':None},17)
    return env,task,state

class CoreTests(unittest.TestCase):
    def test_atomic_budget(self):
        b=Budget(10,1,0)
        with self.assertRaises(ValueError): b.consume(Cost(tokens=11))
        self.assertEqual(b.tokens,10)
        b.consume(Cost(10,1)); self.assertEqual(b.as_dict(),{'tokens':0,'retrievals':0,'verifications':0,'latency':None})

    def test_negative_resources(self):
        with self.assertRaises(ValueError): Budget(-1,0,0)
        with self.assertRaises(ValueError): Cost(tokens=-1)

    def test_random_budget_safety(self):
        env,task,state=fixture()
        for seed in range(100):
            state=env.initial(task,{'tokens':512,'retrievals':2,'verifications':1,'latency':None},seed)
            rollout(env,state,MixedPolicy())
            self.assertGreaterEqual(min(state.budget.tokens,state.budget.retrievals,state.budget.verifications),0)

    def test_mask_reserves_upper_bound(self):
        env,task,state=fixture(); state.budget.tokens=255
        self.assertNotIn('THINK',env.feasible(state))
        with self.assertRaises(ValueError): env.action(state,'THINK')

    def test_verification_requires_candidate(self):
        env,task,state=fixture(); self.assertNotIn('VERIFY',env.feasible(state))
        env.action(state,'THINK'); self.assertIn('VERIFY',env.feasible(state))

    def test_state_restore_isolation(self):
        env,task,state=fixture(); env.action(state,'RETRIEVE'); saved=checkpoint(state)
        a,b=restore(saved),restore(saved); a.evidence[0]['text']='corrupted'
        self.assertNotEqual(a.evidence[0]['text'],b.evidence[0]['text'])
        self.assertEqual(checkpoint(b)['hash'],saved['hash'])

    def test_corrupt_checkpoint_is_fatal(self):
        env,task,state=fixture(); saved=checkpoint(state); saved['payload']['question']='changed'
        with self.assertRaises(ValueError): restore(saved)

    def test_matched_branches_and_seeds(self):
        env,task,state=fixture(); env.action(state,'THINK'); saved=checkpoint(state)
        rows=branches(env,saved,task,MixedPolicy(),k=3)
        self.assertEqual({r['state_hash'] for r in rows},{saved['hash']})
        for row in rows: self.assertEqual(row['state_before'],saved['payload'])
        for k in range(3): self.assertEqual(len({r['branch_seed'] for r in rows if r['replicate']==k}),1)
        self.assertEqual(checkpoint(state)['hash'],saved['hash'])

    def test_answer_leakage_whitelist(self):
        env,task,state=fixture(); task.update(answer='SECRET',aliases=['ALIAS'],supporting_facts=['SECRET'],oracle_action='VERIFY')
        a=env.initial(task,state.initial_budget,17)
        self.assertEqual(a.as_dict(),state.as_dict())
        self.assertNotIn('SECRET',env.prompt_state(a))
        self.assertNotIn('oracle_action',a.as_dict())

    def test_split_leakage(self):
        _,task,_=fixture(); other={**task,'split':'test'}
        with self.assertRaises(ValueError): validate_splits([task,other])
        other={**other,'task_id':'different'}
        with self.assertRaises(ValueError): validate_splits([task,other])
        other={**other,'question':'different question','source_question_ids':[('source','1')]}
        with self.assertRaises(ValueError): validate_splits([{**task,'source_question_ids':[('source','1')]},other])

    def test_metrics(self):
        self.assertEqual(answer_metrics('The Eiffel Tower!','Eiffel Tower')['exact_match'],1)
        self.assertEqual(answer_metrics('yes','no')['f1'],0)
        self.assertAlmostEqual(answer_metrics('a blue river','blue')['f1'],2/3)
        self.assertEqual(answer_metrics('alias','gold',['alias'])['exact_match'],1)
        self.assertAlmostEqual(utility(1,[2048,2,1,99],.1),.685)

    def test_training_rejects_test_oracles(self):
        from crave.training.train_value import arrays
        env,task,state=fixture(); task['split']='test'; state.split='test'
        rows=branches(env,checkpoint(state),task,MixedPolicy(),k=1)
        with self.assertRaises(ValueError): arrays(rows,StateFeatures(HashEncoder()),[.1],[4096,4,2,60])

    def test_mlp_analytic_gradients(self):
        rng=np.random.default_rng(8); model=ValueModel(5,(4,3),3)
        x=rng.normal(size=(3,5)).astype(np.float32); target=rng.normal(size=(3,4,6)).astype(np.float32)
        target[:,:,1]=rng.random((3,4)); target[:,:,2:]=np.abs(target[:,:,2:])
        mask=np.ones((3,4),np.float32); interactions=rng.normal(size=(3,16)).astype(np.float32); imask=np.ones((3,16),np.float32)
        weights={'cost':.25,'cal':.25,'rank':.25,'interaction':.25}
        output,cache=model.forward(x); loss,g=model.loss(output,target,mask,interactions,imask,weights)
        gradients=model.gradients(cache,g)
        for key,index in [('W0',(1,1)),('W1',(1,1)),('W2',(1,5)),('b2',(0,)),('b2',(25,))]:
            old=model.params[key][index]; delta=.002
            model.params[key][index]=old+delta; plus=model.loss(model.predict(x),target,mask,interactions,imask,weights)[0]
            model.params[key][index]=old-delta; minus=model.loss(model.predict(x),target,mask,interactions,imask,weights)[0]
            model.params[key][index]=old
            self.assertAlmostEqual(float(gradients[key][index]),(plus-minus)/(2*delta),delta=.005)

    def test_mlp_learns_observed_targets(self):
        model=ValueModel(3,(8,4),0); x=np.eye(3,dtype=np.float32)
        target=np.zeros((3,4,6),np.float32); target[:,:,0]=.5; target[:,:,1]=1
        mask=np.ones((3,4),np.float32); it=np.zeros((3,16),np.float32)
        losses=model.fit(x,target,mask,it,it,epochs=100,lr=.01)
        self.assertLess(losses[-1],losses[0]*.5)

    def test_statistics_and_shared_pareto_interval(self):
        from crave.evaluation.statistics import paired_bootstrap,paired_permutation,holm,reliability,pareto_auc
        result=paired_bootstrap([2,2,2],1000)
        self.assertEqual((result['estimate'],result['ci_low'],result['ci_high']),(2,2,2))
        self.assertEqual(paired_permutation([0,0,0]),1)
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(reliability([0,1],[0,1])['brier'],0)
        self.assertEqual(reliability([0,1],[0,1])['ece'],0)
        self.assertAlmostEqual(pareto_auc([(1,.5),(2,.8)],(1,2)),.65)
        self.assertIsNone(pareto_auc([(1,.5),(2,.8)],(0,2)))

    def test_interactions_use_common_replicates(self):
        env,task,state=fixture(); env.action(state,'THINK')
        rows=branches(env,checkpoint(state),task,MixedPolicy(),k=2,forced_sequences=[('THINK',),('VERIFY',),('THINK','VERIFY'),()])
        targets=interaction_targets(rows,[0,.1],[4096,4,2,60])
        self.assertIn((('THINK','VERIFY'),0),targets)
        self.assertAlmostEqual(targets[(('THINK','VERIFY'),0)],0)

    def test_invalid_json_charged_without_candidate_corruption(self):
        env,task,state=fixture()
        env.backend.generate=lambda *a,**k: Generation({},100,20,.001,False,False)
        env.action(state,'THINK')
        self.assertEqual(state.budget.tokens,412)
        self.assertEqual(state.candidate_answer,'')

if __name__=='__main__': unittest.main()
