import csv
from collections import defaultdict
import numpy as np
from crave.common import ROOT,save_json,digest,file_hash
from crave.agent.state import State,ACTIONS
from crave.models.value_model import ValueModel,sigmoid
from crave.models.state_encoder import StateFeatures
from crave.evaluation.statistics import paired_bootstrap,paired_permutation,holm,reliability,pareto_auc
from crave.evaluation.task_metrics import utility

def write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def report(runner,rows,oracle):
    output=runner.run/'results'; output.mkdir(exist_ok=True)
    flat=[{k:v for k,v in r.items() if k!='actions'} for r in rows]
    write_csv(output/'per_instance.csv',flat)
    critical=[]; seen=set()
    for r in rows:
        key=(r['dataset'],r['task_id'],r['method'],r['seed'],r['budget_tokens'],r['budget_retrieval'],r['budget_verification'],r['utility_lambda'])
        if key in seen: critical.append('duplicate episode')
        seen.add(key)
        if r['budget_violation']: critical.append('budget violation')
        if not all(np.isfinite(v) for v in r.values() if isinstance(v,(float,int))): critical.append('NaN/Inf')
    for r in oracle:
        if digest(r['state_before'])!=r['state_hash']: critical.append('oracle branch mismatch')
        if r['split']=='train': critical.append('oracle split contamination')
    grouped=defaultdict(list)
    for r in rows: grouped[(r['dataset'],r['method'],r['budget_tokens'],r['budget_retrieval'],r['utility_lambda'])].append(r)
    metrics=['task_score','exact_match','f1','generated_tokens','input_tokens','total_tokens','retrieval_calls','verification_calls','latency','controller_latency','normalized_cost','budget_violation']
    summary=[];provenance={}
    for key,selected in sorted(grouped.items()):
        row=dict(zip(['dataset','method','budget_tokens','budget_retrieval','utility_lambda'],key))
        by_task=defaultdict(list)
        for r in selected: by_task[r['task_id']].append(r)
        row['n_tasks']=len(by_task);row['n_seeds']=len(set(r['seed'] for r in selected));row['status']='PILOT'
        for metric in metrics:
            values=[np.mean([r[metric] for r in task_rows]) for task_rows in by_task.values()]
            interval=paired_bootstrap(values,runner.config['experiments']['bootstrap_resamples'])
            row[metric]=interval['estimate'];row[metric+'_ci_low']=interval['ci_low'];row[metric+'_ci_high']=interval['ci_high']
            provenance['|'.join(map(str,key))+'|'+metric]={'source':str(output/'per_instance.csv'),'metric':metric,'aggregation':'mean seeds within task, mean tasks; task cluster bootstrap','config_hash':runner.hash,'seeds':runner.seeds}
        summary.append(row)
    write_csv(output/'summary.csv',summary)
    save_json(output/'provenance.json',provenance)
    comparisons=[]
    for dataset,bt,br,weight in sorted(set((r['dataset'],r['budget_tokens'],r['budget_retrieval'],r['utility_lambda']) for r in rows)):
        cell=[r for r in rows if (r['dataset'],r['budget_tokens'],r['budget_retrieval'],r['utility_lambda'])==(dataset,bt,br,weight)]
        for other in ['fixed','confidence','llm_router','-CF','-RTG','-Interaction','-Budget']:
            for metric in ['exact_match','normalized_cost']:
                values=defaultdict(lambda:defaultdict(list))
                for r in cell:
                    if r['method'] in ['CRAVE',other]: values[r['task_id']][r['method']].append(r[metric])
                if any(set(v)!=set(['CRAVE',other]) for v in values.values()): critical.append('unpaired methods');continue
                delta=np.array([np.mean(v['CRAVE'])-np.mean(v[other]) for v in values.values()])
                comparison={'dataset':dataset,'budget_tokens':bt,'budget_retrieval':br,'utility_lambda':weight,'comparison':'CRAVE minus '+other,'metric':metric,
                            **paired_bootstrap(delta,runner.config['experiments']['bootstrap_resamples']),'raw_p':paired_permutation(delta),'status':'PILOT_EXPLORATORY_NO_HYPOTHESIS_VERDICT'}
                comparisons.append(comparison)
    corrected=holm([r['raw_p'] for r in comparisons])
    for r,p in zip(comparisons,corrected):r['holm_p']=p
    write_csv(output/'statistics.csv',comparisons)
    diagnostic=[]
    states=defaultdict(list)
    for r in oracle:states[r['state_hash']].append(r)
    for state_hash,branches in states.items():
        state=State.from_dict(branches[0]['state_before'])
        for weight in runner.lambdas:
            empirical={a:np.mean([utility(r['reward'],r['future_cost'],weight,runner.references) for r in branches if r['forced_action']==a]) for a in set(r['forced_action'] for r in branches)}
            for seed in runner.seeds:
                for method in ['CRAVE','-CF','-RTG','-Interaction','-Budget']:
                    model=ValueModel.load(runner.run/f'{method}_{seed}.npz'); features=StateFeatures(runner.encoder,method)
                    heads=model.predict(features(state,weight)[None])[0,:24].reshape(4,6)
                    chosen=max(empirical,key=lambda a:heads[ACTIONS.index(a),0]);oracle_action=max(empirical,key=empirical.get)
                    for action,value in empirical.items():
                        action_rows=[r for r in branches if r['forced_action']==action]
                        samples=[r['reward'] for r in action_rows]
                        utilities=[utility(r['reward'],r['future_cost'],weight,runner.references) for r in action_rows]
                        diagnostic.append({'task_id':state.task_id,'state_hash':state_hash,'method':method,'seed':seed,'lambda':weight,'action':action,
                          'predicted_q':float(heads[ACTIONS.index(action),0]),'empirical_q':float(value),'predicted_success':float(sigmoid(heads[ACTIONS.index(action),1])),
                          'empirical_success':float(np.mean(samples)),'chosen_action':chosen,'oracle_action':oracle_action,
                          'empirical_q_standard_error':float(np.std(utilities,ddof=1)/np.sqrt(len(utilities))) if len(utilities)>1 else None,
                          'regret':float(empirical[oracle_action]-empirical[chosen]),'top1':int(chosen==oracle_action),'k':len(samples)})
    write_csv(output/'counterfactual_diagnostics.csv',diagnostic)
    diagnostic_summary=[]; calibration_bins=[]
    for method,seed,weight in sorted(set((r['method'],r['seed'],r['lambda']) for r in diagnostic)):
        cell=[r for r in diagnostic if (r['method'],r['seed'],r['lambda'])==(method,seed,weight)]
        calibration=reliability([r['predicted_success'] for r in cell],[r['empirical_success'] for r in cell])
        unique={r['state_hash']:r for r in cell}
        base={'method':method,'seed':seed,'lambda':weight}
        diagnostic_summary.append({**base,'n_states':len(unique),'n_action_estimates':len(cell),
            'mean_regret':float(np.mean([r['regret'] for r in unique.values()])),
            'top1_agreement':float(np.mean([r['top1'] for r in unique.values()])),
            'q_rmse':float(np.sqrt(np.mean([(r['predicted_q']-r['empirical_q'])**2 for r in cell]))),
            'brier_to_finite_k_success_mean':calibration['brier'],'ece_to_finite_k_success_mean':calibration['ece'],
            'status':'PILOT_FINITE_K_NO_UNBIASED_ORACLE_CLAIM'})
        calibration_bins.extend({**base,**entry} for entry in calibration['bins'])
    write_csv(output/'value_diagnostic_summary.csv',diagnostic_summary)
    write_csv(output/'success_reliability_bins.csv',calibration_bins)
    save_json(output/'validation.json',{'critical_errors':critical,'passed':not critical,'status':'PILOT_ONLY','raw_rows':len(rows),'n_oracle_states':len(states)})
    if critical:raise ValueError('Critical result validation failed: '+str(critical))
    text=['# CRAVE pilot results','', 'Status: PILOT ONLY. Full preregistered design and submission-ready manuscript: INCOMPLETE.', '',
          f'Configuration: {runner.hash}. Training tasks: {len(runner.train)}. Development tasks reserved: {len(runner.dev)}. Pilot evaluation tasks: {len(runner.test)}. Seeds: {runner.seeds}.', '',
          '## Scope and limits','', 'Quantized Qwen3-4B; frozen MiniLM; pooled benchmark-context corpus; fixed operational defaults; raw verbalized confidence. MuSiQue and 2Wiki use bounded author-linked raw exports, not the complete official release. Development tuning, K stability, full Wikipedia retrieval, full training, full lambda/budget sweeps, official literature baselines, full OOD/cross-model studies and cold latency remain NOT RUN unless separately executed. -CF uses observational trajectory targets with different action coverage; pilot does not isolate a definitive H2 effect.', '',
          '## Observed results','', '|Dataset|Method|Budget tokens|Lambda|N tasks|EM|F1|Generated tokens|Retrieval calls|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"|{r['dataset']}|{r['method']}|{r['budget_tokens']}|{r['utility_lambda']}|{r['n_tasks']}|{r['exact_match']:.4f}|{r['f1']:.4f}|{r['generated_tokens']:.1f}|{r['retrieval_calls']:.2f}|")
    text+=['','All empirical entries above are machine-generated from per_instance.csv. Confidence intervals and paired exploratory tests are in summary.csv and statistics.csv. No H1-H6 support claim is made from this pilot.','',
           '## Hypothesis status','','H1-H6: INCONCLUSIVE / FINAL DESIGN NOT RUN. No final manuscript was generated. Original pre-experimental manuscript is preserved.','',
           '## Validation','',f'Critical checks passed. Episode rows: {len(rows)}; held-out oracle states: {len(states)}. Oracle maxima are finite-K estimates under mu0, not deployment-time features or policy-invariant causal effects.']
    (output/'PILOT_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    save_json(ROOT/'artifacts/results_validation_report.json',{'run':str(runner.run),'critical_errors':critical,'final_manuscript_ready':False})
    make_figures(output,summary,diagnostic)

def make_figures(output,summary,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=output/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    datasets=sorted(set(r['dataset'] for r in summary)); methods=sorted(set(r['method'] for r in summary))
    fig,axes=plt.subplots(1,len(datasets),figsize=(6*len(datasets),4),squeeze=False)
    for ax,dataset in zip(axes[0],datasets):
        for method in methods:
            points=[r for r in summary if r['method']==method and r['dataset']==dataset]
            x=np.array([r['generated_tokens'] for r in points]); y=np.array([r['exact_match'] for r in points])
            low=np.array([r['exact_match_ci_low'] for r in points]); high=np.array([r['exact_match_ci_high'] for r in points])
            ax.errorbar(x,y,yerr=np.vstack([y-low,high-y]),fmt='o',capsize=2,label=method)
        n=sorted(set(r['n_tasks'] for r in summary if r['dataset']==dataset))
        ax.set(xlabel='Generated output tokens (including routing)',ylabel='Exact match',title=f'{dataset}: pilot N={n}; 95% task intervals',ylim=(-.05,1.05))
    axes[0,-1].legend(fontsize=7,ncol=3);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('quality_cost.'+ext))
    plt.close(fig)
    if diagnostics:
        d=[r for r in diagnostics if r['method']=='CRAVE' and r['lambda']==.1]
        fig,ax=plt.subplots(figsize=(5,4));ax.scatter([r['empirical_q'] for r in d],[r['predicted_q'] for r in d],alpha=.7,s=12)
        ax.plot([-1,1],[-1,1],'k--',lw=1);ax.set(xlabel='Empirical matched-branch utility',ylabel='Predicted utility',title=f'Pilot value diagnostic; {len(d)} action/seed estimates')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/('counterfactual_value.'+ext))
        plt.close(fig)
