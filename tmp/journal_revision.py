"""Journal-oriented interpretation and sensitivity checks of frozen results."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ASSETS=ROOT/'tmp/full_manuscript_assets'

def rewrite(text):
    if text.startswith('Language model agents must allocate limited inference resources'):
        return ('Language model agents allocate inference resources among reasoning, retrieval, verification, and stopping. We present CRAVE, a controller trained on matched interventions from identical intermediate states to estimate policy-conditioned downstream utility. A frozen encoder and a compact neural controller predict action values and future resource use, while a feasibility mask enforces hard budgets. Four training questions produced 156 branch records. An initial six-question pilot was followed by a frozen-controller evaluation of 40 fresh questions from bounded HotpotQA and 2WikiMultiHopQA pools, yielding 480 episodes across three seeds and four methods. At a 512-output-token budget, CRAVE achieved 55.0% and 43.3% normalized exact match, versus 50.0% and 46.7% for a confidence rule. Paired accuracy differences were +5.0 percentage points (95% question-bootstrap interval 0.0 to 15.0) and −3.3 points (−10.0 to 0.0); both Holm-adjusted permutation p values were 1.0. CRAVE used 229.5 versus 51.2 and 249.7 versus 60.1 output tokens. Additional token costs were 178.3 (149.1 to 203.9) and 189.6 (165.9 to 211.4), with adjusted p = 0.0018 in both datasets. Repeated reasoning calls accounted for the additional generation. The accuracy differences each depended on one question. Audits found zero budget violations. CRAVE provides an executable framework for matched action valuation, but this small post-pilot study establishes no efficiency advantage over confidence. The public numerical release supports reconstruction of the reported analyses.')
    if text.startswith('This manuscript presents the formulation'):
        return ('We study whether matched downstream supervision can support a shared controller for heterogeneous inference actions. The contribution is an executable formulation and a transparent empirical test at one operating point. The initial pilot checks the learning and accounting machinery; a separate frozen-controller evaluation tests whether its behavior persists on fresh questions. The latter finds additional output consumption without a significant accuracy improvement over a confidence rule. This result motivates a precise distinction between enforcing a budget and allocating that budget effectively.')
    if text.startswith('The most direct comparison in the original six-question pilot'):
        return ('The expanded evaluation identifies an allocation failure: the learned controller performs about five reasoning calls per episode while confidence usually performs one. Additional generation was widespread across questions, but the accuracy differences were concentrated in one question per dataset. This pattern is consistent with overvalued continuation or weak stopping decisions. The current observations cannot distinguish training scarcity, noisy branch targets, continuation-policy mismatch, and inadequate state features as causes. A diagnostic result therefore guides the next experiment without establishing a causal explanation.')
    if text.startswith('The broader design asks whether matched long-horizon values'):
        return ('A complete evaluation should distinguish six questions: whether the controller improves quality and cost jointly, whether matched targets help when label quantity and action coverage are equal, whether delayed returns matter, whether ordered interactions help, whether budgets generalize, and whether behavior transfers. Table 10 specifies the evidence required for each question. The current release supports a methods and diagnostic study rather than a claim of benchmark superiority.')
    if text.startswith('CRAVE implements matched, policy-conditioned long-horizon valuation'):
        return ('CRAVE makes reasoning, retrieval, verification, and stopping comparable through matched action interventions and policy-conditioned return-to-go labels. Its executor guarantees compliance with the implemented hard resource caps when backend caps hold, while prediction quality determines how productively those resources are used. On the expanded 40-question test, CRAVE spent about four times the output tokens of a confidence rule. Paired accuracy differences were small, concentrated in one question per dataset, and not significant after multiplicity correction. Cost components explain why a weighted composite score can favor CRAVE over a verification-heavy schedule even when its generated-token consumption is higher. The contribution is a reproducible action-valuation framework and a documented failure case for learned allocation at small training scale. Coverage-matched supervision, development-only selection, larger official benchmark samples, and multiple backbones are needed to test whether the formulation can deliver an efficiency advantage.')
    if text.startswith('Code, pinned configuration files, dataset subset provenance'):
        return ('The public repository https://github.com/Al1ranger/CRAVE contains implementation code, frozen configurations, three learned CRAVE checkpoints, numerical episode records, statistical tables, audit reports, editable Word source, and figure builders. Released episodes omit answer text, questions, prompts, and passages. Their checksums are distinct from the original raw-record checksums, and the transformation is documented. The local retained records additionally contain benchmark exports, saved intervention states, raw answers, and complete trajectories used for score and restoration audits. Raw benchmark text is excluded from the public release pending verification of upstream redistribution terms. Readers can reconstruct the numerical analyses from the release; exact re-scoring and intervention-state audits require lawful access to the corresponding raw data. Pretrained model and encoder identifiers and digests are provided, but weights and runtime binaries are not bundled. A permanent archival DOI and complete official-release dataset identity have not been established.')
    return text

def add_contributions(p,h):
    h('1 1 Contributions and scope',2)
    p('CRAVE defines matched supervision for heterogeneous agent actions at a common saved state, including terminal answer reward and future resource consumption under a declared continuation policy. It combines utility, success, cost, ranking, and ordered-interaction objectives in a small controller without updating the language backbone. The release exposes question-level outcomes, paired uncertainty, action counts, and cost components alongside an independent reconstruction audit. The contribution concerns formulation, implementation, and a diagnostic evaluation; no result establishes that individual loss terms or counterfactual labels improve efficiency.')

def add_audit(p):
    audit=json.loads((ASSETS/'comprehensive_result_audit.json').read_text());assert audit['passed'];c=audit['counts']
    p(f'A comprehensive reconstruction recomputed all {c["answer_scores_and_episode_counters"]} retained answer-score and episode-counter records, {c["summary_metrics_with_intervals"]} summary metrics with bootstrap intervals, and all {c["paired_comparisons"]} paired comparisons with raw and Holm-adjusted p values. It checked {c["branch_payloads_scores_and_seeds"]} training and oracle branch payloads, scores, and matched seeds; reconstructed {c["oracle_action_predictions"]} action-value and success predictions from saved models; reconciled {c["diagnostic_summary_cells"]} diagnostic cells; and verified {c["retrieval_records"]} retrieval-support records. The maximum absolute reconstructed Q discrepancy was {audit["maximum_absolute_q_prediction_difference"]:.2g}, within the declared 2e-6 float32 tolerance. Checks reconstruct saved computation without rerunning stochastic language-model generation. Physical timings remain recorded measurements, and normalized scoring remains distinct from certification against official benchmark evaluators.')

def add_guarantees(p,h):
    h('4 4 Guarantees and estimation limits',2)
    p('Resource compliance follows by induction under the implemented executor assumptions. Initial hard budgets are nonnegative. Before each nonterminal action, the mask requires a reservation at least as large as the maximum charge for every constrained resource. Actual charges are nonnegative and no greater than those reservations, so subtraction preserves nonnegative remaining budgets. STOP incurs no constrained charge and remains available. The finite step limit permits a final STOP. This establishes compliance for output tokens, retrieval calls, and verification calls when the backend respects its caps; it does not guarantee a latency deadline, answer correctness, or useful allocation.')
    p('Matched randomness has a conditional variance benefit. For replicate returns U(a) and U(b) at one checkpoint, Var(Qhat(a) − Qhat(b)) = [Var(U(a)) + Var(U(b)) − 2 Cov(U(a), U(b))] / K when replicates are independent across replicate indices. Common random seeds can increase positive covariance and reduce contrast variance, but the covariance need not be positive after different interventions. Common seeds therefore support controlled comparison without guaranteeing lower variance. K = 3 gives limited resolution; branch records sharing a checkpoint are not independent training tasks.')
    p('A conditional action-selection bound clarifies continuation mismatch. If every feasible learned value differs from Q under μ0 by at most ε, the loss relative to the best feasible μ0 action is at most 2ε: compare the true value of the best action with its prediction, apply predicted argmax optimality, and compare the selected prediction with its true value. If values under deployment continuation π differ from μ0 values by at most δ for every feasible action, the corresponding one-decision loss under π is at most 2ε + 2δ. This bound does not establish an optimal full policy. Neither ε nor δ was estimated, so it cannot provide a numerical performance certificate.')

def generate_influence():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=json.loads((ROOT/'crave/runs/expanded_8484164aef776a12/evaluation.json').read_text())['data']
    records=[];fig,axes=plt.subplots(2,2,figsize=(6.8,4.2))
    for row,(dataset,title) in enumerate([('HotpotQA','HotpotQA'),('2WikiMultiHopQA','2Wiki')]):
        grouped=defaultdict(list)
        for e in raw:
            if e['dataset']==dataset and e['method'] in ['CRAVE','confidence']:grouped[(e['task_id'],e['method'])].append(e)
        ids=sorted({t for t,m in grouped});assert len(ids)==20
        for column,(metric,scale,label) in enumerate([('exact_match',100,'EM difference percentage points'),('generated_tokens',1,'Additional output tokens')]):
            delta=np.array([np.mean([e[metric] for e in grouped[(t,'CRAVE')]])-np.mean([e[metric] for e in grouped[(t,'confidence')]]) for t in ids])*scale
            full=float(delta.mean());deleted=(delta.sum()-delta)/19
            for task_id,value in zip(ids,deleted):records.append({'dataset':dataset,'metric':metric,'omitted_task_id':task_id,'full_estimate':full,'leave_one_out_estimate':float(value),'unit':'percentage points' if scale==100 else 'output tokens'})
            values,counts=np.unique(np.round(deleted,10),return_counts=True)
            for value,count in zip(values,counts):
                offsets=(np.arange(count)-(count-1)/2)*.045
                axes[row,column].scatter(np.full(count,value),offsets,s=15,color='#0072B2',alpha=.7)
            axes[row,column].axvline(full,color='#D55E00',ls='--',lw=1,label='All 20 questions')
            if column==0:
                axes[row,column].axvline(0,color='gray',lw=.8)
            else:
                low=min(deleted.min(),full);high=max(deleted.max(),full)
                pad=max(2,.15*(high-low))
                axes[row,column].set_xlim(low-pad,high+pad)
            axes[row,column].set(title=title,xlabel=label,yticks=[],ylim=(-.6,.6));axes[row,column].grid(axis='x',alpha=.15)
            axes[row,column].spines[['top','right','left']].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.065,1,1))
    for ext in ['png','pdf']:fig.savefig(ASSETS/f'question_influence.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    with open(ASSETS/'question_influence.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    result={d:{m:[min(r['leave_one_out_estimate'] for r in records if r['dataset']==d and r['metric']==m),max(r['leave_one_out_estimate'] for r in records if r['dataset']==d and r['metric']==m)] for m in ['exact_match','generated_tokens']} for d in ['HotpotQA','2WikiMultiHopQA']}
    (ASSETS/'question_influence_validation.json').write_text(json.dumps({'passed':True,'deletion_estimates':len(records),'ranges':result},indent=2));print(json.dumps(result))

def add_influence(p,h,figure):
    result=json.loads((ASSETS/'question_influence_validation.json').read_text())['ranges']
    h('7 8 Sensitivity to individual questions',2)
    p('We deleted one question at a time from each expanded-test dataset and recomputed CRAVE-minus-confidence means after averaging seeds. Figure 7 displays all twenty deletion estimates per dataset. This diagnostic uses the same observations and was added after inspecting the results; it is neither an independent validation sample nor a simultaneous confidence interval.')
    figure(ASSETS/'question_influence.png',6.25,'Figure 7  Influence of individual questions on CRAVE minus confidence. Each point omits one of twenty questions; vertical offsets separate coincident values. Dashed orange lines show the complete-sample means. Token axes focus on the deletion range. Deletion estimates are dependent sensitivity diagnostics, not new observations or confidence intervals. Exact task identities and estimates are recorded in question_influence.csv.')
    p(f'HotpotQA accuracy differences range from {result["HotpotQA"]["exact_match"][0]:.2f} to {result["HotpotQA"]["exact_match"][1]:.2f} percentage points, and 2Wiki differences from {result["2WikiMultiHopQA"]["exact_match"][0]:.2f} to {result["2WikiMultiHopQA"]["exact_match"][1]:.2f}. Removing the sole question with an accuracy difference makes the respective dataset difference zero. Additional output remains positive for every deletion, ranging from {result["HotpotQA"]["generated_tokens"][0]:.1f} to {result["HotpotQA"]["generated_tokens"][1]:.1f} tokens on HotpotQA and {result["2WikiMultiHopQA"]["generated_tokens"][0]:.1f} to {result["2WikiMultiHopQA"]["generated_tokens"][1]:.1f} on 2Wiki. This contrast strengthens the descriptive cost finding while exposing the fragility of the accuracy estimates.')

def add_interpretation(p,h):
    h('9 4 Interpretation of statistical evidence',2)
    p('A nonsignificant accuracy comparison is not evidence of equivalent policies. No equivalence margin was specified, and the accuracy intervals admit changes that could matter in deployment. Pointwise bootstrap intervals and Holm-adjusted sign-randomization tests address sampling uncertainty and a corrected null comparison, respectively. Sign-randomization additionally requires exchangeability under the tested null, or a suitable symmetry assumption for paired differences. Questions were not randomized to policies, and no design-based causal treatment effect is claimed. Multiplicity correction handles the declared comparisons, not the broader adaptivity of a follow-up motivated by an inspected pilot.')
    p('The engineering implication is to inspect the resource vector before choosing a scalar utility. Fixed verification adds 0.5 to the implemented normalized cost, whereas output generation is divided by 4096. A policy can therefore lower that composite score while consuming more tokens. Cost-reference sensitivity, input-token charging, and cold latency should be specified for the intended deployment before optimizing efficiency. Reweighting existing trajectories characterizes those recorded trajectories; it does not show how a controller behaves after retraining or deployment with new preferences.')

if __name__=='__main__':generate_influence()
