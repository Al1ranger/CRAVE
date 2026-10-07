"""Editorial integration of the completed, independently audited expanded test."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
RUN=Path(json.loads((ROOT/'crave/artifacts/expanded_test/LATEST_RUN.json').read_text())['run_directory'])
def read(name):
    with open(RUN/'results'/name,encoding='utf-8') as f:return list(csv.DictReader(f))
SUMMARY=read('summary.csv');STATS=read('statistics.csv')
LOOKUP={(r['dataset'],r['method']):r for r in SUMMARY}
CONFIG=json.loads((RUN/'FROZEN_EXPANDED_CONFIG.json').read_text())
COMPUTE=json.loads((RUN/'compute_accounting.json').read_text())
AUDIT=json.loads((RUN/'results/independent_audit.json').read_text())
assert AUDIT['passed'] and AUDIT['episodes']==480
def f(r,k,d=1):return f'{float(r[k]):.{d}f}'
def ci(r,k,scale=1,d=1):
    return f'{float(r[k])*scale:.{d}f} [{float(r[k+"_ci_low"])*scale:.{d}f}, {float(r[k+"_ci_high"])*scale:.{d}f}]'
def rewrite(text):
    if text.startswith('Summaries and capped evidence are lossy representations'):
        text=text.replace('a maximum of twelve executed actions bound each rollout','a maximum of twelve nonterminal steps bound each rollout; an additional STOP action can terminate at that limit')
    if text.startswith('The fixed corpus pools all 300 unique passages'):
        text=text.replace('The fixed corpus pools','The original feasibility corpus pools',1)
    if text.startswith('Language model agents must decide'):
        return ('Language model agents must allocate limited inference resources among reasoning, retrieval, verification, and termination. CRAVE learns policy-conditioned long-horizon action values through matched counterfactual branching from identical intermediate states. A frozen encoder and compact multilayer perceptron predict utility, eventual success, future resource costs, and auxiliary ordered interactions; a hard feasibility mask enforces output-token and tool limits. Training used four HotpotQA and MuSiQue tasks, two collection rounds, 156 matched branch records, and three stochastic continuations per intervention. An initial six-question feasibility pilot comprised 162 episodes. A subsequent frozen-controller test evaluated forty new questions, twenty each from HotpotQA and 2WikiMultiHopQA, with three seeds and four methods, producing 480 additional episodes. At a 512-token budget and utility weight 0.1, CRAVE achieved 55.0% and 43.3% exact match, compared with 50.0% and 46.7% for a confidence rule. CRAVE consumed 229.5 versus 51.2 and 249.7 versus 60.1 generated tokens. Its paired additional output cost was 178.3 tokens on HotpotQA and 189.6 on 2Wiki, with Holm-adjusted Monte Carlo permutation p = 0.0018 in both dataset cells. The paired accuracy differences were not significant after correction. No observed efficiency advantage over the confidence rule was established. Independent audits reconciled scores, resource counters, task partitions, and artifact identities, finding zero budget violations. Per-question figures expose cost variation and the limited concentration of accuracy differences. These are bounded, post-pilot results from raw benchmark subsets and a transductive passage pool. Full-data training, development selection, broad budget frontiers, faithful literature baselines, and cross-model validation remain unexecuted.')
    if text.startswith('This manuscript presents the formulation'):
        text=text.replace('The completed experiment is deliberately small:','The original feasibility pilot was deliberately small:')
        text+=' A separate frozen-controller follow-up evaluates forty additional questions with a larger shared passage corpus; Section 7.6 reports that protocol and its results separately.'
    if text.startswith('HotpotQA supports multi-document'):
        text=text.replace('The completed run used small first-record subsets','The original feasibility run used small first-record subsets')
    if text.startswith('The most direct completed comparison is against the confidence rule.'):
        text=text.replace('The most direct completed comparison is against the confidence rule.','The most direct comparison in the original six-question pilot was against the confidence rule.')
    if text.startswith('Six questions cannot support a general benchmark conclusion.'):
        text=text.replace('Six questions cannot support a general benchmark conclusion.','The original six-question pilot cannot support a general benchmark conclusion. The expanded test adds forty fresh questions selected randomly from bounded record pools, but it remains a small post-pilot study rather than a representative full-benchmark evaluation.')
    if text.startswith('CRAVE implements matched, policy-conditioned long-horizon valuation'):
        text+=' The expanded test preserves this limitation: a five-percentage-point HotpotQA accuracy increase and a 3.3-point 2Wiki decrease relative to confidence were each concentrated in one question and were not significant after correction, whereas the additional generated-token cost was consistently positive. Frozen controllers, fresh questions, and task-level plots strengthen the descriptive evidence without establishing a favorable full quality-cost frontier.'
    if text.startswith('Code, pinned configuration files, dataset subset provenance'):
        text+=' The expanded test is saved separately in crave/runs/'+RUN.name+'. It reuses the original seeded CRAVE checkpoints and retains an independently frozen extension configuration, random selection manifest, per-question means, paired statistical outputs, and two replacement figures.'
    if text.startswith('The broader design asks whether matched long-horizon values'):
        text=text.replace('Table 8 records','Table 10 records')
    if text.startswith('Table 8  Research hypotheses'):
        text=text.replace('Table 8','Table 10',1)
    if text.startswith('The pilot configuration SHA256 is '):
        text+=' The expanded test configuration SHA256 is '+CONFIG['sha256']+'. It retains the same framework source identity and frozen controller checkpoint identities while changing the held-out questions and shared corpus.'
    if text.startswith('Checkpoint and raw trajectory files are essential'):
        text+=' For the expanded test, task_means.csv contains seed-averaged question observations, task_manifest.json retains the selected questions and original partition boundaries, and results/figures/paired_question_key.csv maps each chart label to its exact task ID. The extension runner is tmp/expanded_evaluation.py; its frozen SHA256 is '+CONFIG['config']['extension_script_hash']+'.'
    if text.startswith('Results average the three seeds within each question'):
        text+=' The resampling framework follows the bootstrap principle [24], and the multiple-test adjustment follows Holm [25].'
    if text.startswith('Code, pinned configuration files, dataset subset provenance'):
        text=text.replace('A public repository, archival identifier, and complete full-release dataset manifest have not been established.','The repository is https://github.com/Al1ranger/CRAVE. A permanent archival identifier and complete full-release dataset manifest have not been established.')
    if text.startswith('The test completed 480 episodes'):
        text=text.replace('random sign permutations with an add-one correction','random sign permutations with an add-one correction [26]')
    return text

def add_results(p,h,caption,table,figure):
    h('7.6 Expanded frozen controller evaluation',2)
    p('After inspecting the feasibility pilot, we froze an expanded evaluation with twenty new HotpotQA questions and twenty new 2Wiki questions. A NumPy generator with seed 20261005 sampled without replacement from 131 eligible HotpotQA and 125 eligible 2Wiki candidates. IDs, normalized question strings, and available component-question identifiers were disjoint from training, reserved development, and the original evaluation. The candidate pools were bounded public exports: the original forty HotpotQA validation records plus one hundred additional dataset-server records at offset forty, and the first 128 records of the author-linked 2Wiki development raw export. This procedure randomizes selection within those pools; it does not establish a probability sample of a complete benchmark release.')
    p('The original CRAVE checkpoints for seeds 17, 29, and 43 were loaded without retraining. The confidence threshold, backbone, action definitions, 512 generated-token budget, one retrieval and one verification allowance, and deployment λ = 0.1 were unchanged. CRAVE, confidence, the fixed schedule, and the LLM router shared a 687-passage corpus constructed without answer labels from the original partitions and new question contexts. The larger corpus and new question set distinguish this protocol from the original pilot; the two result sets are not pooled into one estimate. No additional ablations or held-out value-oracle states were collected in this follow-up.')
    p('The test completed 480 episodes: forty questions multiplied by three seeds and four methods. Seeds were averaged within question before task-cluster bootstrap intervals were computed using ten thousand resamples. Paired comparisons used ten thousand random sign permutations with an add-one correction, because each dataset has twenty clusters. Holm correction covered a separate family of eighteen exploratory tests: two datasets, three comparators, and exact match, generated tokens, and normalized cost. This post-pilot follow-up was motivated by the earlier observations and is not presented as an independently preregistered confirmatory study.')
    caption('Table 8  Expanded test means and 95 percent task-cluster intervals')
    rows=[]
    for dataset,short in [('HotpotQA','Hotpot'),('2WikiMultiHopQA','2Wiki')]:
        for method,name in [('CRAVE','CRAVE'),('confidence','Confidence'),('fixed','Fixed'),('llm_router','LLM router')]:
            r=LOOKUP[(dataset,method)]
            rows.append([short,name,ci(r,'exact_match',100),f'{100*float(r["f1"]):.1f}',ci(r,'generated_tokens')])
    table(['Dataset','Method','EM percent [CI]','F1 percent','Output tokens [CI]'],rows,[.75,1.15,1.65,1.0,2.0])
    figure(RUN/'results/figures/expanded_quality_cost.png',6.25,'Figure 1  Expanded held-out test with twenty independent question clusters per dataset. Faint points are question-level observations after averaging the three seeds; diamonds are method means and horizontal bars are 95% task-cluster bootstrap intervals. The generated-token axis includes routing output. The points describe one deployment setting, not a budget frontier.')
    p('CRAVE exact match increased from 50.0% to 55.0% relative to confidence on HotpotQA, and decreased from 46.7% to 43.3% on 2Wiki. Its F1 was 62.8% and 49.4%, versus 60.3% and 50.0% for confidence. The paired accuracy differences were +5.0 percentage points with a [0.0, 15.0] interval on HotpotQA and −3.3 points with a [−10.0, 0.0] interval on 2Wiki. Both Holm-adjusted accuracy p values were 1.0. The descriptive point estimates therefore do not establish an accuracy improvement over confidence.')
    p('CRAVE generated 229.5 versus 51.2 tokens on HotpotQA and 249.7 versus 60.1 on 2Wiki, approximately 4.48 and 4.16 times the confidence rule. Paired additional output was 178.3 [149.1, 203.9] and 189.6 [165.9, 211.4] tokens, respectively. Both Holm-adjusted output-cost p values were 0.0018. These exploratory tests support a consistent output-cost increase in the selected pools, while broad benchmark representativeness and favorable quality-cost tradeoffs remain unestablished.')
    caption('Table 9  Expanded paired comparisons with CRAVE minus each control')
    rows=[]
    labels={'confidence':'Confidence','fixed':'Fixed','llm_router':'LLM router'}
    for dataset,short in [('HotpotQA','Hotpot'),('2WikiMultiHopQA','2Wiki')]:
        for other in ['confidence','fixed','llm_router']:
            for metric in ['exact_match','generated_tokens']:
                r=next(r for r in STATS if r['dataset']==dataset and r['comparison']=='CRAVE minus '+other and r['metric']==metric)
                scale=100 if metric=='exact_match' else 1
                rows.append([short,labels[other],'EM points' if metric=='exact_match' else 'Output tokens',f'{float(r["estimate"])*scale:.1f}',f'[{float(r["ci_low"])*scale:.1f}, {float(r["ci_high"])*scale:.1f}]',f'{float(r["holm_p"]):.4f}'])
    table(['Dataset','Control','Metric','Difference','95 percent interval','Holm p'],rows,[.70,1.10,1.10,.85,1.75,1.05])
    p('Relative to the fixed schedule, CRAVE used 56.6 additional output tokens on HotpotQA and 58.5 on 2Wiki, with adjusted p values 0.0369 and 0.0060. Its normalized composite cost was lower by 0.4281 and 0.4119, respectively, because the schedule incurs verification-call cost whereas CRAVE generally avoids verification. Thus, a reduction in the reference-weighted composite cost does not imply a reduction in generated tokens. Relative to the LLM router, CRAVE had higher observed exact match in both datasets but also greater output consumption; neither accuracy comparison was significant after the stated correction.')
    figure(RUN/'results/figures/paired_question_costs.png',6.25,'Figure 2  Per-question additional generated tokens for CRAVE relative to confidence, ordered within dataset after averaging seeds. Color records the corresponding direction of the exact-match difference. Gray means equal seed-averaged exact match. H and W labels map to exact task IDs in paired_question_key.csv. A zero-length bar is a cost tie. Ordering is visual only and did not determine task selection or statistical tests.')
    p('Nineteen of twenty HotpotQA questions and all twenty 2Wiki questions incurred a positive CRAVE output-cost difference; one HotpotQA question was tied. Seed-averaged accuracy differed on only one question per dataset: higher for CRAVE on HotpotQA and lower on 2Wiki. The per-question display therefore reveals that higher spending was widespread, whereas the observed accuracy changes were concentrated in individual questions. It provides substantially more task-level information than the original three-cluster plot.')
    p(f'The expanded invocation made {COMPUTE["physical_requests_this_invocation"]:,} physical model requests and generated {COMPUTE["physical_output_tokens_this_invocation"]:,} physical output tokens in {COMPUTE["invocation_seconds"]/60:.1f} minutes. Logical evaluation output was {COMPUTE["logical_evaluation_output_tokens"]:,} tokens and input was {COMPUTE["logical_evaluation_input_tokens"]:,}; cached computation accounts for the difference between physical and logical generation. Model generation time was {COMPUTE["physical_generation_seconds_this_invocation"]:.1f} seconds. These counts exclude data download and the earlier pilot. No cold-latency or monetary-cost estimate is inferred.')
    p('The extension validation and a separate audit both passed. Independent checks verified the configuration, source, corpus, selected tasks, and frozen checkpoint hashes; exact episode completeness and pairing; disjointness from all original partitions; all 480 answer scores and resource counters; and thirty-two independently reconstructed summary means and bootstrap intervals. There were zero budget violations. The larger sample strengthens the descriptive negative cost finding, but training remains limited to four questions, official evaluator equivalence remains unverified for raw exports, and the full study gate remains unmet.')
