"""Create reviewer deliverables and a numerical archive without raw benchmark text."""
import csv,datetime,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/mive_revision';ART=WORK/'artifacts';DEV=ART/'dev4b'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')

CHANGES=[
 ('Evidence provenance','Missing recovery package supports uncheckable results','Exclude every 1.5B gate/threshold and compact-refresh result; retain an unsupported-claim register','Verified manuscript Sections 5, 8 and Appendix B'),
 ('Independent continuation measurement','Legacy prediction residuals are not continuation mismatch','Restore checkpoints and generate disjoint A/B outcomes under mu0 and legacy CRAVE','dev4b/branches.jsonl and summary.json'),
 ('Variance gate','All actions may have identical reward','Record the exact zero-variance fraction and frozen checkpoint fallback outcome','dev4b/variance_gate.json'),
 ('K stability','Noisy action ranks can look like policy shift','Evaluate K=3 and K=5 independently; retain failed thresholds without certification','dev4b/stability.json'),
 ('Randomness audit','Matched seeds do not necessarily reduce variance','Compare seed-tape and independent contrasts at equal branch budgets','dev4b/checkpoint_metrics.json and utility_checkpoint_metrics.json'),
 ('Estimand correction','Raw EM STOP is an endpoint and cannot strictly reverse sign','Separate primary EM from a declared bounded resource-priced sensitivity','dev4b/secondary_analysis_declaration.json'),
 ('Theory','No finite-sample condition for reliable THINK STOP ranking','Add Theorem 3 with boundedness, independence, fixed-state assumptions and proof','Verified manuscript Section 4.4'),
 ('Negative findings','Accuracy-only reporting obscures additional computation','Retain all four policies, F1, output/input tokens, retrievals, verifications and latency','archival_reanalysis.json'),
 ('Figures','Low-information cluster charts','Add continuation heatmaps, uncertainty plots, noise decomposition and coupling scatter','figures/'),
 ('Related methods','Simplified proxies could be mislabeled reproductions','Compare estimand, continuation and full cost conditions for closest papers','Verified manuscript Table 1'),
 ('Structure and precision','Repetition and protocol status displace results','Use requested nine-section structure; move status and archival proxy to appendices; preserve raw full precision','Verified manuscript'),
 ('References','Unverified identifiers or mixed bibliographic formatting','Retain primary metadata and DOI registry responses; flag unresolved pages','artifacts/references/'),
 ('Completion scope','Scaled model and benchmark claims lack completed runs','Separate prospective intent from completed results; no fake final freeze or power calculation','prospective_scaled_plan.json'),
 ('Declarations and archive','Unconfirmed declarations and absent permanent DOI','Add AI and ethics statements; leave author facts and Zenodo DOI pending','Verified manuscript Declarations')]

UNSUPPORTED=[
 {'claim':'1.5B fixed-schedule baseline EM 0.25 / 0.275 and selected confidence threshold 0.5','reason':'User confirms the recovery package exists nowhere; no raw outcomes or freeze retained','replacement':'No verified result for that run; exclude its numerical claims'},
 {'claim':'Six-episode CPU check with 859 generated tokens and all incorrect answers','reason':'Only source manuscript assertions, no execution artifacts','replacement':'Do not report as a completed experiment'},
 {'claim':'Compact continuation-refresh training completed with 295 generation calls','reason':'Missing branch, generation and training ledgers','replacement':'Continuation refresh is proposed, with no verified held-out benefit'},
 {'claim':'MIVE improves accuracy or efficiency versus legacy/confidence','reason':'No new trained MIVE policy with a frozen held-out comparison','replacement':'A policy-conditioned estimand and a development audit are implemented'},
 {'claim':'7B or 1.5B held-out efficacy or model-size transfer','reason':'No sufficiently powered frozen held-out comparison for the additional backbones','replacement':'Retain completed frozen development execution checks separately, without model ranking or transfer claims'},
 {'claim':'Three datasets with 300 held-out questions, three seeds and three caps','reason':'No completed official-release manifests, training/dev finalization, or full inference grid','replacement':'Legacy 40-question one-cap evaluation and fresh small development audit'},
 {'claim':'Faithful Adaptive-RAG, Self-RAG or FLARE reproduction','reason':'No native classifier/reflection/draft pipeline reproduced with charged costs','replacement':'Conceptual comparison with explicit reproduction conditions'},
 {'claim':'Hard mask, semantic penalty, direct/composite, refresh and label-coupling ablation benefits','reason':'Full requested matched-budget ablation grid not completed','replacement':'Only measured coupling behavior is reported; no component-level causal benefit'},
 {'claim':'Noninferiority, savings, dev-powered sample size or a positive MIVE break-even count','reason':'Missing paired MIVE dev and held-out outcomes; legacy token savings are negative','replacement':'No NI claim; no positive finite legacy token break-even count'},
 {'claim':'Official answer evaluator equivalence and full-Wikipedia RAG generalization','reason':'Bounded remapped exports and supplied-context retrieval; alias equivalence not verified','replacement':'Normalized answer EM/F1 in the declared supplied-context environment'},
 {'claim':'Q1 acceptance readiness or reviewer-proof empirical conclusions','reason':'Scale, baselines, ablations and permanent archive remain incomplete','replacement':'Evidence-led methodological draft; submission needs remaining experiments and author confirmation'},
 {'claim':'A deposited Zenodo DOI','reason':'No Zenodo deposit exists','replacement':'Prepared archive only; insert a DOI after actual deposition'}]

def register():
    config=read(DEV/'freeze.json');completion=read(DEV/'completion.json')
    rows=[
      {'experiment':'Fresh development baseline and initial serial continuation/coupling collection',
       'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/audit.py run',
       'seed':'prefix 17; selection/bootstrap 20261011; every branch seed in branches.jsonl',
       'hardware':'NVIDIA GeForce RTX 3060 Laptop GPU 6144 MiB; Vulkan; four CPU threads; Windows; ctx4096; np1',
       'artifact':str(DEV),'config_sha256':config['sha256'],'status':'serial invocation interrupted for concurrency amendment; journal retained',
       'physical_scope':'See completion.json for all-attempt server totals, including discarded setup and interrupted work'},
      {'experiment':'Resume unchanged continuation/coupling matrix with four slots',
       'command':"$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; & crave/.venv/Scripts/python.exe experiments/mive_revision/parallel_audit.py",
       'seed':'same branch seeds; deterministic execution-order namespace execution-order-20261011',
       'hardware':'NVIDIA GeForce RTX 3060 Laptop GPU 6144 MiB; Vulkan1; four CPU inference threads; Windows; total ctx16384; four ctx4096 slots',
       'artifact':str(DEV/'runtime_amendment.json'),'status':completion['status'],'physical':completion},
      {'experiment':'Primary answer-value analysis and question bootstrap',
       'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/audit.py analyze',
       'seed':'20261011; 10000 question-cluster resamples','hardware':'CPU, NumPy/SciPy',
       'artifact':str(DEV/'summary.json'),'status':'complete'},
      {'experiment':'Secondary bounded utility and variance analysis',
       'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/analyze_utility.py',
       'seed':'20261011; 10000 question-cluster resamples','hardware':'CPU, NumPy/SciPy',
       'artifact':str(DEV/'utility_summary.json'),'status':'complete; analysis amendment during dev collection'},
      {'experiment':'Fresh branch reconstruction',
       'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/verify_audit.py',
       'seed':'none; reconstructs retained outcomes','hardware':'CPU',
       'artifact':str(DEV/'independent_verification.json'),'status':'complete'},
      {'experiment':'Archival full-result reconstruction',
       'command':'crave/.venv/Scripts/python.exe tmp/audit_all_results.py',
       'seed':'retained historical analysis seeds','hardware':'CPU',
       'artifact':str(ROOT/'tmp/full_manuscript_assets/comprehensive_result_audit.json'),'status':'complete; no fresh inference'},
      {'experiment':'Archival quality and every recorded resource',
       'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/prepare_evidence.py',
       'seed':'20261011 bootstrap','hardware':'CPU','artifact':str(ART/'archival_reanalysis.json'),'status':'complete; no fresh inference'},
      {'experiment':'Official 7B and 1.5B model acquisition',
       'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/download_models.py',
       'seed':'none','hardware':'network and local disk','artifact':str(ART),
       'status':'incomplete; full transfers timed out, bounded ranges reset; checksum verification not reached'},
      {'experiment':'Setup attempt 1','command':'audit.py run with failed_setup_freeze.json','seed':'17',
       'hardware':'same local 4B GPU server','artifact':str(DEV/'failed_setup_freeze.json'),
       'status':'failed after one fixed-schedule trajectory, incompatible return unpacking; no outcome included',
       'server_recorded_generation_calls':3,'server_recorded_output_tokens':194,
       'generation_timing_scope':'first three print_timing blocks in crave/runs/server.stderr.log; transport/wall overhead unavailable'},
      {'experiment':'Earlier pre-generation setup','command':'audit.py prepare/run before executor API fixes','seed':'17',
       'hardware':'CPU only before inference','artifact':str(DEV/'pre_execution_freeze_superseded_2.json'),
       'status':'failed before generation; Budget mapping API mismatch; superseded configs retained'}]
    for name,script,artifact in [
        ('Matched replicate pairing correction','correct_pairing.py','pairing_correction.json'),
        ('Independent statistical reconstruction','verify_statistics.py','statistics_verification.json'),
        ('Offline audit resource analysis','cost_analysis.py','offline_audit_cost.json'),
        ('Homogeneous-runtime coupling sensitivity','runtime_sensitivity.py','runtime_sensitivity.json'),
        ('Physical timing reconstruction','timing_ledger.py','numeric_server_timings.json')]:
        rows.append({'experiment':name,'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/'+script,
                     'seed':'20261011 for bootstrap analyses; none for deterministic checks','hardware':'CPU; see hardware_environment.json',
                     'artifact':str(DEV/artifact),'status':'complete'})
    for model in ['Qwen2.5-1.5B','Qwen2.5-7B']:
        location=ART/('smoke_'+model)
        if (location/'verified_results.json').exists():
            frozen=read(location/'freeze.json')
            rows.append({'experiment':model+' frozen development execution check',
                         'command':'crave/.venv/Scripts/python.exe experiments/mive_revision/backbone_smoke.py '+model+' prepare\ncrave/.venv/Scripts/python.exe experiments/mive_revision/backbone_smoke.py '+model+' run',
                         'seed':'17; one previously frozen question per dataset; two fixed policies',
                         'hardware':frozen['config']['hardware'],'artifact':str(location),
                         'status':'complete; six development episodes; no held-out efficacy claim','config_sha256':frozen['sha256']})
    for row in rows:
        if row['experiment']=='Official 7B and 1.5B model acquisition':
            verified=[m for m in ['Qwen2.5-1.5B','Qwen2.5-7B'] if (ART/(m+'_download_manifest.json')).exists() and read(ART/(m+'_download_manifest.json')).get('status')=='verified']
            row['status']='Checksum-verified: '+', '.join(verified)+'; earlier failed network attempts retained'
    hardware=read(ART/'hardware_environment.json')
    for row in rows:row['hardware'] += '; host '+hardware['cpu']['Name'].strip()
    write(ART/'experiment_register.json',rows)
    with open(WORK/'experiment_register.csv','w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['experiment','command','seed','hardware','artifact','status']);writer.writeheader()
        writer.writerows({k:r.get(k,'') for k in writer.fieldnames} for r in rows)
    return rows

def reviewer_outputs():
    write(WORK/'unsupported_claims.json',UNSUPPORTED)
    with open(WORK/'reviewer_change_map.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['Change','Reviewer concern','Revision','Evidence']);w.writerows(CHANGES)
    lines=['# Revision and reviewer concern map','',
           'This is a verified methods and diagnostic revision, not a completed scaled Q1 efficacy study. Missing recovery results are excluded. Raw retained scores remain unchanged; decimal formatting affects display only.','',
           '| Change | Reviewer concern | Revision | Evidence |','|---|---|---|---|']
    lines += ['| '+' | '.join(row)+' |' for row in CHANGES]
    if (DEV/'summary.json').exists():
        summary=read(DEV/'summary.json');gate=summary['variance_gate'];selection=summary['K_selection']
        lines += ['','## Completed development measurement','',
                  f'{summary["counts"]["branches"]} branches across {summary["counts"]["checkpoints"]} checkpoints and {summary["counts"]["questions"]} questions. Artifact: `{DEV / "summary.json"}`.',
                  f'Reward-variance gate: {"PASS" if gate["passed"] else "FAIL"}; identical action means at {gate["n_identical_action_means"]}/{gate["n_checkpoints"]} checkpoints. K selected by the complete A/B rule: {selection["selected"]}. If none passes, reported K={selection["reported_K"]} remains exploratory.',
                  'The theory now separates deterministic EM stopping from priced-value rank reversals and adds a bounded-return ranking condition. No continuation mechanism is asserted as the cause of the historical allocation failure.']
    lines += ['','## Unsupported claims and incomplete experiments','']
    for row in UNSUPPORTED:lines += ['- **'+row['claim']+'** — '+row['reason']+'. Honest replacement: '+row['replacement']+'.']
    lines += ['','## Remaining scaled study','',
              'The requested primary comparison requires 32,400 episodes before published baselines or ablations. The prospective plan is not a final freeze and permits no held-out inference. Complete official dataset/alias manifests, new MIVE training, development quality and K gates, paired development power, and final model-specific settings first. Additional backbone execution checks, when completed, are reported separately with their own frozen development configurations. No baseline proxy is presented as a published-method reproduction.','',
              'Zenodo is not connected and no deposit DOI exists. Funding and competing interests require explicit author confirmation. The archive contains sanitized numerical outcomes and code; raw benchmark text, model weights, binary runtimes and credentials are excluded.']
    (WORK/'revision_summary.md').write_text('\n'.join(lines),encoding='utf-8')

def archive():
    assert read(DEV/'independent_verification.json')['passed']
    assert read(DEV/'statistics_verification.json')['passed']
    assert read(DEV/'numeric_server_timings.json')['passed']
    layout=read(ART/'layout_verification.json')
    assert layout['passed'] and layout['pages_reviewed']==layout['page_count']
    assert layout['docx_sha256']==hashlib.sha256((ROOT/'MIVE_Verified_Manuscript.docx').read_bytes()).hexdigest()
    sanitized=[]
    for line in (DEV/'branches.jsonl').read_text().splitlines():
        row=json.loads(line)
        kept={k:v for k,v in row.items() if k!='final_state'}
        kept['action_counts']=dict(__import__('collections').Counter(x['action'] for x in row['final_state']['action_log']))
        kept['raw_record_sha256']=hashlib.sha256(line.encode()).hexdigest()
        sanitized.append(kept)
    write(ART/'sanitized_branch_outcomes.json',sanitized)
    checkpoints=[{k:r[k] for k in ['id','task_id','dataset','stage','feasible']}|{
        'state_hash':r['saved']['hash'],'candidate_answer_nonempty':bool(r['saved']['payload']['candidate_answer'])}
        for r in read(DEV/'checkpoints.json')]
    write(DEV/'sanitized_checkpoints.json',checkpoints)
    files=list(WORK.glob('*.py'))+list(WORK.glob('*.ps1'))+list(WORK.glob('*.md'))+list(WORK.glob('*.csv'))+list(WORK.glob('unsupported_claims.json'))
    files+=list((WORK/'figures').glob('*'))
    permitted=['freeze.json','summary.json','stability.json','variance_gate.json','initial_variance_gate.json','completion.json',
               'independent_verification.json','checkpoint_metrics.json','checkpoint_metrics_K3_K5.json',
               'utility_summary.json','utility_checkpoint_metrics.json','secondary_analysis_declaration.json','runtime_amendment.json',
               'utility_analysis_implementation.json','statistics_verification.json','sanitized_checkpoints.json',
               'offline_audit_cost.json','runtime_sensitivity.json','numeric_server_timings.json','pairing_correction.json']
    files += [DEV/name for name in permitted if (DEV/name).exists()]
    files += [ART/name for name in ['archival_reanalysis.json','prospective_scaled_plan.json','experiment_register.json','sanitized_branch_outcomes.json','manuscript_structure.json'] if (ART/name).exists()]
    registry=[]
    for path in sorted((ART/'references').glob('*_doi.json')):
        data=read(path)
        if 'message' in data:
            fields=['DOI','title','author','published','published-print','container-title','volume','issue','page','publisher','URL','type']
            metadata={k:data['message'][k] for k in fields if k in data['message']}
        else:
            fields=['doi','titles','creators','publicationYear','publisher','types','version','url','rightsList']
            metadata={k:data['data']['attributes'][k] for k in fields if k in data['data']['attributes']}
        registry.append({'source_record':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metadata':metadata})
    write(ART/'references/public_registry_metadata.json',registry)
    files += [ART/'references'/name for name in ['audit.json','reviewed_references.json','public_registry_metadata.json']]
    files += [path for path in (ART/'source_snapshots').glob('*') if path.is_file()]
    files += [ART/'hardware_environment.json']
    files += list(ART.glob('*_download_manifest.json'))
    for model in ['Qwen2.5-1.5B','Qwen2.5-7B']:
        location=ART/('smoke_'+model)
        files += [location/name for name in ['freeze.json','verified_results.json'] if (location/name).exists()]
    files += [ROOT/'MIVE_Verified_Manuscript.docx',ART/'layout_verification.json',WORK/'rendered/MIVE_Verified_Manuscript.pdf']
    manifest={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    write(ART/'archive_manifest.json',manifest);files.append(ART/'archive_manifest.json')
    dest=ROOT/'MIVE_Verified_Revision_Archive.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(ROOT).as_posix())
    print(json.dumps({'archive':str(dest),'files':len(files),'bytes':dest.stat().st_size}))

if __name__=='__main__':
    import sys
    if '--prepare-reviewer-output' in sys.argv:reviewer_outputs()
    elif '--register' in sys.argv:register()
    else:register();reviewer_outputs();archive()
