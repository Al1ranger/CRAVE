# CRAVE Windows runbook

Commands below run from `D:\Study\Papers\p6`. Python 3.12.14 is used because the bundled runtime supplies it; Python >=3.11 is supported. All installations/downloads remain inside this workspace except ordinary pip's external cache.

## Environment

```powershell
& 'C:\Users\ehsan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv --system-site-packages crave/.venv
& crave/.venv/Scripts/python.exe -m pip install --timeout 120 --retries 5 -r crave/requirements.txt
```

Never bypass a pip hash mismatch. Retry the affected wheel from its official package source and retain the error log.

## Assets and data

```powershell
& crave/.venv/Scripts/python.exe -u -m crave.scripts.prepare all
& crave/.venv/Scripts/python.exe -u -m crave.scripts.prepare_author_data
```

Pinned model/encoder revisions, runtime release, SHA256s are recorded in artifacts/model_manifest.json. A .part file is not a usable model. Preparation resumes partial model files and rejects mismatched digests. Public HotpotQA rows are preserved with source URL and repository revision; returned dataset-server rows are checksummed, but the server snapshot-to-repository linkage requires additional verification for final experiments.

The author-data command saves small raw MuSiQue train/development and 2Wiki development exports linked by the MuSiQue authors. These variants carry explicit provenance and subset checksums; they do not establish equivalence to the complete official releases. If available, official `musique_ans_v1.0_train.jsonl` and `2wiki_dev.json` take precedence. Their licenses and full source checksums must be recorded before a final run.

The pilot corpus pools all selected benchmark paragraphs without consulting supporting facts or answers. This is not the preregistered full-Wikipedia setting. Prepare and checksum a full snapshot before final experiments.

## Tests and inference

```powershell
& crave/.venv/Scripts/python.exe -m unittest discover -s crave/tests -v
& crave/scripts/start_server.ps1
Invoke-RestMethod http://127.0.0.1:8096/health
```

The server is local-only and selects Vulkan1, observed to be the RTX 3060. Recheck `llama-server --list-devices` on another machine. Server PID and stdout/stderr are in crave/runs. Stop only the PID recorded for this CRAVE server when finished.

## Tiny real-data smoke

```powershell
$env:OPENBLAS_NUM_THREADS='2'
& crave/.venv/Scripts/python.exe -u -m crave.pipeline --train-tasks 2 --eval-tasks 3 --budgets 1 --interaction-states 1
```

This command runs real Qwen inference, matched K=3 branches, two collection rounds, three seeds, fixed/confidence/router/myopic, decisive ablations, held-out branch diagnostics, paired exploratory statistics, CSV tables, and two plots. It is a smoke run, not a hypothesis test with adequate sample size.

## Larger pilot

```powershell
& crave/.venv/Scripts/python.exe -u -m crave.pipeline --train-tasks 4 --eval-tasks 25 --budgets 2 --full-lambda-sweep
```

Stages and individual episodes resume under their exact code/config hash. Do not change code or config during evaluation. `FROZEN_PILOT_CONFIG.json` is a pilot freeze only. Raw stage JSON, per-instance CSV, paired statistics, provenance, validation and PILOT_REPORT.md are stored under crave/runs/<hash>/results. Generation caches store original measured latency and distinguish physical from logical costs.

## Full study gate

Before final evaluation: finish both training datasets, development-only confidence/K/hyperparameter selection, exact split manifests, full Wikipedia retrieval, official baseline fidelity, unseen budget definition, full sweep, second model and OOD where feasible. Record all decisions; independently freeze final config. The current runner cannot certify this gate and cannot produce a submission-ready final manuscript.

Manuscript results may be edited only from validated final artifacts; preserve negative findings and mark unexecuted comparisons NOT RUN. Original manuscript remains untouched while this gate is incomplete.
