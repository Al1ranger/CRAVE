"""Exact audit charges and cluster intervals for mean cost per question."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'
def main():
    assert json.loads((OUT/'independent_verification.json').read_text())['passed']
    raw=[json.loads(line) for line in (OUT/'branches.jsonl').read_text().splitlines()]
    questions=sorted({r['task_id'] for r in raw});rng=np.random.default_rng(20261011)
    report={'questions':len(questions),'branches':len(raw),'resources':{},
            'scope':'future charges in completed audit branches only; prefixes, baselines, setup and interrupted work excluded',
            'interval_scope':'bootstrap interval for mean audit cost per question, not uncertainty in the observed total',
            'break_even':'No MIVE deployment savings measured; no new MIVE break-even estimate. Legacy token savings are negative.'}
    for i,resource in enumerate(['generated_tokens','retrieval_calls','verification_calls','logical_seconds']):
        values=np.array([sum(r['cost'][i] for r in raw if r['task_id']==q) for q in questions])
        bootstrap=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
        report['resources'][resource]={'observed_total':float(values.sum()),'mean_per_question':float(values.mean()),
                                       'mean_per_question_ci95':np.quantile(bootstrap,[.025,.975]).tolist()}
    (OUT/'offline_audit_cost.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
