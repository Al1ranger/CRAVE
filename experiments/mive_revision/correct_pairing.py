"""Correct primary matched-action pairing without changing the frozen collector.

Parallel journal order is not replicate order. Preserve the initial analysis
and align by replicate identity before estimating a paired contrast variance.
"""
import hashlib,json,pathlib,shutil
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'

def read(name):return json.loads((OUT/name).read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    backup=OUT/'analysis_before_pairing_correction';backup.mkdir(exist_ok=True)
    for name in ['summary.json','checkpoint_metrics.json','checkpoint_metrics_K3_K5.json','utility_summary.json','utility_checkpoint_metrics.json']:
        if not (backup/name).exists():shutil.copy2(OUT/name,backup/name)
    journal=OUT/'branches.jsonl';before=sha(journal)
    raw=[json.loads(line) for line in journal.read_text().splitlines()]
    estimates=read('checkpoint_metrics_K3_K5.json');changes=[]
    for k_text,records in estimates.items():
        k=int(k_text)
        for row in records:
            actions=list(row['action_values_mu']);paired=[]
            if 'THINK' not in actions:continue
            for policy in ['mu','pi']:
                values={action:{r['replicate']:r['exact_match'] for r in raw
                    if r['checkpoint']==row['checkpoint'] and r['group']=='matched_'+policy
                    and r['action']==action and r['replicate']<k} for action in actions}
                assert all(set(v)==set(range(k)) for v in values.values())
                for action in actions:
                    if action!='THINK':
                        gaps=[values['THINK'][rep]-values[action][rep] for rep in range(k)]
                        paired.append(float(np.var(gaps,ddof=1)/k))
            corrected=float(np.mean(paired))
            changes.append({'checkpoint':row['checkpoint'],'K':k,'before':row['matched_gap_variance'],'after':corrected})
            row['matched_gap_variance']=corrected
    summary=read('summary.json');selected=str(summary['K_selection']['reported_K']);records=estimates[selected]
    rng=np.random.default_rng(20261011)
    for metric in records[0]:
        if metric in ['checkpoint','task_id','dataset','K','action_values_mu','action_values_pi']:continue
        values=[]
        for question in sorted({r['task_id'] for r in records}):
            cell=[r[metric] for r in records if r['task_id']==question and r[metric] is not None]
            if cell:values.append(np.mean(cell))
        if not values:continue
        values=np.array(values);draws=rng.integers(0,len(values),(10000,len(values)))
        if metric=='matched_gap_variance':
            summary[metric]={'estimate':float(values.mean()),'ci95':np.quantile(values[draws].mean(1),[.025,.975]).tolist(),'n_questions':len(values)}
    write('checkpoint_metrics_K3_K5.json',estimates);write('checkpoint_metrics.json',records);write('summary.json',summary)
    assert before==sha(journal)
    report={'raw_journal_sha256':before,'raw_journal_unchanged':True,'primary_changes':changes,
        'reason':'Independent verifier detected a priced matched-variance discrepancy; journal order differed across actions.',
        'correction':'Align every paired contrast by replicate ID. Primary frozen collector source remains unchanged; secondary analysis sorts replicates.',
        'initial_analysis_sha256':{p.name:sha(p) for p in backup.glob('*.json')},'source_sha256':sha(pathlib.Path(__file__))}
    write('pairing_correction.json',report)
    print(json.dumps({'raw_journal_unchanged':True,'changed_primary_cells':sum(x['before']!=x['after'] for x in changes)}))
if __name__=='__main__':main()
