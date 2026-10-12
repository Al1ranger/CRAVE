"""Report coupling sensitivity to the recorded serial/four-slot amendment."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'
def read(name):return json.loads((OUT/name).read_text())
def main():
    rows=[json.loads(x) for x in (OUT/'branches.jsonl').read_text().splitlines()]
    versions={cp:{r.get('runtime_version','initial-serial') for r in rows if r['checkpoint']==cp}
              for cp in {r['checkpoint'] for r in rows}}
    mixed=sorted(cp for cp,values in versions.items() if len(values)>1)
    report={'mixed_runtime_checkpoints':mixed,'reason':'execution-only amendment; no outcome-dependent filtering',
            'limits':'Homogeneous-runtime sensitivity excludes mixed checkpoints, not their outcomes from the main reported audit.',
            'analyses':{}}
    for objective,file in [('terminal_EM','checkpoint_metrics.json'),('priced_value','utility_checkpoint_metrics.json')]:
        records=[r for r in read(file) if r['checkpoint'] not in mixed]
        rng=np.random.default_rng(20261011);summary={}
        for metric in ['independent_gap_variance','matched_gap_variance','matched_minus_independent']:
            values=[]
            for question in sorted({r['task_id'] for r in records}):
                selected=[r for r in records if r['task_id']==question and r['independent_gap_variance'] is not None]
                if selected:values.append(np.mean([r['matched_gap_variance']-r['independent_gap_variance'] if metric=='matched_minus_independent' else r[metric] for r in selected]))
            values=np.array(values)
            bootstrap=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            summary[metric]={'estimate':float(values.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'n_questions':len(values)}
        report['analyses'][objective]={'checkpoints':len(records),'statistics':summary}
    (OUT/'runtime_sensitivity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
