"""Publish numerical server timing blocks without prompts or benchmark text."""
import hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/mive_revision/artifacts/dev4b'
def main():
    records=[];sources={}
    pattern=re.compile(r'slot print_timing: id\s+(\d+)\s*\| task\s+(\d+)\s*\|\s*(prompt eval time|eval time|total time)\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens')
    for name in ['initial_server.stderr.log','parallel_server.stderr.log']:
        path=OUT/name;sources[name]=hashlib.sha256(path.read_bytes()).hexdigest();blocks={}
        for line in path.read_text(errors='replace').splitlines():
            found=pattern.search(line)
            if not found:continue
            slot,task,kind,duration,tokens=found.groups()
            row=blocks.setdefault((slot,task),{'source_log':name,'slot':int(slot),'task':int(task)})
            key={'prompt eval time':'prompt','eval time':'decode','total time':'total'}[kind]
            row[key+'_milliseconds']=float(duration);row[key+'_tokens']=int(tokens)
        records.extend(r for r in blocks.values() if 'total_milliseconds' in r)
    totals={'completed_requests':len(records),'decoded_tokens':sum(r.get('decode_tokens',0) for r in records),
            'prompt_and_decode_seconds':sum(r['total_milliseconds'] for r in records)/1000}
    completion=json.loads((OUT/'completion.json').read_text())
    assert totals['completed_requests']==completion['server_recorded_generation_calls_all_attempts']
    assert totals['decoded_tokens']==completion['server_recorded_output_tokens_all_attempts']
    assert abs(totals['prompt_and_decode_seconds']-completion['server_recorded_generation_seconds_all_attempts'])<1e-8
    report={'passed':True,'sources_sha256':sources,'totals':totals,'records':records,
            'scope':'Completed server timing blocks only. Excludes startup, transport and requests without a completion record. Durations sum concurrent work, not elapsed wall time.'}
    (OUT/'numeric_server_timings.json').write_text(json.dumps(report,indent=2));print(json.dumps(totals))
if __name__=='__main__':main()
