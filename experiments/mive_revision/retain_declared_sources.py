"""Retain code bytes matching the scientific declarations, without outcome edits."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2];WORK=ROOT/'experiments/mive_revision';ART=WORK/'artifacts'
DEST=ART/'source_snapshots';DEST.mkdir(exist_ok=True)
def sha(content):return hashlib.sha256(content).hexdigest()
def retain(name,content,expected):
    assert sha(content)==expected,(name,sha(content),expected)
    target=DEST/name
    if target.exists():assert target.read_bytes()==content
    else:target.write_bytes(content)
    return {'path':str(target),'sha256':expected}
def main():
    records=[]
    freeze=json.loads((ART/'dev4b/freeze.json').read_text())
    expected=next(value for key,value in freeze['config']['source_sha256'].items() if key.endswith('audit.py'))
    records.append(retain('audit_initial_freeze.py',(WORK/'audit.py').read_bytes(),expected))
    amendment=json.loads((ART/'dev4b/runtime_amendment.json').read_text())
    records.append(retain('parallel_audit_runtime_amendment.py',(WORK/'parallel_audit.py').read_bytes(),amendment['source_sha256']))
    declaration=json.loads((ART/'dev4b/secondary_analysis_declaration.json').read_text())
    text=(WORK/'analyze_utility.py').read_text()
    text=text.replace("    # Exact integer units prevent floating-point ties from creating reversals.\n    assert row['exact_match'] in [0,1]\n    assert all(int(x)==x for x in c[:3])\n    return int(row['exact_match']*40960-c[0]-1024*c[1]-2048*c[2])", "    return row['exact_match']-.1*(c[0]/4096+c[1]/4+c[2]/2)")
    start=text.index("    save_json(OUT/'utility_analysis_implementation.json',{")
    end=text.index("    k=read_json",start)
    text=text[:start]+text[end:]
    text=text.replace("means={g:x.sum(1)/(40960*k) for g,x in data.items()}","means={g:x.mean(1) for g,x in data.items()}")
    text=text.replace("        mu=(data['ind_mu_A'].sum(1)+data['ind_mu_B'].sum(1))/(81920*k)\n        pi=(data['ind_pi_A'].sum(1)+data['ind_pi_B'].sum(1))/(81920*k)","        mu=(ma+mb)/2;pi=(pa+pb)/2")
    text=text.replace("/(k*40960**2)","/k")
    records.append(retain('analyze_utility_initial_declaration.py',text.encode(),declaration['config']['script_sha256']))
    smoke=json.loads((ART/'smoke_Qwen2.5-1.5B/freeze.json').read_text())
    original=DEST/'backbone_smoke_1_5B_freeze.py'
    records.append(retain(original.name,original.read_bytes() if original.exists() else (WORK/'backbone_smoke.py').read_bytes(),smoke['config']['source_sha256']))
    if (ART/'smoke_Qwen2.5-7B/freeze.json').exists():
        smoke=json.loads((ART/'smoke_Qwen2.5-7B/freeze.json').read_text())
        records.append(retain('backbone_smoke_7B_freeze.py',(WORK/'backbone_smoke.py').read_bytes(),smoke['config']['source_sha256']))
    (DEST/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
if __name__=='__main__':main()
