"""Check the prepared public archive's hashes and numerical-only data boundary."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[2]

def main():
    with zipfile.ZipFile(ROOT/'MIVE_Verified_Revision_Archive.zip') as archive:
        names=archive.namelist();assert len(names)==len(set(names))
        assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in names)
        manifest=json.loads(archive.read('experiments/mive_revision/artifacts/archive_manifest.json'))
        assert set(names)==set(manifest)|{'experiments/mive_revision/artifacts/archive_manifest.json'}
        for name,digest in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
        forbidden={'question','answer','aliases','candidate_answer','final_state','saved','payload','passages'}
        def check(value,path):
            if isinstance(value,dict):
                assert not forbidden.intersection(value),path
                for key,item in value.items():check(item,path+'/'+key)
            elif isinstance(value,list):
                for item in value:check(item,path)
        json_files=[n for n in names if n.endswith('.json')]
        for name in json_files:check(json.loads(archive.read(name)),name)
        rows=json.loads(archive.read('experiments/mive_revision/artifacts/sanitized_branch_outcomes.json'))
        assert len(rows)==1080 and all('raw_record_sha256' in row for row in rows)
        print(json.dumps({'passed':True,'files':len(names),'json_files_without_raw_answer_fields':len(json_files),'sanitized_branches':len(rows),'all_manifest_hashes_match':True}))
if __name__=='__main__':main()
