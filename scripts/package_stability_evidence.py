"""Export every registered paired endpoint, with portable path metadata only.

Inference weights and normalization are byte-preserved. Training resume pickles
are intentionally not exported; the original hashes and omissions are recorded.
No metric, failed episode, optimizer record or control trace is filtered.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(data):return hashlib.sha256(data).hexdigest()


def package(batch,output):
    batch,output=Path(batch).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve existing asset')
    protocol=read(batch/'protocol.json');completed=read(batch/'completed.json')
    if (batch/'failure.json').exists():raise ValueError('Incomplete or failed batch requires a separate export protocol')
    content,provenance={},{}
    def portable(value):
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,str):
            for prefix in (str(ROOT)+'\\',ROOT.as_posix()+'/'):
                if value.startswith(prefix):return value[len(prefix):].replace('\\','/')
        return value
    def add(path):
        relative=path.relative_to(batch).as_posix();original=path.read_bytes()
        data=json.dumps(portable(read(path)),indent=2).encode() if path.suffix=='.json' else original
        content[batch.name+'/'+relative]=data
        provenance[relative]={'original_sha256':sha(original),'export_sha256':sha(data),
            'transformation':'JSON path metadata made workspace-relative; values retained' if data!=original else 'none'}
    for relative,digest in protocol['source_hashes'].items():
        path=batch/'source'/relative
        assert sha(path.read_bytes())==digest
        add(path)
    initial_hashes={};omitted=[]
    for branch in protocol['branches']:
        name=branch['name'];folder=batch/name
        selected=[r for r in completed['results'] if r['branch']==name]
        assert [r['target'] for r in selected]==protocol['milestones']
        original_hashes=protocol.get('initial_checkpoint_hashes',{}).get(name,protocol['checkpoint_hashes'])
        for filename,digest in original_hashes.items():
            assert sha((folder/'initial_checkpoint'/filename).read_bytes())==digest
        checkpoints={folder/'initial_checkpoint'}
        for result in selected:
            path=folder/f"validation_{result['target']}.json";evaluation=read(path)
            assert len(evaluation['episodes'])==len(protocol['cases'])
            assert sum(e['success'] for e in evaluation['episodes'] if 'label' not in e)==result['full_task_success']
            checkpoint=folder/Path(evaluation['checkpoint']).name
            assert sha((checkpoint/'model.zip').read_bytes())==evaluation['model_sha256']
            checkpoints.add(checkpoint)
            add(path);add(folder/f"validation_{result['target']}_trace.jsonl.gz")
        for checkpoint in sorted(checkpoints):
            for filename in ('model.zip','normalization.pkl','config.json','training_summary.json'):
                add(checkpoint/filename)
            resume=checkpoint/'resume.pkl'
            if resume.exists():omitted.append({'file':resume.relative_to(batch).as_posix(),'sha256':sha(resume.read_bytes()),'reason':'Inference evidence export; not a portable training-resume bundle'})
        initial_hashes[name]={filename:sha(content[batch.name+'/'+(folder/'initial_checkpoint'/filename).relative_to(batch).as_posix()])
            for filename in ('model.zip','normalization.pkl','config.json','training_summary.json')}
        for filename in ('episodes.jsonl','optimization.jsonl','ppo_options.json','resolved_config.json','version.json'):
            add(folder/filename)
        for path in folder.glob('transitions*.jsonl.gz'):add(path)
        for path in folder.glob('*interventions.jsonl'):add(path)
    for filename in ('completed.json','initial_pairing_audit.json'):add(batch/filename)
    for folder in ('analysis','obstacle_response_lower'):
        for path in (batch/folder).glob('*'):
            if path.suffix in ('.json','.png','.gz'):add(path)
    protocol=portable(protocol)
    protocol['initial_checkpoint_hashes']=initial_hashes
    protocol['export_scope']='Inference evidence only; original checkpoint hashes retained; per-branch exported hashes reflect portable JSON and omitted resume.pkl'
    content[batch.name+'/protocol.json']=json.dumps(protocol,indent=2).encode()
    content[batch.name+'/export_provenance.json']=json.dumps({'original_protocol_sha256':sha((batch/'protocol.json').read_bytes()),
        'files':provenance,'omitted_resume_files':omitted,'scope':'All registered evaluation endpoints and new training control traces; periodic unevaluated checkpoints not duplicated'},indent=2).encode()
    patterns=[r'(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]',r'/(?:home|Users)/[A-Za-z0-9_.-]+/',
        r'\bgh[pousr]_[A-Za-z0-9]{30,}',r'\bgithub_pat_[A-Za-z0-9_]{40,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    def scan(name,data):
        if name.endswith('.png'):return
        if name.endswith('.gz'):data=gzip.decompress(data)
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.namelist():scan(member,archive.read(member))
            return
        if any(re.search(p,data.decode('utf-8',errors='replace')) for p in patterns):raise ValueError('Private data candidate: '+name)
    for name,data in content.items():scan(name,data)
    content['LICENSE']=(ROOT/'LICENSE').read_bytes()
    content['README.txt']=(
        'Paired PPO development evidence; one seed, correlated branches, no hierarchy superiority claim.\n'
        'All registered endpoints, failures and new training controls are included. All inherited and discarded sibling costs remain in parent reports.\n'
        'Inference model/norm bytes are unchanged. JSON absolute paths are made relative, with original/export digests.\n'
        'No CARLA binaries, full maps, environments, resume pickles or private runtime paths. This is not a training-resume bundle.\n'
        'From the source checkout: python scripts/analyze_update_stability.py --batch EXTRACTED_BATCH\n'
        'Load any endpoint using navstack.experiments.load_run(EXTRACTED_BATCH/BRANCH/CHECKPOINT).\n').encode()
    manifest={'scope':protocol['scope'],'files':{n:sha(d) for n,d in sorted(content.items())}}
    content['manifest.json']=json.dumps(manifest,indent=2).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(content.items()):archive.writestr(name,data)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist())==set(content)
        for name,digest in manifest['files'].items():assert sha(archive.read(name))==digest
    print(json.dumps({'files':len(manifest['files']),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes())}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();package(a.batch,a.output)
