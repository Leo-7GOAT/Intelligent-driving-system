"""Transfer a checked evidence ZIP in small parts and reconstruct exact bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

VERSION='native_asset_parts_v1'
MAX_PART=8*1024**2


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def split(archive,directory,part_bytes=MAX_PART):
    archive,directory=Path(archive),Path(directory)
    if type(part_bytes) is not int or not 1<=part_bytes<=MAX_PART:
        raise ValueError('Invalid bounded part size')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',archive.name) or not 0<archive.stat().st_size<=2*1024**3:
        raise ValueError('Require a portable nonempty archive up to2GiB')
    if (archive.stat().st_size+part_bytes-1)//part_bytes>2048:raise ValueError('Too many transfer parts')
    before=digest(archive);directory.mkdir(parents=True,exist_ok=False);parts=[]
    with archive.open('rb') as stream:
        for index,block in enumerate(iter(lambda:stream.read(part_bytes),b''),1):
            name=archive.name+f'.part{index:03d}'
            with (directory/name).open('xb') as output:output.write(block)
            parts.append({'name':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
    if digest(archive)!=before:raise ValueError('Archive changed while splitting')
    manifest={'version':VERSION,'archive':archive.name,'bytes':sum(p['bytes'] for p in parts),'sha256':before,'parts':parts}
    path=directory/(archive.name+'.parts.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    return path


def join(manifest_path,output,expected_sha256):
    manifest_path,output=Path(manifest_path),Path(output)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('version')!=VERSION or not isinstance(expected_sha256,str)
            or not re.fullmatch('[0-9a-f]{64}',expected_sha256) or manifest.get('sha256')!=expected_sha256
            or type(manifest.get('bytes')) is not int or not 0<manifest['bytes']<=2*1024**3):
        raise ValueError('Manifest or explicit trusted archive hash invalid')
    parts=manifest.get('parts')
    if not isinstance(parts,list) or not parts or len(parts)>2048:raise ValueError('Invalid part inventory')
    names=set()
    for part in parts:
        if (not isinstance(part,dict) or set(part)!={'name','bytes','sha256'}
                or not isinstance(part['name'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',part['name'])
                or part['name'] in ('.','..') or part['name'] in names
                or type(part['bytes']) is not int or not 0<part['bytes']<=MAX_PART
                or not isinstance(part['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',part['sha256'])):
            raise ValueError('Unsafe,duplicate or malformed part')
        names.add(part['name'])
        path=manifest_path.parent/part['name']
        if path.is_symlink() or path.stat().st_size!=part['bytes'] or digest(path)!=part['sha256']:
            raise ValueError('Part bytes differ from declared digest')
    if sum(p['bytes'] for p in parts)!=manifest['bytes']:raise ValueError('Total size mismatch')
    temporary=output.with_name(output.name+'.partial')
    if output.exists() or temporary.exists():raise FileExistsError('Preserve existing output or interrupted assembly')
    output.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('xb') as stream:
        for part in parts:
            with (manifest_path.parent/part['name']).open('rb') as source:
                for block in iter(lambda:source.read(1024**2),b''):stream.write(block)
    if temporary.stat().st_size!=manifest['bytes'] or digest(temporary)!=expected_sha256:
        raise ValueError('Reassembled archive differs; partial evidence retained')
    temporary.rename(output)
    return {'bytes':output.stat().st_size,'sha256':expected_sha256,'parts':len(parts),'archive_byte_exact':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('split');s.add_argument('--archive',required=True);s.add_argument('--output',required=True)
    j=sub.add_parser('join');j.add_argument('--manifest',required=True);j.add_argument('--output',required=True);j.add_argument('--sha256',required=True)
    args=parser.parse_args()
    print(split(args.archive,args.output) if args.mode=='split' else json.dumps(join(args.manifest,args.output,args.sha256)))
