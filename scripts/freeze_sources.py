"""Immutable experiment source snapshot; does not create a new project or Git repository."""
import argparse,hashlib,json,shutil,time
from pathlib import Path

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("destination");args=p.parse_args()
    root=Path(__file__).resolve().parents[1];dest=Path(args.destination).resolve();dest.mkdir(parents=True,exist_ok=False)
    files=[]
    for name in ("navstack","scripts","configs","tests"):
        for source in (root/name).rglob("*"):
            if source.is_file() and source.suffix in (".py",".json",".ps1") and "__pycache__" not in source.parts:
                target=dest/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);files.append(target)
    for name in ("requirements.txt","requirements.lock.txt"):
        shutil.copy2(root/name,dest/name);files.append(dest/name)
    manifest={"frozen_epoch":time.time(),"root":str(root),"hashes":{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (dest/"frozen_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(dest)
