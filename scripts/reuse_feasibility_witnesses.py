"""Check each discovered geometric witness against every fixed unseen layout.

This second descriptive audit uses no policy outcome for candidate selection.
All successful first-audit spline references form the shared candidate library.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from navstack.world import World
from spline_feasibility_diagnostic import execute, read


def main(batch):
    batch = Path(batch)
    previous = batch / "analysis/posthoc_spline_feasibility"
    output = batch / "analysis/posthoc_shared_witness_library"
    output.mkdir(exist_ok=False)
    marker = read(batch / "formal_v5/H_seed11/final_primary.json")
    source = Path(marker["run"])
    config = EnvConfig.from_dict(read(source / "config.json")["env"])
    scenes = {r["case_id"]: r for line in (source / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
              if (r := json.loads(line))["case"]["scenario"] == "unseen"}
    library = [(r["case_id"],np.load(previous/f"path_case{r['case_id']}.npy"))
               for r in read(previous/"results.json")["results"] if r.get("outcome")=="success"]
    write_json(output / "protocol.json", {"source_candidate_ids":[i for i,_ in library],
        "target_case_ids":list(scenes), "selection":"All successful independent geometry witnesses, never selected by PPO outcomes",
        "ppo_training_steps":0,"total_wall_seconds":120,"translation":"y offset only; identical relative x and yaw"})
    until = min(time.time()+120,read(batch/"batch.json")["deadline_epoch"]-90)
    results=[]
    with gzip.open(output/"physical_trace.jsonl.gz","wt",encoding="utf-8") as stream:
        for case_id, case in scenes.items():
            scene=case["initial_scenario"]
            world=World(config.width,config.height,static=[tuple(r) for r in scene["static"]])
            result={"case_id":case_id,"case":case["case"],"diagnostic_only":True,
                    "status":"no_witness_in_library_inconclusive"}
            for source_id,candidate in library:
                if time.time()>=until:
                    result["status"]="diagnostic_wall_budget"
                    break
                path=candidate.copy()
                path[:,1]+=scene["initial"]["y"]-scenes[source_id]["initial_scenario"]["initial"]["y"]
                if all(world.free(p,config.vehicle,.10) for p in path):
                    np.save(output/f"path_case{case_id}.npy",path)
                    result.update(status="physical_witness_executed",source_case_id=source_id,
                                  **execute(path,scene,config,stream,case_id,until))
                    break
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!="final_state"}),flush=True)
    write_json(output/"results.json",{"scope":"Post-test diagnostic shared geometry library applied to all ten fixed unseen layouts; no policy changes, no exclusions",
        "physical_steps_executed":sum(r.get("physical_steps",0) for r in results),
        "successful_physical_witnesses":sum(r.get("outcome")=="success" for r in results),
        "scheduled_cases":len(scenes),"complete":len(results)==len(scenes),"results":results})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",required=True)
    main(parser.parse_args().batch)
