import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize("explicit_queue", [True, False])
def test_only_ready_models_are_evaluated_before_deadline(tmp_path,monkeypatch,explicit_queue):
    from scripts import final_evaluation_queue as queue
    batch=tmp_path/"batch";batch.mkdir();deadline=10100
    (batch/"batch.json").write_text(json.dumps({"deadline_epoch":deadline}),encoding="utf-8")
    jobs=[{"version":"supplementary_v4","name":n,"budget":1000000} for n in ("H_shared_seed11","B1_shared_seed11")]
    file=tmp_path/"jobs.json";file.write_text(json.dumps(jobs),encoding="utf-8")
    ready=batch/("supplementary_v4/B1_shared_seed11" if explicit_queue else "formal_v5/B1_seed11");ready.mkdir(parents=True)
    (ready/"validation_1000000.json").write_text("{}",encoding="utf-8")
    (ready/"latest.json").write_text(json.dumps({"checkpoint":str(ready/"checkpoint")}),encoding="utf-8")
    clock={"now":10000,"sleeps":0};calls=[]
    def sleep(seconds):
        clock["sleeps"]+=1;clock["now"]+=1000 if clock["sleeps"]>=3 else 1
    class Process:
        pid=123
        returncode=0
        def __init__(self,command,**kwargs):
            checkpoint=Path(command[command.index("--checkpoint")+1]);label=command[command.index("--label")+1]
            calls.append((checkpoint.parent.name,label))
            (checkpoint.parent/("final_"+label+".json")).write_text(json.dumps({"complete":True}),encoding="utf-8")
        def poll(self):return 0
    monkeypatch.setattr(queue.time,"time",lambda:clock["now"]);monkeypatch.setattr(queue.time,"sleep",sleep)
    monkeypatch.setattr(queue,"resources_ok",lambda path:(True,None));monkeypatch.setattr(queue.subprocess,"Popen",Process)
    queue.main(SimpleNamespace(batch=str(batch),queue=str(file) if explicit_queue else None,source_root=str(tmp_path),version="formal_v5",workers=1))
    assert calls==[(ready.name,"primary"),(ready.name,"generalization")]
    summary=json.loads((batch/("final_evaluation_queue_jobs" if explicit_queue else "final_evaluation_queue_v5")/"summary.json").read_text(encoding="utf-8"))
    assert summary["planned_jobs"]==(4 if explicit_queue else 20) and len(summary["completed"])==2 and not summary["complete"]
