"""Own-child watchdog: no CARLA or unrelated process termination."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


class ExistingOwnedProcess:
    def __init__(self,pid,batch):
        import ctypes
        self.ctypes=ctypes;self.kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        self.kernel.OpenProcess.restype=ctypes.c_void_p
        self.kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        self.kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        self.kernel.QueryFullProcessImageNameW.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)]
        records=[json.loads(p.read_text(encoding="utf-8")) for p in batch.glob("watchdog_*.json")]
        if not any(r.get("child_pid")==pid for r in records):raise RuntimeError("PID not owned by this batch")
        self.handle=self.kernel.OpenProcess(0x1000|0x100000|1,False,pid)
        if not self.handle:raise RuntimeError("Cannot open owned process")
        buf=ctypes.create_unicode_buffer(32768);size=ctypes.c_ulong(len(buf))
        if not self.kernel.QueryFullProcessImageNameW(self.handle,0,buf,ctypes.byref(size)) or Path(buf.value).resolve()!=Path(sys.executable).resolve():
            raise RuntimeError("Owned process executable identity mismatch")
        self.pid=pid;self.returncode=None

    def poll(self):
        code=self.ctypes.c_ulong()
        if not self.kernel.GetExitCodeProcess(self.handle,self.ctypes.byref(code)):raise RuntimeError("Process status unavailable")
        self.returncode=None if code.value==259 else code.value
        return self.returncode

    def terminate(self):
        subprocess.run(["taskkill","/PID",str(self.pid),"/T","/F"],capture_output=True,
                       creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))

    def wait(self,timeout):
        end=time.time()+timeout
        while self.poll() is None and time.time()<end:time.sleep(.2)
        return self.returncode

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--queue",required=True)
    p.add_argument("--attach-pid",type=int);p.add_argument("--source-root");args=p.parse_args()
    batch=Path(args.batch).resolve();root=Path(__file__).resolve().parents[1]
    deadline=json.loads((batch/"batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
    stamp=str(time.time_ns());out=(batch/("queue_stdout_"+stamp+".log")).open("w");err=(batch/("queue_stderr_"+stamp+".log")).open("w")
    source=Path(args.source_root).resolve() if args.source_root else root
    command=[sys.executable,str(source/"scripts/research_queue.py"),"--batch",str(batch),"--queue",str(Path(args.queue).resolve())]
    child=ExistingOwnedProcess(args.attach_pid,batch) if args.attach_pid else subprocess.Popen(command,cwd=root,stdout=out,stderr=err,
                           creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    record={"watchdog_pid":os.getpid(),"child_pid":child.pid,"command":command,"deadline_epoch":deadline,
            "started_epoch":time.time(),"stdout":out.name,"stderr":err.name}
    (batch/("watchdog_"+stamp+".json")).write_text(json.dumps(record,indent=2))
    last_size=0;last_growth=time.time();stop_at=None;reason=None
    while child.poll() is None:
        time.sleep(5)
        size=Path(out.name).stat().st_size
        if size>last_size:last_size=size;last_growth=time.time()
        if time.time()>deadline-60:reason="six_hour_budget";stop_at=stop_at or time.time()
        # Validation can legitimately have no optimizer updates. Its raw transition file
        # must keep growing; track both streams rather than killing a working evaluator.
        states=[json.loads(p.read_text(encoding="utf-8")) for p in batch.glob("queue_status_*.json")]
        active=[s for s in states if s.get("parent_pid")==child.pid and s.get("run")]
        scope=Path(active[-1]["run"]) if active else batch
        recent=max((p.stat().st_mtime for p in scope.rglob("heartbeat.json")),default=0)
        recent=max(recent,max((p.stat().st_mtime for p in scope.rglob("transitions.jsonl")),default=0))
        if time.time()-max(last_growth,recent)>300:reason="no_training_or_evaluation_progress_300_seconds";stop_at=stop_at or time.time()
        if stop_at:
            (batch/"STOP").write_text(reason)
            if time.time()>=deadline or time.time()-stop_at>60:
                if os.name=="nt":
                    subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],capture_output=True,
                                   creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                else:child.terminate()
                child.wait(timeout=20);break
    record.update({"exit_code":child.returncode,"finished_epoch":time.time(),"reason":reason or "normal_completion"})
    (batch/("watchdog_"+stamp+".json")).write_text(json.dumps(record,indent=2));out.close();err.close()
    sys.exit(0 if child.returncode==0 else 1)
