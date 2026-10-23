"""Windows guard terminates its retained process handle, with identity checks."""
import ctypes
from ctypes import wintypes
import os
import subprocess
import sys
import pytest
from navstack.owned_process import OwnedProcess


@pytest.mark.skipif(os.name!='nt',reason='Windows retained-handle ownership')
def test_owned_handle_rejects_wrong_identity_and_terminates_child():
    executable=getattr(sys,'_base_executable',sys.executable)
    child=subprocess.Popen([executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        times=[wintypes.FILETIME() for _ in range(4)]
        assert kernel.GetProcessTimes(int(child._handle),*[ctypes.byref(t) for t in times])
        creation=((times[0].dwHighDateTime<<32)+times[0].dwLowDateTime)/10**7-11644473600
        with pytest.raises(RuntimeError,match='creation time mismatch'):
            OwnedProcess(child.pid,creation+1,executable,allow_terminate=True)
        readonly=OwnedProcess(child.pid,creation,executable)
        with pytest.raises(RuntimeError,match='read-only'):readonly.terminate()
        owned=OwnedProcess(child.pid,creation,executable,allow_terminate=True)
        assert owned.poll() is None
        owned.terminate()
        assert owned.wait()==1 and child.wait(timeout=10)==1
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
