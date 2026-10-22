"""Adopt an already running Windows child without relying on PID alone."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import time


class OwnedProcess:
    def __init__(self, pid, creation_epoch, executable):
        if os.name != "nt":
            raise RuntimeError("Windows process adoption only")
        self.pid=int(pid);self.returncode=None
        self.kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        self.kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        self.kernel.OpenProcess.restype=wintypes.HANDLE
        self.kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        self.kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.handle=self.kernel.OpenProcess(0x1000|0x100000,False,self.pid)
        if not self.handle:raise RuntimeError("Owned process handle unavailable")
        try:
            times=[wintypes.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(self.handle,*[ctypes.byref(t) for t in times]):
                raise RuntimeError("Cannot verify process creation time")
            actual=((times[0].dwHighDateTime<<32)+times[0].dwLowDateTime)/10**7-11644473600
            if abs(actual-creation_epoch)>.02:raise RuntimeError("PID creation time mismatch")
            size=wintypes.DWORD(32768);buffer=ctypes.create_unicode_buffer(size.value)
            if not self.kernel.QueryFullProcessImageNameW(self.handle,0,buffer,ctypes.byref(size)):
                raise RuntimeError("Cannot verify executable")
            if Path(buffer.value).resolve()!=Path(executable).resolve():raise RuntimeError("Executable mismatch")
        except BaseException:
            self.kernel.CloseHandle(self.handle);raise

    def poll(self):
        code=wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle,ctypes.byref(code)):raise RuntimeError("Process query failed")
        self.returncode=None if code.value==259 else code.value
        return self.returncode

    def wait(self,timeout=20):
        until=time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic()>until:raise TimeoutError("Owned process still running")
            time.sleep(.1)
        return self.returncode
