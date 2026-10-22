"""Crash-released exclusive lease, compatible with the CARLA Python 3.7 bridge.

The inode is never removed: unlinking a locked file permits a second owner on
POSIX. PID text is diagnostic; the kernel lock is the authority for new owners.
Legacy live PID files are respected even though those owners lack a kernel lock.
"""
import os
import time
from pathlib import Path


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:
                return False
            return True  # access denied/unknown: fail closed
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE; retained handles can outlive exit
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ProcessLease:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def acquire(self):
        if self.fd is not None:
            raise RuntimeError("Lease already acquired")
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            previous = os.read(fd, 256).decode("ascii").strip()
            if previous:
                if not previous.isdigit() or int(previous) <= 0:
                    raise RuntimeError("Unrecognized owner metadata; preserve for inspection")
                if pid_alive(int(previous)):
                    raise RuntimeError("Live legacy lease owner: " + previous)
                # Preserve evidence before replacing a dead owner's PID. A bridge
                # must still verify the world is empty before acquiring tick control.
                archive = self.path.with_name(self.path.name + ".stale." + str(time.time_ns()))
                archive.write_text(previous, encoding="ascii")
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.fsync(fd)
            self.fd = fd
            return self
        except BaseException:
            if locked:
                self._unlock(fd)
            os.close(fd)
            raise

    @staticmethod
    def _unlock(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)

    def close(self):
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.fsync(fd)
        finally:
            try:
                self._unlock(fd)
            finally:
                os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *unused):
        self.close()
