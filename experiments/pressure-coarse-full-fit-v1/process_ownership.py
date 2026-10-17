"""Linux phase ownership: adopt orphaned descendants and prove drainage by ECHILD.

A CPU phase may create workers in separate sessions. Killing a process group or
waiting for its leader cannot establish cessation. This CLI owner becomes their
subreaper before launch; every phase must reach ECHILD before the next starts.
"""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import time


def become_subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0):raise OSError(ctypes.get_errno(),'CPU phase subreaper unavailable')


def children():
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:continue
        if int(fields[1])==os.getpid():result.append(int(path.name))
    return result


def _drain_adopted(grace_seconds):
    started=time.monotonic();reaped=[];signalled=set()
    while True:
        try:
            pid,status=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:
            return dict(echild=True,reaped=reaped,signals_sent=len(signalled),elapsed_s=time.monotonic()-started)
        if pid:
            reaped.append(dict(pid=pid,returncode=os.waitstatus_to_exitcode(status)));continue
        elapsed=time.monotonic()-started
        if elapsed>=grace_seconds:
            number=signal.SIGTERM if elapsed<grace_seconds+3 else signal.SIGKILL
            for pid in children():
                key=(pid,int(number))
                if key in signalled:continue
                try:
                    fd=os.pidfd_open(pid)
                    try:signal.pidfd_send_signal(fd,number)
                    finally:os.close(fd)
                except ProcessLookupError:continue
                signalled.add(key)
        # Retain ownership until the kernel establishes ECHILD; never drop a live
        # descendant merely because a deadline or owner process has elapsed.
        time.sleep(.05)


def drain_adopted(grace_seconds=1.):
    # Cancellation is delivered only after custody has reached kernel ECHILD.
    previous={number:signal.getsignal(number)for number in(signal.SIGTERM,signal.SIGINT)}
    pending=[]
    def defer(number,_frame):pending.append(number)
    for number,handler in previous.items():
        if handler!=signal.SIG_IGN:signal.signal(number,defer)
    try:proof=_drain_adopted(grace_seconds)
    finally:
        for number,handler in previous.items():signal.signal(number,handler)
    if pending:raise KeyboardInterrupt('CPU drainage completed before cancellation')
    return proof


def cancel_and_drain(child,cooperative_seconds=30.):
    previous={number:signal.getsignal(number)for number in(signal.SIGTERM,signal.SIGINT)}
    for number in previous:signal.signal(number,signal.SIG_IGN)
    forced=False
    try:
        if child.poll()is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=cooperative_seconds)
            except subprocess.TimeoutExpired:
                # Descendants remain owned through automatic subreaper adoption.
                forced=True;child.kill();child.wait()
        proof=drain_adopted()
        return dict(leader_returncode=child.returncode,leader_escalated=forced,**proof)
    finally:
        for number,handler in previous.items():signal.signal(number,handler)
