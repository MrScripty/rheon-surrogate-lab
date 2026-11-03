"""Linux memory measurements scoped to the current executable image.

getrusage's maximum can survive execve and include the preceding executable.
Keep it for diagnosis; VmHWM measures this executing program's address space.
"""
from pathlib import Path
import resource


def process_memory():
    fields={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmHWM','VmRSS'):
            count,unit=value.split()
            if unit!='kB':raise ValueError('Unsupported Linux memory unit')
            fields[key]=int(count)*1024
    if set(fields)!={'VmHWM','VmRSS'} or fields['VmHWM']<fields['VmRSS']:
        raise ValueError('Incomplete process memory telemetry')
    return dict(peak_since_exec_bytes=fields['VmHWM'],current_rss_bytes=fields['VmRSS'],
        getrusage_peak_including_preexec_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
