import json
from pathlib import Path
import subprocess
import sys
import unittest

from capture_memory_telemetry_v1 import process_memory


class MemoryTelemetryTests(unittest.TestCase):
    def test_current_executable_has_complete_memory_counters(self):
        row=process_memory()
        self.assertGreater(row['current_rss_bytes'],0)
        self.assertGreaterEqual(row['peak_since_exec_bytes'],row['current_rss_bytes'])

    def test_counter_from_previous_executable_is_not_the_new_program_peak(self):
        here=str(Path(__file__).resolve().parent)
        child="import sys,json;sys.path.insert(0,"+repr(here)+");from capture_memory_telemetry_v1 import process_memory;print(json.dumps(process_memory()))"
        launcher="import os,sys;allocation=bytearray(128*1024**2);os.execv(sys.executable,[sys.executable,'-B','-c',"+repr(child)+"])"
        observed=json.loads(subprocess.check_output([sys.executable,'-B','-c',launcher],text=True))
        self.assertGreaterEqual(observed['getrusage_peak_including_preexec_bytes'],128*1024**2)
        self.assertLess(observed['peak_since_exec_bytes'],64*1024**2)


if __name__=='__main__':unittest.main()
