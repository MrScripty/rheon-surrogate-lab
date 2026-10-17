"""Exercise the actual CLI's selected current-bank argument without admission."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch
import assemble


class AssemblyCLI(unittest.TestCase):
    def test_cli_forwards_optional_current_bank(self):
        args=[item for name in ('root','template','declarations','probe','tuldok','runtime','configuration','output')for item in ('--'+name,'selected-'+name)]
        for selection,expected in (([],None),(['--current-bank','native-bank'],Path('native-bank'))):
            with patch('sys.argv',['assemble.py',*args,*selection]),patch.object(assemble,'assemble',return_value={'project':{},'updates':5000})as run,redirect_stdout(StringIO()):
                assemble.main()
            self.assertEqual(run.call_args.args[-1],expected)

if __name__=='__main__':unittest.main()
