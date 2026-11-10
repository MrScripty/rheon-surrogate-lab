import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from surrogate.bench import __main__ as cli
from surrogate.bench import results
from surrogate.bench.subjects import Persistence
from surrogate.tests.test_packet import document


def call(*arguments):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        code = cli.main(list(arguments))
    return code, output.getvalue()


class CommandLine(unittest.TestCase):
    def test_lists_every_suite_with_its_population(self):
        code, text = call('suites')
        self.assertEqual(code, 0)
        self.assertIn('dense-development-240', text)
        self.assertIn('(consumed)', text)

    def test_writes_packets_once_and_refuses_to_overwrite_them(self):
        with tempfile.TemporaryDirectory() as directory:
            result = results.write(document(), Path(directory) / 'result.json')
            code, text = call('packets', str(result), '--output-directory', str(Path(directory) / 'out'))
            written = Path(text.strip())
            self.assertEqual((code, written.name), (0, 'unit-suite.measurements.json'))
            self.assertEqual(json.loads(written.read_bytes())['format'], 'retained_measurements_v1')
            with self.assertRaises(SystemExit):
                call('packets', str(result), '--output-directory', str(Path(directory) / 'out'))

    def test_run_refuses_an_existing_output(self):
        with tempfile.NamedTemporaryFile(suffix='.json') as existing, self.assertRaises(SystemExit):
            call('run', '--suite', 'dense-development-240', '--subject', 'persistence', '--output', existing.name)

    def test_subject_specifications(self):
        self.assertIsInstance(cli.subject_from('persistence', 'cpu'), Persistence)
        for wrong in ('persistence:3', 'checkpoint', 'unknown'):
            with self.assertRaisesRegex(ValueError, 'Unknown subject'):
                cli.subject_from(wrong, 'cpu')


if __name__ == '__main__':
    unittest.main()
