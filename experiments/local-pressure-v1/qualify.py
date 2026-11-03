"""Run the unchanged GN arithmetic contracts and new portability contracts."""
import sys
import unittest

from contract import HERE, LAB


def main():
    # Separate loaders avoid the historical train.py and the configured train.py
    # sharing a discovery namespace. Run the historical numerical suite in a
    # separate interpreter; it constructs no optimizer and executes no training.
    import subprocess
    subprocess.run([sys.executable, "-c", "import unittest; s=unittest.defaultTestLoader.discover('experiments/global-pressure-v1/groupnorm-v1',pattern='test_*.py'); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(0 if r.wasSuccessful() and r.testsRun and not r.skipped else 1)"], cwd=LAB, check=True, timeout=120)
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
