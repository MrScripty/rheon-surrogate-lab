"""Render the offline comparison directly from the one retained row receipt."""
import argparse
import json
from pathlib import Path


def render(directory):
    result = json.loads((directory/'comparison/results.json').read_text())
    template = (directory/'view.html').read_text()
    if template.count('__STUDY_DATA__') != 1 or result['status'] != 'completed':
        raise ValueError('One data slot and completed evidence required')
    compact = json.dumps(result, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (directory/'index.html').write_text(template.replace('__STUDY_DATA__', compact))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    render(parser.parse_args().directory.resolve())
