"""Render an offline view from the retained scalar diagnosis."""
import argparse
import json
from pathlib import Path


def render(directory):
    result = json.loads((directory/'results.json').read_text())
    template = (directory/'view.html').read_text()
    if result['status'] != 'completed' or template.count('__STUDY_DATA__') != 1:
        raise ValueError('Completed evidence and one explicit data slot required')
    data = json.dumps(result, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (directory/'index.html').write_text(template.replace('__STUDY_DATA__', data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    render(parser.parse_args().directory.resolve())
