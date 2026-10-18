"""Render a self-contained offline view without changing historical chart data."""
import argparse
import json
from pathlib import Path

from common import read


def render(results, template, output):
    data = read(results)
    if data['schema'] != 'rheon-four-pass-results-v1' or data['status'] != 'completed':
        raise ValueError('Completed measured study required')
    page = template.read_text()
    if page.count('__STUDY_DATA__') != 1:
        raise ValueError('Exact study-data seam required')
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    output.write_text(page.replace('__STUDY_DATA__', payload))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['results', 'template', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.template, args.output)
