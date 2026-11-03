"""Correct composed-record selection; preserve the original failed tool bytes."""
import argparse

import adopt_matched_generator_outcome_notes as original

def target_records(snapshot):
    result = {}
    for entry in snapshot['inventory']:
        if entry['family'] != 'project_file' or entry['namespace'] != snapshot['project_id']:
            continue
        body = (original.PROJECT/'.tuldok'/entry['path']).read_bytes()
        original.require(len(body) == entry['bytes'] and original.digest(body) == entry['sha256'],
            'Current canonical note identity differs')
        doc = __import__('json').loads(body)
        # Composition keeps the source documents and their owner identities.
        # The inventory namespace describes the container, not file ownership.
        if doc['project_id'] != snapshot['project_id']:
            continue
        original.require(doc['path'] not in result,'Current note ownership/path differs')
        result[doc['path']] = entry,doc
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','adopt','after'))
    args = parser.parse_args()
    original.target_records = target_records
    {'before':original.before,'adopt':original.adopt,'after':original.after}[args.phase]()
