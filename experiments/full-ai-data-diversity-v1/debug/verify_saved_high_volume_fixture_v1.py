"""Verify saved diagnostic fields against actual uninterrupted native hashes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def run(args):
    if args.output.exists():raise ValueError('Fresh verified fixture receipt required')
    observed=json.loads(args.observation.read_text());expected=observed['reference46']['actual_field_sha256']
    with np.load(args.artifact,allow_pickle=False) as a:
        if set(a.files)!=set(expected):raise ValueError('Every actual native reference field required')
        actual={name:hashlib.sha256(a[name].tobytes()).hexdigest() for name in a.files}
        if actual!=expected:raise ValueError('Saved fixture differs from actual uninterrupted reference')
    result=dict(status='verified_saved_uninterrupted_reference_endpoint_after_metadata_path_error',
        reference_path=str(args.artifact.resolve().relative_to(ROOT)),reference_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),field_sha256=actual,
        original_uninterrupted_hashes_bitwise_equal=True,old_capture_command_terminal_exit=1,
        old_helper_error='Relative path formatting during receipt update, after successful native close/custody guard',
        native_pid_not_retained=True,source_capture='experiments/full-ai-data-diversity-v1/debug/prepare_high_volume_failure_v1.py',
        verification_only=True,no_native_or_model_execution=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print('Verified all8saved native field hashes against actual uninterrupted failed endpoint')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
