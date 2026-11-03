"""Small stdlib metadata boundaries for original component checkpoints."""
import hashlib
import json
from pathlib import Path


def require(value,message):
    if not value:
        raise ValueError(message)


def observed_file(root,path,expected=None):
    root = root.resolve()
    path = Path(path)
    require(not path.is_absolute() and '..' not in path.parts,'Component material must remain inside its bound root')
    full = root/path
    require(full.is_file() and not full.is_symlink() and full.resolve().is_relative_to(root),
            'Ordinary in-root component material required')
    before = full.stat()
    with full.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    after = full.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Material changed while hashing')
    result = dict(path=str(path),sha256=digest,bytes=after.st_size)
    if expected is not None:
        require(digest == expected['sha256'] and ('bytes' not in expected or after.st_size == expected['bytes']),
                'Original component material bytes differ: '+str(path))
    return result


def selected_inventory(receipt,history,weights_sha256):
    """Support the actual three historical inventory owners,not fake restores."""
    if 'final_inventory' in receipt:
        inventory = receipt['final_inventory']
        require(inventory == history.get('final_inventory'),'Receipt/history final inventories differ')
        authority = 'receipt/final_inventory + history/final_inventory'
    elif 'checkpoint_inventories' in receipt:
        inventory = receipt['checkpoint_inventories']['weights.npz']
        require(inventory == history['checkpoint_inventories']['weights.npz'],'Selected guided checkpoint inventory differs')
        authority = 'receipt/checkpoint_inventories/weights.npz + history/checkpoint_inventories/weights.npz'
    else:
        selected = [row['inventory'] for row in history.get('checkpoints',{}).values() if row.get('sha256') == weights_sha256]
        require(selected and all(value == selected[0] for value in selected),'Original selected checkpoint inventory unavailable or ambiguous')
        inventory,authority = selected[0],'history/checkpoints with exact selected archive SHA'
    require(isinstance(inventory,dict) and inventory,'Complete selected component tensor inventory required')
    for spec in inventory.values():
        require(isinstance(spec,dict) and spec.get('dtype') in ('<f4','<f8')
                and isinstance(spec.get('shape'),list) and all(type(v) is int and v > 0 for v in spec['shape'])
                and isinstance(spec.get('sha256'),str) and len(spec['sha256']) == 64,
                'Typed original F32/F64 component inventory required')
    return inventory,authority


def current_publication(root):
    """Scheduling boundary for new work;prior frozen workers stay unchanged."""
    report = root/'docs/plans/model-portfolio-migration-v1/reports/native-pressure-publication-UI-01.json'
    proof = json.loads(report.read_bytes())
    header_path = root/'projects/rheon-model-portfolio-unified-04/.tuldok/project.json'
    raw = header_path.read_bytes()
    header = json.loads(raw)
    require(proof['status'] == 'all150pressure_comparison_and23corrected_outcome_notes_actual_native_UI_accepted'
            and proof['guide_revision'] == 9 and proof['native_measurement_sources'] == 31 and proof['I57_resolved']
            and proof['both_browsers_stopped'] and proof['requested_chart_comparisons'] == 13
            and proof['displayed_numeric_presence_assertions'] == 1092
            and header['checkpoint'] == proof['checkpoint']
            and hashlib.sha256(raw).hexdigest() == proof['canonical_header_sha256'],
            'Current accepted pressure publication required before new component work')
    return report,header
