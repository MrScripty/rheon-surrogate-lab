"""Remove repeated scene metadata;prove exact reconstruction before retirement."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import observed_file


def run():
    old_path=RUN/'guided-component-contract-01.json'
    new_path=RUN/'guided-component-contract-02.json'
    report_path=DOC/'reports/guided-contract-compaction-01.json'
    require(not new_path.exists() and not report_path.exists(), 'Preserve original compact metadata disposition')
    protocol=read(RUN/'guided-original-GPU-probe-protocol-01.json')
    old_spec=observed_file(ROOT,old_path.relative_to(ROOT),protocol['contract'])
    for fid in protocol['fit_ids']:
        probe=read(ROOT/protocol['output_root']/fid/'result.json')
        require(probe['status']=='original_guided_selected_GPU_probe_replayed'
                and probe['original_selected_GPU_probe_bitwise_equal'] and probe['cleanup_complete']
                and probe['temporary_source_view_removed'] and not Path('/proc',str(probe['pid'])).exists(),
                'Both original probe owners must be terminal before changing their metadata source')
    original=read(old_path)
    with zipfile.ZipFile(ROOT/original['data_archive']['path']) as archive:
        manifest=json.loads(archive.read('manifest.json'))
    require(hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            ==original['data_manifest_semantic_sha256'], 'Original native semantic authority required')
    for case in original['cases']:
        require(case.pop('scene')==manifest['masters'][case['identifier']]['scene'],
                'Every removed scene must equal its existing authoritative data definition')
    require(len(original['cases'])==2880 and sum(c['K'] for c in original['cases'])==7200,
            'Every original physical case must remain')
    compact=dict(original,case_scene_binding='data_archive/manifest.json:masters[identifier].scene',
        scene_definitions_copied=False,retired_expanded_contract=old_spec)
    # Reconstruct original logical metadata using references to the existing
    # scene definitions. Stream its canonical encoding into a hash, never a
    # second 2.4GB file or whole encoded string.
    reconstructed=dict(original,cases=[dict(c,scene=manifest['masters'][c['identifier']]['scene']) for c in original['cases']])
    digest=hashlib.sha256();size=0
    for chunk in json.JSONEncoder(indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False).iterencode(reconstructed):
        raw=chunk.encode();digest.update(raw);size+=len(raw)
    digest.update(b'\n');size+=1
    require(digest.hexdigest()==old_spec['sha256'] and size==old_spec['bytes'],
            'Original exact bytes must reconstruct before retiring repeated metadata')
    save(new_path,compact)
    new_spec=observed_file(ROOT,new_path.relative_to(ROOT))
    require(new_spec['bytes']<32*1024**2, 'Compact contract must remain metadata-sized')
    report=dict(status='expanded_guided_scene_metadata_losslessly_compacted_and_retired',created_unix=time.time(),
        producer=binding(Path(__file__)),previous_expanded_contract=old_spec,compact_contract=new_spec,
        exact_old_byte_reconstruction_sha256=digest.hexdigest(),exact_old_byte_reconstruction_bytes=size,
        original_native_scene_authority=compact['data_archive'],all2880cases_and7200outputs_preserved=True,
        scene_definitions_in_compact_contract=0,original_source_and_weight_changes=False,
        scientific_payload_copies=0,native_mutations=0,new_model_forwards=0,
        bytes_removed=old_spec['bytes']-new_spec['bytes'],full_portfolio_complete=False,
        reconstruction='Load compact contract;remove case_scene_binding,scene_definitions_copied,retired_expanded_contract;restore each cases[].scene from existing data_archive/manifest.json masters[identifier].scene;canonical indent2/sort_keys/ensure_asciiFalse/allow_nanFalse encoding plus newline gives exact retired digest.')
    save(report_path,report)
    save(old_path,dict(status='retired_repeated_scene_metadata_current_authority_is_compact02',
        previous_expanded_contract=old_spec,current_contract=new_spec,
        lossless_reconstruction_and_disposition=binding(report_path)))
    print('Lossless compact metadata:',old_spec['bytes'],'→',new_spec['bytes'],'bytes;original repeated controls retired.',flush=True)


if __name__=='__main__':run()
