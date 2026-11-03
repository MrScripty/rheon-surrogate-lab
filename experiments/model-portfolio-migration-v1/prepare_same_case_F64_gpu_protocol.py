"""Preregister a matched original F64 cell baseline, without copying old archives."""
from pathlib import Path
import time

from review_water_learning_sequence import ROOT, binding, canonical, read, require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'same-case-F64-GPU-protocol-01.json'


def prepare():
    require(not OUTPUT.exists(),'Preserve original matched F64 protocol')
    original_path = RUN/'frozen-full-state-GPU-timing-protocol-01.json'
    original = read(original_path)
    cell_path = ROOT/original['original_protocols']['cell']
    cell = read(cell_path)
    require(binding(cell_path)['sha256'] == original['original_protocol_sha256']['cell'],'Original cell protocol changed')
    common_path = RUN/'complete-generator-GPU-protocol-01.json'
    common = read(common_path)
    cases_path = ROOT/common['cases']['path']
    require(binding(cases_path) == common['cases'],'Original common cases changed')
    cases = read(cases_path)
    require(len(cases['cases']) == 64,'All64common cases required')
    for case in cases['cases']:
        scene = case['scene']
        require(not scene['sgs'] and 0 < scene['cells_per_metre'] <= 1024
                and len(scene['counts']) == 3 and all(type(n) is int and 5 <= n <= 256 for n in scene['counts'])
                and 1 <= scene['steps'] <= 600 and 0 <= case['start'] < case['start']+4 <= scene['steps']
                and case['output_dt_s'] == 1/60 and not scene.get('forcing_program') and not scene.get('forcing_schedule'),
                'Common case is outside the unchanged legacy-cell control contract')
    old = original['source_sha256']
    selected = {p for p in old if Path(p).suffix in ('.py','.ptx','.so','.cu','.cuh','.cpp','.hpp')}
    kernel_dirs = {str(Path(p).parent) for p in selected if Path(p).suffix in ('.ptx','.so')}
    prefixes = ['runs/full-ai-next-priorities-v1/b1-fixed-viscosity-pair-01/',
        cell['transport_pair']+'/',cell['shared_pair']+'/',
        'runs/full-ai-tight-targets-v1/mixed-modal-pair-11/']
    selected.update(p for p in old if any(p.startswith(prefix) for prefix in prefixes)
        or str(Path(p).parent) in kernel_dirs)
    # The parent modal constructor reads its geometry manifest, then the expanded
    # constructor replaces that pressure component. Both reads remain exact.
    receipt = read(ROOT/'runs/full-ai-tight-targets-v1/mixed-modal-pair-11/receipt.json')
    assembly_path = ROOT/receipt['source_project']/'assembly.json'
    assembly = read(assembly_path)
    selected.update([str(assembly_path.relative_to(ROOT)),assembly['static_inputs']['manifest']['path']])
    for path in list(selected):
        if Path(path).name != 'receipt.json' or str(Path(path).parent) not in kernel_dirs:
            continue
        value = read(ROOT/path)
        for key in ('source','sampler_parameter_manifest','sampler_codegen_receipt'):
            item = value.get(key)
            if isinstance(item,str) and (ROOT/item).is_file():
                selected.add(item)
    dependencies = {}
    for path in sorted(selected):
        require(path in old,'Selected legacy artifact is absent from original timing custody: '+path)
        spec = binding(ROOT/path)
        require(spec['sha256'] == old[path],'Original F64 source/kernel/asset changed: '+path)
        dependencies[path] = spec
    for spec in common['dependencies']:
        dependencies[spec['path']] = spec
    for path in [original_path,cell_path,common_path,cases_path,
            *[Path(__file__).with_name(n) for n in ('prepare_same_case_F64_gpu_protocol.py',
                'same_case_F64_gpu_benchmark.py','run_same_case_F64_gpu_benchmark.py','reduce_same_case_F64_gpu_benchmark.py')]]:
        dependencies[str(path.relative_to(ROOT))] = binding(path)
    blocks = [dict(K=k,phase=phase,result=f'runs/model-portfolio-migration-v1/same-case-F64-GPU-benchmark-01/K{k}-phase{phase}.json')
        for phase,sequence in [(0,(1,2,3,4)),(1,(4,3,2,1))] for k in sequence]
    contract = dict(status='legacy_cell_source_contract_supported_temporal_fine_clock_unsupported',created_unix=time.time(),
        cases=common['cases'],cell=dict(state='supported_by_source_contract_not_actual_execution',
            cases=64,grid_dimensions=sorted({tuple(c['scene']['counts']) for c in cases['cases']}),
            output_dt_s=1/60,internal_advances_per_output=1),
        temporal=dict(state='unsupported',reason='Original entry requires explicit240/480fine-clock control schedules; all64legacy cases lack reference_substeps,forcing_control_substeps and forcing_schedule. Do not fabricate fine states/schedules.',
            original_cadence_population_retained=True,matched_result_claimed=False),
        evidence=[binding(ROOT/'experiments/full-ai-data-diversity-v1/dynamic_cuda_controls_v1.py'),
            binding(ROOT/'experiments/full-ai-data-diversity-v1/legacy_exact_forcing.py'),
            binding(ROOT/'experiments/full-ai-data-diversity-v1/temporal_frozen_graph_v1.py'),
            binding(ROOT/'experiments/full-ai-data-diversity-v1/temporal_control_packets_v1.py')],
        new_neural_forwards=0,full_portfolio_complete=False)
    require(all(not any(n in c['scene'] for n in ('reference_substeps','forcing_control_substeps','forcing_schedule'))
                for c in cases['cases']),'Temporal unsupported disposition must match actual missing controls')
    contract_path = DOC/'reports/same-case-F64-input-contract-01.json'
    require(not contract_path.exists(),'Preserve original input contract report')
    contract_path.write_bytes(canonical(contract)+b'\n')
    protocol = dict(status='preregistered_not_dispatched',created_unix=time.time(),cases=common['cases'],
        dependencies=sorted(dependencies.values(),key=lambda s:s['path']),process_blocks=blocks,
        constructor_arguments=[cell[n] for n in ('transport_pair','shared_pair','build','control_build')],
        constructor='Unchanged CellCoordinatePressureNativeEntry; original default N8coordinate kernel build retained; N1branch used here.',
        worker='experiments/model-portfolio-migration-v1/same_case_F64_gpu_benchmark.py',
        output='runs/model-portfolio-migration-v1/same-case-F64-GPU-benchmark-01',
        runtime_manifest=common['runtime_manifest'],host_cpu=4,warmups=3,samples=10,
        worker_deadline_seconds=900,max_CUDA_allocated_bytes=4*1024**3,
        expected_primary_calls=5120,expected_primary_physical_outputs=12800,expected_cases=512,
        original_reference_binary_sha256=cases['reference_native_probe_binary_sha256'],
        timing=common['timing'],accuracy='Same frozen per-field/per-lead pressure,velocity,tracer and equation score as all35generators.',
        source_selection='All original Python source and compiled kernel code; exact managed source/transport/pressure pairs,parent geometry manifest and kernel receipts. Original unrelated evaluation archives excluded; actual imported project modules must be preregistered.',
        bound_dependency_bytes=sum((ROOT/s['path']).stat().st_size for s in dependencies.values()),
        dispatch_gate='Original70generator workers terminal AND independent all35reduction passed; no simultaneous native writes/browser/GPU owner.',
        reference_scope='Old coarse1/60native references,16consumed fitting scenes,fourstarts. This is new baseline evaluation,not its original case population or independent holdout.',
        physical_work='Sequential original N1stage per physical output; Koutputs are not learned joint generation.',
        unsupported_temporal=contract['temporal'],input_contract=binding(contract_path),
        CPU_neural_forwards=0,new_reference_generation=False,output_arrays_written=False,native_mutations=0,
        full_portfolio_complete=False,producer=binding(Path(__file__)))
    OUTPUT.write_bytes(canonical(protocol)+b'\n')
    print('Matched F64 K1–4/8freshworkers prepared;',protocol['bound_dependency_bytes'],'bound dependency bytes;no GPU dispatch.')


if __name__ == '__main__':
    prepare()
