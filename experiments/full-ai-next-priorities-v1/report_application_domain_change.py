"""Publish actual fresh-prediction boundary behavior and its separate error diagnostics."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'application-domain-change-results.json'
    if output.exists():raise ValueError('Fresh supported-boundary result required')
    path=BASE/'application-domain-change-qualification-01/receipt.json';value=read(path)
    if value['status']!='passed'or len(value['rows'])!=32 or not value['all_supported_boundary_changes_trigger_fresh_predictions']:
        raise ValueError('All four actual trained horizons and all eight native scenes required')
    sources=[path];rows=[]
    for relative,digest in value['source_evidence'].items():
        p=ROOT/relative
        if sha(p)!=digest:raise ValueError('Supported-boundary qualification evidence changed')
        sources.append(p)
    for name,digest in value['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Supported-boundary source changed')
    for row in value['rows']:
        p=ROOT/row['receipt'];record=read(p)
        if sha(p)!=row['receipt_sha256']or len(record['diagnostics'])!=48 or record['physical_steps']!=48:
            raise ValueError('Every physical output must be consumed and scored')
        if not record['fresh_prediction_after_supported_boundary_change']or record['native_numerical_steps_executed']or record['fallback']:
            raise ValueError('Actual fresh predictions without numerical correction required')
        if not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in record['native_processes']):
            raise ValueError('Every native child must complete/reap')
        rows.append(dict(row,worst_error=record['worst_error']))
    write(output,dict(status='completed',rows=rows,source_evidence={str(p.relative_to(ROOT)):sha(p)for p in sources},
        reporter_sha256=sha(__file__),actual_horizons=[1,2,3,4],scenes=8,physical_states_scored=32*48,
        supported_domain_boundary_change_regenerates_predictions=True,
        cells_per_metre_changes=True,fluid_mass_density_changes=False,logical_cell_counts_unchanged=True,
        closed_boundary_type_unchanged=True,external_volume_change_is_declared=True,
        arbitrary_new_boundary_type_accuracy_claim=False,performance_claim=False,
        application_acceptance_limits=False,original_primary_reselected=False,original_goal_complete=False))
    lines=['# Supported physical-domain boundary change','',
        'After accepting step13, change the physical box size by switching8↔16cells per metre at fixed logical '
        'cell counts. The controller copies the current arrays, discards unused predictions and calls the same '
        'frozen model with the changed geometry. Fluid mass density and the closed boundary type remain unchanged. '
        'Every output through48 is evaluated using its matching native geometry; the externally imposed volume '
        'and mass mapping is recorded. Unsupported boundary types retain the separate explicit stop guard.','',
        '| Model | Scene | Cells/metre before→after | Discarded at event | Fresh continuation | Worst diagnostic error | Accuracy diagnostic pass |',
        '| --- | --- | --- | ---: | --- | ---: | --- |']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["identifier"]} | {row["cells_per_metre_before"]}→{row["cells_per_metre_after"]} | {row["event_discarded_states"]} | {row["fresh_prediction_at_step13"]} | {row["worst_error"]:.7g} | {row["accuracy_pass"]} |')
    lines.extend(['','This verifies supported-boundary invalidation and regeneration on actual independently trained '
        'horizons. It is a functional/accuracy diagnostic, with no new CPU speed, arbitrary-boundary generalization '
        'or application-acceptance claim.','', '[Actual receipts, all fields and source evidence](application-domain-change-results.json)',''])
    (DOCS/'application-domain-change-results.md').write_text('\n'.join(lines))
    print('Published32actual supported-domain-change episodes and separate accuracy diagnostics',flush=True)


if __name__=='__main__':run()
