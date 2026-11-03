"""Bind actual native domain-change references before four-model qualification."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'application-domain-change-qualification-protocol-01.json'
    if output.exists():raise ValueError('Fresh actual supported-boundary model protocol required')
    original_path=DOCS/'application-domain-change-reference-protocol-01.json';original=read(original_path)
    for name,digest in original['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered supported-boundary source changed')
    bank=BASE/'application-domain-change-reference-01/bank.json';value=read(bank)
    if value['status']!='completed'or len(value['trajectories'])!=8 or value['states_per_scene']!=49:
        raise ValueError('All eight48-step native event references required')
    if value['protocol_sha256']!=sha(original_path):raise ValueError('Native references belong to another protocol')
    registry=BASE/'temporal-finalists-01/application-models.json'
    if sha(registry)!=original['registry_sha256']:raise ValueError('Actual primary selection changed')
    for row in value['trajectories']:
        receipt=ROOT/row['receipt'];record=read(receipt)
        if sha(receipt)!=row['receipt_sha256']or not record['first14states_and_RHS_bitwise_preserved']or not record['event_changes_next_step_physics']:
            raise ValueError('Original prefix and actual physical event effect required')
        if len(record['native_pressure_request_geometry_checks'])!=48:
            raise ValueError('Every actual native request must use the correct geometry')
    write(output,dict(status='registered_before_actual_model_supported_boundary_qualification',
        source_sha256=original['source_sha256'],execution_assets_sha256=original['execution_assets_sha256'],
        bank_sha256=sha(bank),registry_sha256=sha(registry),development_bank_sha256=original['bank_sha256'],
        scene_identifiers=original['scene_identifiers'],actual_primary_horizons=[1,2,3,4],
        event_step=13,steps=48,qualification_cases=32,optimizer_updates=0,
        candidate_numerical_pressure_solve=False,performance_claim=False,application_acceptance_limits=False))
    print('Registered32actual primary-model supported-boundary episodes with piecewise native references',flush=True)


if __name__=='__main__':run()
