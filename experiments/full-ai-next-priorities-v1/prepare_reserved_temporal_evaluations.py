"""Register every frozen primary/replica on all eight original reserved scenes."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-final-reserved-evaluation-batch-specification-01.json'
    if output.exists():raise ValueError('Fresh reserved evaluation list required')
    frozen_path=DOCS/'temporal-final-selection-01.json';frozen=read(frozen_path)
    bank_path=BASE/'temporal-reserved-240-01/bank.json';bank=read(bank_path)
    if bank['status']!='completed'or bank['frozen_selection_sha256']!=sha(frozen_path)or bank['reservation_sha256']!=frozen['original_scene_reservation_sha256']:
        raise ValueError('Exact original reserved population captured after actual source/weight freeze required')
    if len(bank['trajectories'])!=8 or any(r['split']!='reserved'or r['states']!=241 for r in bank['trajectories']):
        raise ValueError('All eight original240-step scenes required')
    for name,digest in frozen['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Pre-capture evaluation or inference source changed')
    registry_path=ROOT/frozen['model_registry']
    if sha(registry_path)!=frozen['model_registry_sha256']:raise ValueError('Pre-capture model registry changed')
    methods=read(registry_path)['methods'];commands=[]
    if len(methods)!=10:raise ValueError('All four primaries and six additional seeds required')
    for method in methods:
        identifier=method['identifier'];spec=ROOT/method['specification_path']
        if sha(spec)!=method['specification_sha256']:raise ValueError('Exact pre-capture model specification changed')
        protocol_path=DOCS/(identifier+'-reserved-evaluation-protocol-01.json')
        if protocol_path.exists():raise ValueError('Fresh concrete reserved model protocol required')
        write(protocol_path,dict(status='registered_before_actual_frozen_model_reserved_evaluation',
            frozen_selection_sha256=sha(frozen_path),source_sha256=frozen['source_sha256'],
            execution_assets_sha256=frozen['execution_assets_sha256'],model_specification_sha256=sha(spec),
            bank_sha256=sha(bank_path),every_intermediate_output_required=True,own_history_steps=240,
            original_accuracy_rules_unchanged=True,accuracy_limit=.004,optimizer_updates=0,primary_seed_reselected=False))
        commands.append(dict(id=identifier+'-all-reserved-outputs-and-own240',runtime='torch',timeout=1800,
            arguments=['experiments/full-ai-next-priorities-v1/evaluate_application_policy.py',
                '--specification',str(spec.relative_to(ROOT)),'--protocol',str(protocol_path.relative_to(ROOT)),
                '--bank',str(bank_path.relative_to(ROOT)),
                '--output','runs/full-ai-next-priorities-v1/temporal-reserved-evaluation-01/'+identifier]))
    write(output,dict(commands=commands))
    print('Registered10frozen primary/replica evaluations on all8original reserved scenes; no fitting',flush=True)


if __name__=='__main__':run()
