"""Capture one finite managed command list from the actual selected seed registry."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-training-batch-specification-01.json'
    if output.exists():raise ValueError('Fresh actual replica fit batch required')
    registry_path=BASE/'fixed-viscosity-replication-inputs-01/models.json';registry=read(registry_path)
    registration=read(DOCS/'fixed-viscosity-replication-fit-registration-01.json')
    if sha(registry_path)!=registration['model_registry_sha256']or len(registry['models'])!=6:
        raise ValueError('Exactly two added seeds for two bundles and B1required')
    for name,digest in registration['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified replica implementation changed')
    def c(identifier,script,*arguments,timeout=120,runtime='control'):
        return dict(id=identifier,runtime=runtime,
            arguments=['experiments/full-ai-next-priorities-v1/'+script,*arguments],timeout=timeout)
    commands=[c('authorized-owner-ready','owner_ready.py')]
    for model in registry['models']:
        identifier=model['identifier']
        commands.append(c(identifier+'-capture','prepare_fixed_viscosity_replica_project.py',
            '--identifier',identifier,'--horizons',str(model['horizons'])))
    for model in registry['models']:
        identifier=model['identifier'];pair='runs/full-ai-next-priorities-v1/'+identifier+'-pair-01'
        commands.append(c(identifier+'-authorized-owner-ready','owner_ready.py'))
        commands.append(c(identifier+'-managed-fit-and-exact-CPU-restore','run_fixed_viscosity_replica_pair.py',
            '--project','runs/full-ai-next-priorities-v1/'+identifier+'-project-01','--output',pair,timeout=3100))
        commands.append(c(identifier+'-every-output-and-own240','evaluate_fixed_viscosity_replica.py',
            '--pair',pair,'--bank','runs/full-ai-next-priorities-v1/temporal-development-240-01/bank.json',
            '--output','runs/full-ai-next-priorities-v1/fixed-viscosity-replication-evaluation-01/'+identifier,
            timeout=1800,runtime='torch'))
    commands.append(c('audit-six-managed-replicas-and-accuracy','report_fixed_viscosity_replicas.py',runtime='torch'))
    write(output,dict(commands=commands))
    write(DOCS/'fixed-viscosity-replication-batch-registration-01.json',dict(status='registered_before_replica_fits',
        specification_sha256=sha(output),registry_sha256=sha(registry_path),
        fit_registration_sha256=sha(DOCS/'fixed-viscosity-replication-fit-registration-01.json'),
        source_sha256=sha(__file__),commands=len(commands),managed_fits=6,exact_CPU_restores=6,
        training_device='cuda',restore_and_evaluation_device='cpu',automatic_retry=False,
        new_physical_target_states=6*36000,reserved_scenes_consumed=False,original_goal_complete=False))
    print('Captured concrete finite batch:6managed CUDA fits,6exact CPU restores and6own240 evaluations',flush=True)


if __name__=='__main__':run()
