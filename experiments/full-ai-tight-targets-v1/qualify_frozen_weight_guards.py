"""Reject stale caller coefficients before running learned point specialization."""
import argparse
from pathlib import Path
from profile_frozen_components import read, write, digest
from frozen_sampler_core_support import prepare
from restored_recurrent_sampler import load_model
from recurrent_sampler_frozen_v7 import library
import recurrent_sampler_native as native
from diagnose_causal_composition import inventory
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh frozen-coefficient guard qualification required')
    reg = read(args.protocol)
    team = prepare(reg)
    model = load_model(reg['pair'], reg['library_path'])
    before = inventory(model)
    native.library = library
    rows = []
    try:
        with torch.inference_mode():
            fields = [torch.zeros(shape, dtype=torch.float32)
                      for shape in native.dimensions((5, 7, 9))]
            for index, name in enumerate(('rows', 'channels', 'offsets', 'weights')):
                changed = [value.clone() for value in model.coefficients['point']]
                if index == 3:
                    value = changed[index].flatten()
                    value[0] = torch.nextafter(value[0], torch.tensor(float('inf')))
                else:
                    changed[index].flatten()[0] += 1
                try:
                    native.transport(fields, (5, 7, 9), changed, model.mid_gate, model.mix, 1 / 60 * 8)
                except ValueError as error:
                    if not str(error).endswith(': 4'):
                        raise
                else:
                    raise ValueError('Specialized kernel accepted changed ' + name)
                rows.append(dict(changed_component=name, rejected_with_status=4))
            if library().rheon_set_execution_threads(3) != 1:
                raise ValueError('Undeclared worker count accepted')
            native.transport(fields, (5, 7, 9), model.coefficients['point'], model.mid_gate, model.mix, 1 / 60 * 8)
        if inventory(model) != before or torch.cuda.is_initialized():
            raise ValueError('Guard qualification changed model/device')
        write(args.output, dict(status='passed_frozen_coefficient_guards', rows=rows,
              cores=reg['cores'], execution_team=team, optimizer_updates=0,
              learned_tensors_unchanged=True, exact_current_coefficients_accepted=True,
              invalid_three_worker_count_rejected=True, all_plan_owners_closed=True,
              protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
