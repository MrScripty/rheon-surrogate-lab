"""One source-bound N8 sampler coordinate re-expression; no learned forward."""
import argparse
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(protocol):
    reg = json.loads(protocol.read_text())
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen coordinate-generation dependency changed: '+path)
    source = ROOT/reg['template_source']
    output = ROOT/reg['output_source']
    receipt = ROOT/reg['output_receipt']
    if output.exists() or receipt.exists():
        raise ValueError('Fresh source and receipt required')
    text = source.read_text()
    # Whitespace-independent matching is deliberately avoided: these exact
    # expressions bind this transformation to the reviewed parent source.
    substitutions = [
        ('double upper[3]={double(nx)*h,double(ny)*h,double(nz)*h};',
         'const double alpha=dt/h;\n double upper[3]={double(nx),double(ny),double(nz)};'),
        ('position[a]=(base[a]+(kind==a?0.:.5))*h;',
         'position[a]=base[a]+(kind==a?0.:.5);'),
        ('double world=position[a]+(dt*bound_gate(kind*3+a))*p;',
         'double world=position[a]+(alpha*bound_gate(kind*3+a))*p;'),
        ('query[a]=world/h;', 'query[a]=world;'),
        ('double delta=dt*value;delta=delta< -h?-h:delta;delta=delta>h?h:delta;',
         'double delta=alpha*value;delta=delta< -1.?-1.:delta;delta=delta>1.?1.:delta;'),
        ('final[a]=world/h-(kind==a?0.:.5);',
         'final[a]=world-(kind==a?0.:.5);'),
    ]
    for old, new in substitutions:
        if text.count(old) != 2:
            raise ValueError('Expected exactly generic and geometry sampler expressions: '+old)
        text = text.replace(old, new)
    text = '// N8-only trial: cell coordinates, changed floating-point rounding.\n'+text
    output.write_text(text)
    manifest = ROOT/reg['sampler_parameter_manifest']
    record = dict(status='generated_single_cell_coordinate_sampler',
                  manifest=str(manifest.relative_to(ROOT)), manifest_sha256=digest(manifest),
                  template_source=str(source.relative_to(ROOT)), template_source_sha256=digest(source),
                  generated_source=str(output.relative_to(ROOT)), generated_source_sha256=digest(output),
                  protocol_sha256=digest(protocol), substitutions=substitutions,
                  learned_coefficients_changed=False, coefficient_arithmetic_order_unchanged=True,
                  coordinate_rounding_changed=True, original_N1_path_unchanged=True,
                  model_forward_executed=False, cpu_model_forward=False, optimizer_updates=0)
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    print('Generated one cell-coordinate sampler; fresh native accuracy required', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    generate(parser.parse_args().protocol)
