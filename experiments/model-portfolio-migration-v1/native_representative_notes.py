"""Add concise real model cards and a diagnostic Study through native owners."""
import hashlib
import json

from native_representative_models import ROOT, BASE, OUTPUT, perform


def run():
    registered = json.loads((OUTPUT / 'representative-models.json').read_text())
    models = {m['identifier']: m for m in registered['models']}
    score = json.loads((ROOT / 'docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json').read_text())
    cards = {
        'C107': (
            'Half-width pressure CNN with learned coarse proposals; the no-distillation control of C106. '
            'Encoder widths 4/8/12/16, bottleneck24, decoder16/12/8/4; linear fine CNN, '
            'float32 network and float64 pressure. Two correction stages. 93,758 parameters. '
            'C106/C107 are sibling fits from a channel-sliced C101 initialization, not a C106-to-C107 fine-tune.\n\n'
            'Training: 2,400 updates; distillation weight0; hash-bound captured data and source. '
            'Exact scene membership and clean regeneration still need the reproduction exercise.\n\n'
            'Lesson: lower equation residual is distinct from lower pressure error. C107 has lower '
            'pressure error than C53 on122/128 requests but a worse maximum. Neither result is a complete-state or drift test.'),
        'C128': (
            'a-tail8: a learned final spectral correction after a frozen routed pressure parent. '
            'The tail covers up to eight cosine modes per axis (511 nonconstant gains), conditioned '
            'by a3->64tanh->511 shape head. Eight means modes, not eight simulation outputs or CNN stages. '
            'The parent has two correction stages. Total128,938 parameters;33,982 trainable tail parameters.\n\n'
            'Training:10,000 calibration updates, batch256, with parent parameters frozen. '
            'The parent is the a-head-calibration-s2 routed assembly over C107 and its calibrated small-grid head. '
            'The final tail proposal is accepted only if its actual equation residual does not increase.\n\n'
            'Applicability: the tail is bypassed whenever any grid axis exceeds16 cells. '
            'Its leading result on the128 small-grid requests therefore does not establish an advantage on32-cubed or larger domains.\n\n'
            'Lesson: retain the parent/calibration/assembly distinctions and benchmark the larger-grid branch explicitly. '
            'No inference-time eigenvalue inverse is evaluated by the tail; its gains are trained parameters.')
    }
    for index, identifier in enumerate(('C107', 'C128')):
        record = score['models'][identifier]
        model = models[identifier]
        text = (identifier + ' · retained model card\n\n' + cards[identifier] + '\n\n'
                'Fresh consumed-case GPU diagnostic:128 original native pressure requests; densities8 and16 cells/metre. '
                f"Worst mean-free pressure relativeL2={record['worst']['pressure_mean_free_relative_l2']:.17g}; "
                f"worst equation relativeL2={record['worst']['common_equation_relative_l2']:.17g}. "
                'All fields independently rescored. No fresh GPU latency, independent-validation or physical acceptance claim.\n\n'
                'Native model definition: ' + model['model_definition']['id'] + '\n'
                'Published source material: ' + model['published']['ref']['sha256'] + '\n'
                'Selected checkpoint SHA256: ' + model['original_checkpoint_sha256'] + '\n'
                'Original source snapshot: ' + model['source_project_id'] + '\n\n'
                'Reproduction: retain the captured project source/configuration, exact parameter artifact and original old-Rheon data identities. '
                'Use the frozen pressure evaluation protocol and source-bound adapter before comparing new implementations. '
                'New Rheon cannot regenerate an old-physics target under the same identity.\n\n'
                'This card and native source definition were authored retrospectively. '
                'Original training Run/Study history has not yet been extracted into this project; no new fit or execution authority is claimed.')
        perform(str(20 + index), 'data_import_text', dict(text=text, name=identifier + ' · method, result and reproduction',
            groups=['rheon-portfolio', 'model-' + identifier, 'retained-model-card'],
            rights='Project-owned retrospective documentation grounded in retained source, receipts and measured results.'))
    overview = ('Rheon model portfolio · migration in progress\n\n'
        '127 available named pressure checkpoints have GPU results on the same128 original requests each. '
        'C01/C02 remain unavailable. These are pressure-only consumed diagnostics; larger cases, intermediate checkpoints, '
        'full-state variants, own-history drift and uncontended timing remain incomplete.\n\n'
        'Both cited full-state accuracy anchors reproduced exactly. Their historical timings remain historical. '
        'The cell-coordinate quoted case uses N1 despite its N8 family label; the temporal quote uses the fine-reference schedule.\n\n'
        'This native project currently contains representative C107/C128 source definitions and cards. '
        'The complete old Run/Trial history and retrospective lineage still require migration. '
        'Do not interpret the new editable definitions as recreated historical execution authority.\n\n'
        'No new Rheon solver has been adopted. Original and new physics will retain separate reference identities. '
        'No unique model weights were deleted. GPU inference remains primary; all thresholds remain diagnostic until application limits are set.')
    perform('22', 'data_import_text', dict(text=overview, name='Portfolio status and comparison rules',
        groups=['rheon-portfolio', 'portfolio-overview'], rights='Project-owned experiment documentation.'))
    definition = dict(format='rheon_pressure_mean_free_relative_l2_v1', field='pressure', units='1',
                      spatial_norm='L2', gauge='subtract_each_field_spatial_mean',
                      denominator='max(mean-free reference L2,1e-30)', case_aggregation='maximum',
                      population='Exact original128 consumed native pressure requests; complete membership retained.',
                      acceptance='diagnostic')
    raw = json.dumps(definition, sort_keys=True, indent=2) + '\n'
    (OUTPUT / 'pressure-metric-definition.json').write_text(raw)
    perform('23', 'data_import_text', dict(text=raw, name='Pressure metric definition · diagnostic',
        groups=['rheon-portfolio', 'metric-definition'], rights='Project-owned metric declaration.'))
    perform('24', 'study_create', dict(name='Old-reference pressure comparison · diagnostic retained checkpoints',
        limits=dict(attempt_slots=4, admitted_wall_seconds=600, retained_bytes=1073741824, peak_staging_bytes=1073741824),
        policy=dict(format='selection_policy_v1', metric_definition_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    checkpoint_rule='explicit', replicate_reduce='individual', required_capabilities=[],
                    numerical_acceptance='diagnostic', tie_break='trial_id')))
    print('Added native model cards, comparison rules, metric definition and diagnostic Study; no Trial or Run fabricated', flush=True)


if __name__ == '__main__':run()
