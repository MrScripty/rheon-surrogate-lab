"""Keep the new matched study discoverable without merging historical populations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import math


def completed_four_pass_studies(base):
    """Read canonical study reports, including the sealed per-arm bank reports."""
    for name, family, tag in [
            ('pressure-four-pass-distillation-v1', 'Four-pass fine-tuning', 'four-pass-study'),
            ('pressure-low-mode-tail-v1', 'Low-mode tail fine-tuning', 'low-mode-tail-study'),
            ('pressure-single-mode-fit-v1', 'Single-mode and curriculum', 'single-mode-study'),
            ('pressure-fundamental-bank-fit-v1', 'Fundamental bank fitting', 'fundamental-bank-study'),
            ('pressure-shape-decay-v1', 'Shape learning-rate decay', 'shape-decay-study'),
            ('pressure-balanced-shapes-v1', 'Balanced shape aggregation', 'balanced-shapes-study')]:
        result = Path(base) / name / 'results.json'
        if name == 'pressure-balanced-shapes-v1':
            result = result.parent / 'comparison' / 'results.json'
        if not result.is_file():
            continue
        data = json.loads(result.read_text())
        if data['status'] != 'completed':
            raise ValueError(f'Completed fine-tuning study required: {result}')
        reports = [(result, data)]
        if name == 'pressure-fundamental-bank-fit-v1':
            reports = []
            for arm, evidence in data['arms'].items():
                path = result.parent / arm / 'results.json'
                source = os.path.relpath(path, Path(base).parent.parent)
                if (hashlib.sha256(path.read_bytes()).hexdigest() != data['sources'][source]
                        or json.loads(path.read_text()) != evidence or evidence['status'] != 'completed'):
                    raise ValueError(f'Fundamental bank arm evidence changed: {path}')
                reports.append((path, evidence))
        elif name == 'pressure-shape-decay-v1':
            reports = []
            for arm, summary in data['arms'].items():
                path = result.parent / 'reports' / arm / 'result.json'
                source = os.path.relpath(path, Path(base).parent.parent)
                evidence = json.loads(path.read_text())
                identifier = summary['training']['id']
                if (hashlib.sha256(path.read_bytes()).hexdigest() != data['sources'][source]
                        or evidence['status'] != 'completed'
                        or evidence['training'] != [summary['training']]
                        or evidence['cpu_execution'] != summary['cpu_execution']
                        or evidence['snapshots'][identifier]['summary'] != summary['snapshots'][identifier]['summary']):
                    raise ValueError(f'Shape-decay arm evidence changed: {path}')
                reports.append((path, evidence))
        for path, evidence in reports:
            yield name, family, tag, path, evidence


def reserved_four_pass_ids(base):
    experiments = Path(base).resolve().parents[1] / 'experiments'
    from full_ai_chart import IDENTIFIERS as full_ai_ids
    reserved = set(full_ai_ids.values())
    from detached_correction_chart import IDENTIFIERS, REPORT
    if (Path(base) / REPORT).is_file():
        reserved.update(IDENTIFIERS.values())
    from pressure_methods_chart import IDENTIFIERS as method_ids
    if (Path(base) / 'pressure-methods-v1/results.json').is_file():
        reserved.update(method_ids.values())
    from pressure_methods_v2_chart import IDENTIFIERS as method_v2_ids
    if (Path(base) / 'pressure-methods-v2/results.json').is_file():
        reserved.update(method_v2_ids.values())
    from pressure_methods_v3_chart import IDENTIFIERS as method_v3_ids
    if (Path(base) / 'pressure-methods-v3/results.json').is_file():
        reserved.update(method_v3_ids.values())
    from pressure_methods_v6_chart import IDENTIFIERS as method_v6_ids
    if (Path(base) / 'pressure-methods-v6/results.json').is_file():
        reserved.update(method_v6_ids.values())
    from pressure_methods_v5_chart import IDENTIFIERS as method_v5_ids
    if (Path(base) / 'pressure-methods-v5/results.json').is_file():
        reserved.update(method_v5_ids.values())
    from pressure_methods_v4_chart import IDENTIFIERS as method_v4_ids
    if (Path(base) / 'pressure-methods-v4/results.json').is_file():
        reserved.update(method_v4_ids.values())
    for study in ['pressure-low-mode-tail-v1', 'pressure-single-mode-fit-v1', 'pressure-fundamental-bank-fit-v1', 'pressure-shape-decay-v1', 'pressure-balanced-shapes-v1']:
        config = experiments / study / 'config.json'
        if config.is_file():
            reserved.update(row['model_id'] for row in json.loads(config.read_text())['arms'])
    result = Path(base) / 'pressure-four-pass-distillation-v1/results.json'
    if not result.is_file():
        return reserved
    data = json.loads(result.read_text())
    if data['schema'] != 'rheon-four-pass-results-v1' or data['status'] != 'completed':
        raise ValueError('Completed study required for its model identifiers')
    return {row['id'] for row in data['training']} | reserved


def completed_four_pass_models(base, lab, source_lab):
    """Bind existing fine-tuning study IDs to verified restored checkpoints."""
    reserved_four_pass_ids(base)
    models = []
    for study_name, family, _, result, data in completed_four_pass_studies(base):
        for row in data['training']:
            receipt_path = lab / row['source']
            receipt = json.loads(receipt_path.read_text())
            study = receipt_path.parent
            intent = json.loads((study / 'intent.json').read_text())
            assembly = Path(intent['assembly'])
            artifacts = study / 'restore'
            history = json.loads((artifacts / 'history.json').read_text())
            sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            updates = row['updates'] if 'updates' in row else row['optimizer_updates']
            if (sha(receipt_path) != row['sha256'] or sha(assembly) != receipt['assembly_sha256']
                    or receipt['status'] != 'completed' or receipt['optimizer_updates'] != updates
                    or not all(receipt['artifact_parity'].values())
                    or sha(artifacts / 'weights.npz') != row['weights_sha256']
                    or json.loads((artifacts / 'receipt.json').read_text()) != receipt['restore_receipt']):
                raise ValueError(f'Fine-tuning fit/restore identity changed: {row["id"]}')
            models.append(dict(name=row['arm']['id'], cohort_id=row['id'],
                study_id=study_name, family=family,
                project=os.path.relpath(assembly.parent, lab), artifacts=os.path.relpath(artifacts, lab),
                weights_sha256=row['weights_sha256'], source_lab=os.path.relpath(source_lab, lab),
                started_at=datetime.fromtimestamp(history['training_started_unix'], timezone.utc).isoformat(),
                validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
                training_source=os.path.relpath(receipt_path, base), training_sha256=row['sha256'],
                study_source=os.path.relpath(result, base), study_sha256=sha(result),
                config_source=os.path.relpath(assembly.parent / 'config.json', base),
                optimizer_updates=updates, initialized_from='C53'))
            if row.get('single_case'):
                single = row['single_case']
                case = single['case']
                cells = 1
                for count in case['counts_xyz']:
                    cells *= count
                forwards = single['optimizer_examples'] * single['optimizer_network_passes_per_example']
                if (single['optimizer_examples'] != updates
                        or history['optimizer_work']['network_forwards'] != forwards
                        or history['optimizer_work']['forward_cells'] != forwards*cells):
                    raise ValueError('Single-case optimizer exposure differs from recorded work')
                models[-1]['manufactured_training_case'] = dict(counts_xyz=case['counts_xyz'],
                    cell_size_m=1/case['cells_per_metre'], mode_zyx=case['mode_zyx'],
                    presentations=single['optimizer_examples'])
            if row.get('fundamental_bank'):
                bank = row['fundamental_bank']
                stats = receipt['training_receipt']['fine_tuning_statistics']
                cases, exposures = bank['cases'], bank['optimizer_exposures']
                passes = row['arm']['steps']
                examples = bank['optimizer_examples']
                aggregated = bank.get('aggregation') == 'mean-complete-balanced-cycle'
                if aggregated and (bank['optimizer_updates'] != updates or row['examples'] != examples
                        or any(n != updates for n in exposures) or examples != updates*len(cases)):
                    raise ValueError('Balanced-cycle example/update counters disagree')
                if (stats['fundamental_bank'] != bank or stats['teacher_forwards'] != 0
                        or len(cases) != len(exposures) or exposures != bank['expected_exposures']
                        or sum(exposures) != examples or (not aggregated and examples != updates)
                        or history['optimizer_work']['network_forwards'] != examples*passes
                        or history['optimizer_work']['forward_cells'] != passes*sum(
                            math.prod(c['counts_xyz'])*n for c,n in zip(cases,exposures))
                        or any(history.get(key) is not None for key in
                            ['analytical_feedback','supplemental_feedback','current_feedback','direction_feedback'])):
                    raise ValueError('Fitting-bank optimizer exposure differs from recorded work')
                models[-1]['manufactured_training_bank'] = [dict(counts_xyz=c['counts_xyz'],
                    cell_size_m=1/c['cells_per_metre'], mode_zyx=c['mode_zyx'], presentations=n)
                    for c,n in zip(cases,exposures)]
                if aggregated:
                    models[-1].update(optimizer_examples=examples, optimizer_aggregation=bank['aggregation'],
                        optimizer_work=history['optimizer_work'])
    return models


def import_four_pass_rollouts(data, base, lab, load):
    """Add measured fresh episodes, retaining fixed-pass policy and scene identity."""
    result = Path(base) / 'pressure-four-pass-distillation-v1/results.json'
    if not result.is_file():
        return
    study = load(result)
    checkpoints = {p['id']: p for p in data['cohort_points']}
    trace_ids = {t['id'] for t in data['drift_traces']}
    summaries = []
    for name, campaign in study['campaigns'].items():
        if name == 'native-control':
            continue
        identifier = name.split('-p')[0]
        if identifier not in checkpoints:
            raise ValueError(f'Missing restored four-pass checkpoint: {identifier}')
        weight = checkpoints[identifier]['weights_sha256']
        observed = 0
        for episode in campaign['episodes']:
            episode_path = lab / episode['source']
            saved = load(episode_path, episode['sha256'])
            receipt_path = episode_path.parent / 'native/receipt.json'
            receipt = load(receipt_path, episode['native_receipt_sha256'])
            frames_path = receipt_path.with_name('frames.jsonl')
            raw = frames_path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha != episode['native_frames_sha256'] or not receipt['native_process']['reaped']:
                raise ValueError(f'Four-pass native frame/custody mismatch: {episode_path}')
            data['history_sources'][os.path.relpath(frames_path, base)] = sha
            adapter = load(receipt['adapter_manifest'], receipt['adapter_manifest_sha256'])
            material = load(adapter['material'], episode['material_sha256'])
            model = next(m for m in material['models'] if m['id'] == identifier)
            if (model['weights_sha256'] != weight or adapter['passes'] != episode['passes']
                    or saved['points'] != episode['points']):
                raise ValueError('Four-pass checkpoint/policy identity differs')
            frames = [json.loads(line) for line in raw.decode().splitlines()]
            ai = [f for f in frames if f.get('ai_quality')]
            values = [[f['step'], f['time_s'], f['relative_pressure_difference'], f['ai_quality']['qualified'],
                       f['ai_quality']['relative_residual']] for f in ai]
            expected = [[p['step'],p['time_s'],p['relative_pressure_difference'],p['qualified'],p['relative_equation_residual']]
                        for p in episode['points']]
            if values != expected:
                raise ValueError('Four-pass episode differs from its native frames')
            if not values:
                continue  # Rejection is retained in the summary, never a fabricated curve.
            policy = {k: receipt.get(k) for k in ['learned_proposals_per_pressure_solve','adaptive_stopping',
                      'proposal_network_evaluations','minimum_network_evaluations','pressure_bound']}
            scene_key = hashlib.sha256(json.dumps(dict(scene=receipt['scene'],binary=receipt['binary_sha256']),sort_keys=True).encode()).hexdigest()[:20]
            trace_id = hashlib.sha256(json.dumps([scene_key,weight,policy,values],sort_keys=True).encode()).hexdigest()[:20]
            if trace_id not in trace_ids:
                data['drift_traces'].append(dict(id=trace_id,scene_key=scene_key,scene=receipt['scene'],model=checkpoints[identifier]['name'],
                    policy_label=f"Fixed {episode['passes']} passes · fresh fine-tuning study",weights_sha256=weight,policy=policy,
                    points=values,completed=episode['status']=='completed',hard_stop=bool(receipt['result']['hard_stop']),
                    qualified_steps=episode['qualified_ai_steps'],source=os.path.relpath(result,base),
                    frames_source=os.path.relpath(frames_path,base),receipt_source=os.path.relpath(receipt_path,base),
                    chart_study='pressure-four-pass-distillation-v1'))
                trace_ids.add(trace_id)
            observed += 1
        summaries.append(dict(model_id=identifier,weights_sha256=weight,passes=campaign['episodes'][0]['passes'],
            scheduled_scenes=len(campaign['episodes']),observed_traces=observed,
            complete_windows=sum(e['status']=='completed' for e in campaign['episodes']),
            rejected_windows=sum(e['status']=='pressure_rejection' for e in campaign['episodes']),
            source=os.path.relpath(result,base)))
    data['four_pass_rollout_summary'] = summaries


def four_pass_link(text, base):
    report = Path(base) / 'pressure-four-pass-distillation-v1/index.html'
    if not report.is_file():
        return text
    start, end = '<!-- BEGIN FOUR PASS STUDY -->', '<!-- END FOUR PASS STUDY -->'
    block = start + '\n<p class="note"><a href="pressure-four-pass-distillation-v1/index.html">New: matched four-pass CUDA training study</a> — C53/C55 baselines, C57–C59 fine-tunes, CPU accuracy per pass, fresh-simulation drift, timing and retained methods.</p>\n' + end + '\n'
    if (Path(base) / 'pressure-low-mode-tail-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-low-mode-tail-v1/index.html">Small-grid low-mode training screen: C60/C61</a> — matched CUDA training, CPU accuracy/time, mode/amplitude diagnostics and retained results.</p>\n' + end)
    if (Path(base) / 'pressure-single-mode-fit-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-single-mode-fit-v1/index.html">Fundamental fit and frequency curriculum: C62/C63</a> — learning curve, matched spectral exposure, restored CPU accuracy/time and retained results.</p>\n' + end)
    if (Path(base) / 'pressure-fundamental-bank-fit-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-fundamental-bank-fit-v1/index.html">Shared fundamental fitting capacity: C64–C66</a> — per-grid/axis capacity, learning curves, CPU regression accuracy and retained time/memory.</p>\n' + end)
    if (Path(base) / 'pressure-shape-decay-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-shape-decay-v1/index.html">Later-decay fitting comparison: C67; C68 skipped</a> — matched learning curves, shape-gradient relationships, restored CPU accuracy and time/memory.</p>\n' + end)
    if (Path(base) / 'pressure-loss-components-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-loss-components-v1/index.html">Loss-component diagnosis: C53/C66/C67</a> — final/early shape-gradient alignment, precision limitations and retained methods/timings.</p>\n' + end)
    if (Path(base) / 'pressure-balanced-shapes-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-balanced-shapes-v1/index.html">Balanced shape aggregation: C69</a> — 1,600 examples combined into 200 Adam updates, CPU fitting/regression accuracy and retained time/memory.</p>\n' + end)
    if (Path(base) / 'pressure-aggregate-gradients-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-aggregate-gradients-v1/index.html">Shared gradient diagnosis: C53/C66/C69</a> — case influence, final/early directions, per-case norm caps and retained CPU timings/limits.</p>\n' + end)
    if (Path(base) / 'pressure-adam-replay-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-adam-replay-v1/index.html">C69 Adam observation replay</a> — exact checkpoint replay, observed optimizer steps, pressure changes, timings and precision limits.</p>\n' + end)
    if (Path(base) / 'pressure-adam-step-screen-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-adam-step-screen-v1/index.html">C69 CPU Adam step screen</a> — five saved optimizer steps and 57 distinct CPU parameter probes, with retention guards, calibration and measured runtime. These are diagnostics of C69; no new trained cohort was produced.</p>\n' + end)
    if (Path(base) / 'flow-reasoning-assessment/execution/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="flow-reasoning-assessment/outcome.md">Detached-correction training: C70–C77</a> — eight new checkpoints from C53/C44/C37; the ninth run exactly reproduces C66. Compare unmodified, attached and detached objectives. <a href="flow-reasoning-assessment/execution/results.json">Measured gates and source records</a>.</p>\n' + end)
    if (Path(base) / 'pressure-methods-v1/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#methods">Pressure-method training: C78–C88</a> — eleven distinct checkpoints; larger domains, teacher timesteps, numerical/learned coarse stages and spacing ablations. <a href="pressure-methods-v1/results.md">Study results</a> · <a href="pressure-methods-v1/results.json">Sealed campaign evidence</a>.</p>\n' + end)
    if (Path(base) / 'pressure-methods-v2/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#methodsv2">Pressure-method training II: C89–C99</a> — shape retention, learned shape features, numerical coarse nodes and sampling repeats; the small control reproduces C78. <a href="pressure-methods-v2/results.html">Full report</a> · <a href="pressure-methods-v2/results.json">Sealed results</a>.</p>\n' + end)
    if (Path(base) / 'pressure-methods-v3/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#methodsv3">Pressure-method training III: C100–C103</a> — retained replay, native 64³/128³ training and genuinely reduced-width fits. Matched adaptive policy timing, large-grid qualification and four-step coupled checks remain separate. <a href="pressure-methods-v3/results.html">Full report and audits</a>.</p>\n' + end)
    if (Path(base) / 'full-ai-next-priorities-v1/frontiers.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#fullai">Full AI continuation: C119–C130</a> — learned head calibration, compact architecture and eight-mode tail fits; separate multi-step full-state diagnostics. <a href="full-ai-next-priorities-v1/results.html">Campaign report</a>.</p>\n' + end)
    if (Path(base) / 'pressure-methods-v6/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#methodsv6">Pressure-method training VI: C115–C118</a> — solver-aware local corrections, 27/9-parameter filters, selective inference and 256-step independent drift. <a href="pressure-methods-v6/results.html">Full report and audits</a>.</p>\n' + end)
    if (Path(base) / 'pressure-methods-v5/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#methodsv5">Pressure-method training V: C112–C114</a> — tiny local correctors, matched solver timing and fresh/recycled trajectory drift. <a href="pressure-methods-v5/results.html">Full report and audits</a>.</p>\n' + end)
    if (Path(base) / 'pressure-methods-v4/results.json').is_file():
        block = block.replace(end, '<p class="note"><a href="cohort-progress.html#methodsv4">Pressure-method training IV: C104–C111</a> — student/control fits, solver-aware training, diverse and on-policy states, repeated solver timing and independent drift. <a href="pressure-methods-v4/results.html">Full report and audits</a>.</p>\n' + end)
    if (Path(base) / 'pressure-adam-momentum-damping-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-adam-momentum-damping-v1/index.html">C69 partial first-moment damping</a> — 30 distinct CPU parameter probes with second moment and Adam counter preserved; all-case guards, step geometry and timing retained.</p>\n' + end)
    if (Path(base) / 'pressure-cuda-gradient-conflicts-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-cuda-gradient-conflicts-v1/index.html">C69 CUDA gradient conflicts</a> — exact captured-gradient replay at three saved states; per-case/pass conflicts, layer contributions, time and memory.</p>\n' + end)
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Ambiguous study link region')
        return re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n?', lambda _: block, text, count=1, flags=re.S)
    anchor = '<noscript>'
    if text.count(anchor) != 1:
        raise ValueError('Dashboard study link seam missing or ambiguous')
    return text.replace(anchor, block + anchor, 1)
