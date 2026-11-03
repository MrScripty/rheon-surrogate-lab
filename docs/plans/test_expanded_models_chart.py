"""Scope and source checks for the recorded expanded CPU/GPU measurements."""
import json
from pathlib import Path
import statistics
import unittest

from dashboard_build import DATA

ROOT=Path(__file__).resolve().parent


class ExpandedMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(DATA.search((ROOT/'cohort-progress.html').read_text())[2])['full_ai_study']['expanded_models']

    def test_new_training_and_reserved_scope(self):
        self.assertEqual(self.data['component']['parameters'],4)
        self.assertEqual(self.data['component']['optimizer_calls'],160)
        self.assertEqual(self.data['training']['training_scenes'],148)
        self.assertEqual(self.data['training']['development_scenes'],24)
        self.assertEqual(self.data['training']['reserved_opened'],0)
        self.assertEqual(self.data['training']['complete_K4_windows'],592)
        self.assertEqual(len(self.data['component']['groups']),24)
        self.assertTrue(any(v['id']=='native-cpu1' and v['kind']=='reference' for v in self.data['variants']))

    def test_timing_samples_and_matching_block_accuracy(self):
        timings=self.data['timing']
        self.assertEqual({r['device'] for r in timings},{'cpu','gpu'})
        self.assertTrue(any(r['internal_substeps']==8 for r in timings))
        for source in {r['timing_source'] for r in timings}:
            value=json.loads((ROOT/source).read_text())
            relevant=[r for r in timings if r['timing_source']==source]
            bycase={}
            if value.get('timings'):
                for row in value['timings']:bycase[(row['identifier'],row['K'],row['start'])]=row
                for row in relevant:
                    raw=bycase[(row['identifier'],row['K'],row['start'])]
                    clock=row['scope'].removesuffix('_provided_controls').removesuffix('_complete_controls')
                    self.assertEqual(row['median_block_ms'],raw['full_block_ms'][clock]['median'])
                    self.assertEqual(row['maximum_block_ms'],raw['full_block_ms'][clock]['maximum'])
                    self.assertEqual(row['repetitions'],raw['full_block_ms'][clock]['samples'])
            else:
                for raw in value['rows']:
                    if raw.get('warm_call'):continue
                    key=(raw['identifier'],raw.get('K',raw.get('horizons')),raw['start'],raw.get('mode'))
                    bycase.setdefault(key,[]).append(raw)
                for row in relevant:
                    mode=row['variant'].split('-',1)[1] if row['variant'].startswith('entry') else None
                    selected=bycase[(row['identifier'],row['K'],row['start'],mode)]
                    samples=[r.get('full_block_ms',r.get('complete_block_ms',r.get('complete_block_with_state_transfer_ms'))) for r in selected]
                    self.assertEqual(row['median_block_ms'],statistics.median(samples))
                    self.assertEqual(row['maximum_block_ms'],max(samples))
                    self.assertEqual(row['repetitions'],len(samples))
            self.assertTrue(all(r['median_block_ms']>0 and r['maximum_block_ms']>=r['median_block_ms'] for r in relevant))
        for source in {r['source'] for r in timings}:
            value=json.loads((ROOT/source).read_text());cases={}
            for raw in value['rows']:
                if raw.get('warm_call'):continue
                key=(raw['identifier'],raw.get('K',raw.get('horizons')),raw['start'])
                metrics=raw['metrics'] if isinstance(raw.get('metrics'),list) else [raw.get('metrics')]
                if metrics==[None]:continue
                cases.setdefault(key,[]).extend(metrics)
            for row in (r for r in timings if r['source']==source):
                selected=cases[(row['identifier'],row['K'],row['start'])]
                self.assertEqual(row['field_worst'],{f:max(r['relative'][f] for r in selected) for f in row['field_worst']})

    def test_original_failures_and_repair_are_separate(self):
        traces=self.data['traces']
        self.assertTrue(all([p[0] for p in t['errors']]==list(range(1,241)) for t in traces))
        self.assertTrue(any(t['variant']=='raw' and t['failed_steps']>0 for t in traces))
        self.assertTrue(all(t['failed_steps']==0 for t in traces if t['variant']=='repair7' and t['population']=='legacy_consumed'))
        large=[t for t in traces if t['population']=='streamed_large' and t['variant']=='fusion6']
        self.assertEqual(len(large),16)
        self.assertEqual(len({t['identifier'] for t in large if t['failed_steps']>0}),2)
        self.assertFalse(self.data['repair']['full_large_retest_complete'])
        self.assertFalse(self.data['repair']['maximum_latency_verified'])
        self.assertTrue(any(max(t['counts_xyz'])>100 and t['failed_steps']>0 for t in large))

    def test_later_repairs_and_untimed_layout_scope(self):
        traces=self.data['traces']
        repaired=[t for t in traces if t['variant']=='transform9' and t['population']=='streamed_large']
        self.assertEqual(len(repaired),16)
        self.assertTrue(all(t['failed_steps']==0 for t in repaired))
        self.assertTrue(any(t['variant']=='sampler8' and t['counts_xyz']==[128]*3 and t['failed_steps'] for t in traces))
        failed=[t for t in traces if t['variant']=='affine12' and t['counts_xyz']==[64]*3 and t['cells_per_metre']==100]
        fixed=[t for t in traces if t['variant']=='layout14' and t['counts_xyz']==[64]*3 and t['cells_per_metre']==100]
        self.assertEqual(len(failed),4);self.assertEqual(len(fixed),4)
        self.assertTrue(all(t['failed_steps']>0 for t in failed))
        self.assertTrue(all(t['failed_steps']==0 for t in fixed))
        self.assertFalse(any(t['variant']=='layout14' for t in self.data['timing']))
        self.assertEqual(sum(t['scored_states'] for t in self.data['accuracy'] if t['variant']=='layout14' and t['interval']=='single_block'),960)
        self.assertFalse(self.data['progress']['full_generator_targets_verified'])
        self.assertLess(self.data['progress']['layout_development_cases'],24)

    def test_equivalent_histories_keep_both_sources(self):
        traces=[t for t in self.data['traces'] if t.get('equivalence_source')]
        self.assertTrue({'allocation10','integer11','affine12','tables13','layout14','native-entry2','FFT-pressure'}<={t['variant'] for t in traces})
        for source in {t['equivalence_source'] for t in traces}:
            raw=json.loads((ROOT/source).read_text())
            self.assertIn(len(raw['rows']),(3840,23040,24960))
            indexed={(r['identifier'],r['K'],r['step']):r for r in raw['rows']}
            for baseline in {t['source'] for t in traces if t['equivalence_source']==source}:
                native=json.loads((ROOT/baseline).read_text())
                for row in native['rows']:
                    actual=indexed[(native['identifier'],row['K'],row['step'])]
                    self.assertEqual(actual['output_sha256'],row['output_sha256'])
                    self.assertTrue(actual.get('native_qualified_baseline_exact',actual.get('all_field_bits_exact',False)))
                for t in (t for t in traces if t['equivalence_source']==source and t['source']==baseline):
                    expected=[[r['step'],*[r['metrics']['relative'][f] for f in ('equation','pressure','velocity','tracer')]] for r in native['rows'] if r['K']==t['K']]
                    self.assertEqual(t['errors'],expected)

    def test_all24_development_and_later_execution_coverage(self):
        for variant in ['layout14','native-entry2','bound-sampler','cell-coordinate','FFT-pressure']:
            rows=[t for t in self.data['traces'] if t['variant']==variant and t['identifier'].startswith('diversity-development-')]
            self.assertEqual(len(rows),96,variant)
            self.assertEqual(len({t['identifier'] for t in rows}),24)
            self.assertTrue(all(t['failed_steps']==0 for t in rows))
        self.assertEqual(self.data['current_status']['development_outputs'],23040)
        self.assertEqual(self.data['current_status']['retained_baseline'],'native-entry2')
        self.assertFalse(any(self.data['current_status']['full_generator_targets_verified'].values()))
        self.assertEqual(self.data['current_status']['reserved_opened'],0)

    def test_students_keep_training_population_and_actual_work(self):
        self.assertEqual(self.data['students']['additional_fits'],2)
        self.assertEqual(self.data['students']['additional_optimizer_calls'],320)
        for stages in (3,4):
            variant=f'student{stages}'
            rows=[r for r in self.data['accuracy'] if r['variant']==variant]
            self.assertEqual(sum(r['scored_states'] for r in rows),1018)
            self.assertEqual({r['split'] for r in rows},{'train','development'})
            n8=[r for r in rows if r['teacher_substeps']==8]
            self.assertTrue(n8)
            self.assertTrue(all(r['internal_substeps']==stages and not r['accuracy_pass'] for r in n8))
            self.assertFalse(any(r['variant']==variant for r in self.data['timing']+self.data['traces']))
            source=ROOT/rows[0]['source'];raw=json.loads(source.read_text())['rows']
            for split in ['train','development']:
                actual=[r for r in n8 if r['split']==split]
                expected=[r for r in raw if r['split']==split and r['teacher_substeps']==8]
                self.assertEqual(max(r['field_worst']['equation'] for r in actual),max(r['metrics']['relative']['equation'] for r in expected))

    def test_rejected_precision_and_slower_fft_remain_visible(self):
        fp32=[r for r in self.data['accuracy'] if r['variant']=='FP32-interpolator']
        self.assertEqual(sum(r['scored_states'] for r in fp32),1018)
        self.assertTrue(any(not r['accuracy_pass'] for r in fp32))
        self.assertFalse(any(r['variant']=='FP32-interpolator' for r in self.data['timing']+self.data['traces']))
        path=ROOT/'../../runs/full-ai-data-diversity-v1/FFT-pressure-benchmark-01/results.json'
        value=json.loads(path.read_text())
        for k in range(1,5):
            rows=[r for r in value['rows'] if r['counts_xyz']==[32]*3 and r['cells_per_metre']==100 and r['K']==k]
            parent=statistics.median(r['ms_per_generated_step'] for r in rows if r['mode']=='cell_coordinate_entry')
            trial=statistics.median(r['ms_per_generated_step'] for r in rows if r['mode']=='FFT_pressure_entry')
            self.assertGreater(trial,parent)

    def test_residual_fits_and_initial_selection_remain_explicit(self):
        fits={r['id']:r for r in self.data['residual_fits']}
        self.assertTrue({'residual-s1','residual-s2','residual-s3','residual-s4'}<=set(fits))
        self.assertEqual(fits['residual-s1']['selected_epoch'],2)
        self.assertEqual(fits['residual-s2']['selected_epoch'],0)
        self.assertEqual(fits['residual-s1']['native_failed'],1151)
        self.assertEqual(fits['residual-s2']['native_failed'],1147)
        for variant,fit in ((k,fits[k]) for k in ('residual-s1','residual-s2')):
            self.assertEqual(fit['parameters'],10876);self.assertEqual(fit['optimizer_updates'],1566)
            self.assertEqual((fit['active_scenes'],fit['control_scenes']),(29,119))
            rows=[r for r in self.data['accuracy'] if r['variant']==variant]
            self.assertEqual(sum(r['scored_states'] for r in rows),2120)
            self.assertEqual({r['split'] for r in rows},{'train','development'})
            self.assertTrue(any(not r['accuracy_pass'] for r in rows))
            self.assertFalse(any(r['variant']==variant for r in self.data['timing']+self.data['traces']))
        folder=ROOT/'../../runs/full-ai-data-diversity-v1/internal-residual-s2-pair-01/restore'
        self.assertEqual((folder/'weights.npz').read_bytes(),(folder/'initial-weights.npz').read_bytes())

    def test_residual_diagnostic_costs_have_no_invented_accuracy_pairing(self):
        timings=self.data['diagnostic_timing']
        self.assertEqual(sum(r['repetitions'] for r in timings),1160)
        self.assertTrue(all(not r['accuracy_paired_to_actual_timed_outputs'] and 'field_worst' not in r for r in timings))
        self.assertEqual({r['parent_variant'] for r in timings},{'residual-s1','residual-s2','residual-graph','residual-fused','residual-s3-fused','residual-s4-fused'})
        for source in {r['source'] for r in timings}:
            raw=json.loads((ROOT/source).read_text());groups={}
            for row in raw['rows']:
                key=(row.get('identifier',raw.get('identifier')),row['K'],row.get('mode','fused_graph' if 's3-' in source or 's4-' in source else 'eager'))
                groups.setdefault(key,[]).append(row.get('wall_block_ms',row.get('block_ms')))
            for row in (r for r in timings if r['source']==source):
                selected=groups[(row['identifier'],row['K'],row['variant'].split('-',1)[1])]
                self.assertEqual(row['median_block_ms'],statistics.median(selected))
                self.assertEqual(row['maximum_block_ms'],max(selected))
                self.assertEqual(row['repetitions'],len(selected))
        parent=json.loads((ROOT/'../../runs/full-ai-data-diversity-v1/internal-residual-s1-native-blocks-01/results.json').read_text())
        for name in ['internal-residual-s1-native-graph-01','internal-residual-s1-native-fused-graph-01']:
            child=json.loads((ROOT/f'../../runs/full-ai-data-diversity-v1/{name}/results.json').read_text())
            self.assertEqual([r['output_sha256'] for r in child['rows']],[r['output_sha256'] for r in parent['rows']])

    def test_compensated_and_alpha_trials_keep_their_scopes(self):
        paired=[r for r in self.data['accuracy'] if r['variant']=='paired-interpolator']
        self.assertEqual(sum(r['scored_states'] for r in paired),1018)
        self.assertTrue(any(not r['accuracy_pass'] for r in paired))
        timed=[r for r in self.data['timing'] if r['variant'].startswith('entry16-')]
        self.assertTrue(timed)
        self.assertTrue(all(r['scope']=='diagnostic_complete_entry' for r in timed))
        self.assertFalse(any(r['variant']=='paired-interpolator' for r in self.data['traces']))
        alpha=[r for r in self.data['traces'] if r['variant']=='alpha-argument' and r['identifier'].startswith('diversity-development-')]
        self.assertEqual(len(alpha),96)
        self.assertTrue(all(not r['failed_steps'] for r in alpha))

    def test_s3_s4_keep_optimizer_and_closure_counts_separate(self):
        fits={r['id']:r for r in self.data['residual_fits']}
        for name,updates,closures in [('s3',7,18),('s4',24,51)]:
            fit=fits[f'residual-{name}']
            self.assertEqual(fit['optimizer_updates'],updates)
            self.assertEqual(fit['closure_evaluations'],closures)
            self.assertEqual(fit['selected_evaluation'],closures)
            self.assertIsNone(fit['selected_epoch'])
            self.assertEqual(fit['native_failed'],1151)
            self.assertFalse(any(r['variant']==f'residual-{name}' for r in self.data['traces']+self.data['timing']))
            root=ROOT/'../../runs/full-ai-data-diversity-v1'
            eager=json.loads((root/f'internal-residual-{name}-native-blocks-01/results.json').read_text())
            graph=json.loads((root/f'internal-residual-{name}-native-fused-graph-01/results.json').read_text())
            self.assertEqual([r['output_sha256'] for r in eager['rows']],[r['output_sha256'] for r in graph['rows']])

    def test_temporal_pilot_raw_clocks_and_cadence_work(self):
        pilots=self.data['temporal_pilots']
        self.assertEqual(len(pilots['captures']),16)
        self.assertEqual({r['cadence'] for r in pilots['points']},{30,60,120})
        original={'frozen01','frozen02','frozen03','residual01','residual02','shared01','shared02','shared03'}
        self.assertTrue(all(r['split']=='train' for r in pilots['points'] if r['capture'] in original))
        for capture in (c for c in pilots['captures'] if c['id'] in original):
            raw=json.loads((ROOT/capture['source']).read_text())
            rows=[r for r in pilots['points'] if r['capture']==capture['id']]
            self.assertEqual(len(rows),len(raw['rows']))
            for actual,expected in zip(rows,raw['rows']):
                self.assertEqual(actual['wall_samples_ms'],expected['block_wall_ms'])
                self.assertEqual(actual['event_samples_ms'],expected['pipeline_cuda_event_span_ms'])
                self.assertEqual(actual['median_block_ms'],statistics.median(actual['wall_samples_ms']))
                self.assertAlmostEqual(expected['block_duration_s'],actual['K']/actual['cadence'])
                if 'native_metrics_per_output' in expected:
                    self.assertEqual(actual['field_worst'],{f:max(m['relative'][f] for m in expected['native_metrics_per_output']) for f in actual['field_worst']})
        self.assertTrue(any(not r['accuracy_pass'] for r in pilots['points'] if r['capture']=='shared03'))
        self.assertTrue(all(r['encoder_calls']==1 and r['coarse_endpoint_transports']==1 and r['pressure_projections']==r['K']+1 for r in pilots['points'] if r['capture'].startswith('shared')))
        ready=pilots['readiness']
        self.assertEqual((ready['training_masters'],ready['development_masters']),(24,12))
        self.assertEqual(ready['finite_f32_recoveries'],4)
        self.assertFalse(ready['temporal_convergence_proven'])
        self.assertEqual(ready['new_parameter_fits'],0)

    def test_managed_shared_fits_preserve_initial_checkpoint_and_failed_job(self):
        training=self.data['temporal_pilots']['training']
        self.assertEqual(training['new_parameter_fits'],2)
        self.assertEqual(training['recovery_new_parameter_fits'],0)
        self.assertEqual(training['original_training_state'],'failed')
        self.assertFalse(training['detailed_window_history_available'])
        for width,parameters in [('4',1724),('8',3432)]:
            fit=training['fits'][width]
            self.assertEqual(fit['parameters'],parameters)
            self.assertEqual(fit['Adam_updates'],960)
            self.assertEqual(fit['selected_epoch'],0)
            self.assertTrue(fit['selected_parameters_equal_initial'])
            self.assertEqual(len(fit['full_epoch_objectives']),4)
            self.assertTrue(all(v>fit['initial_full_objective'] for v in fit['full_epoch_objectives']))
        folder=ROOT/'../../runs/full-ai-data-diversity-v1/shared-temporal-recovery-pair-01'
        for name in ['weights.npz','initial-weights.npz','last-evaluated-weights.npz','history.json','probe.npy','normalization.json']:
            self.assertEqual((folder/'train'/name).read_bytes(),(folder/'restore'/name).read_bytes())

    def test_successor_native_metrics_and_clock_coverage(self):
        pilots=self.data['temporal_pilots'];ids={'trainedDev','guideInitial','guideFull','fineInitial','fineFull'}
        for capture in (c for c in pilots['captures'] if c['id'] in ids):
            raw=json.loads((ROOT/capture['source']).read_text())
            rows=[r for r in pilots['points'] if r['capture']==capture['id']]
            self.assertEqual(len(rows),len(raw['rows']))
            for actual,expected in zip(rows,raw['rows']):
                self.assertEqual(actual['wall_samples_ms'],expected['block_wall_ms'])
                self.assertIsNone(actual['event_samples_ms'])
                self.assertEqual(actual['median_block_ms'],statistics.median(expected['block_wall_ms']))
                self.assertEqual(actual['field_worst'],{f:max(m['relative'][f] for m in expected['native_metrics_per_output']) for f in actual['field_worst']})
        dev=[r for r in pilots['points'] if r['capture']=='trainedDev']
        self.assertEqual(len(dev),5760)
        self.assertEqual(sum(r['K'] for r in dev),14400)
        self.assertEqual(len({r['identifier'] for r in dev}),12)
        self.assertEqual({(r['hidden_width'],r['mode']) for r in dev},{(4,'selected'),(4,'last_epoch'),(8,'selected'),(8,'last_epoch')})
        self.assertTrue(all(r['split']=='development' for r in dev))
        self.assertTrue(all(r['split']=='train' for r in pilots['points'] if r['capture'] in ids-{'trainedDev'}))

    def test_folded_cost_keeps_matched_original_accuracy(self):
        pilots=self.data['temporal_pilots'];scores={(r['identifier'],r['start_fine_step'],r['cadence'],r['K']):r for r in pilots['points'] if r['capture']=='fineFull' and r['mode']=='preoutput_fine_fullstate_moment_tracer'}
        rows=[r for r in pilots['points'] if r['capture']=='folded']
        self.assertEqual(len(rows),5760)
        self.assertEqual({r['mode'] for r in rows},{'original','folded'})
        raw=json.loads((ROOT/rows[0]['source']).read_text());indexed={(r['identifier'],r['start_fine_step'],r['output_steps_per_s'],r['K']):r for r in raw['rows']}
        self.assertTrue(raw['zero_head_certificate']['all_exact_zero'])
        for row in rows:
            key=(row['identifier'],row['start_fine_step'],row['cadence'],row['K']);parent=scores[key];record=indexed[key]
            self.assertEqual(row['field_worst'],parent['field_worst'])
            self.assertEqual(row['accuracy_source'],parent['source'])
            self.assertTrue(record['all_complete_output_bytes_equal'])
            self.assertEqual(row['wall_samples_ms'],record['block_wall_ms'][row['mode']])
            self.assertEqual(row['median_block_ms'],statistics.median(row['wall_samples_ms']))
        self.assertEqual(pilots['guided_readiness']['new_parameter_fits'],0)

    def test_guided_fit_unique_states_and_native_development_measurements(self):
        pilots=self.data['temporal_pilots'];fit=pilots['guided_training']
        self.assertEqual((fit['new_parameter_fits'],fit['parameter_trial_updates'],fit['accepted_steps']),(1,8,4))
        self.assertEqual(len(fit['training_trials']),8)
        self.assertEqual(fit['checkpoint_aliases'],{'best_mean':'selected','last':'selected'})
        self.assertLess(fit['selected_training_objective'],fit['initial_training_objective'])
        self.assertTrue(all(r['objective']>fit['initial_training_objective'] for r in fit['training_trials'] if r['label'].startswith('probe')))
        rows=[r for r in pilots['points'] if r['capture']=='guidedDev']
        self.assertEqual((len(rows),sum(r['K'] for r in rows)),(2880,7200))
        self.assertEqual({r['mode'] for r in rows},{'selected','initial'})
        raw=json.loads((ROOT/rows[0]['source']).read_text())
        for actual,expected in zip(rows,raw['rows']):
            self.assertEqual(actual['wall_samples_ms'],expected['block_wall_ms'])
            self.assertIsNone(actual['event_samples_ms'])
            self.assertEqual(actual['field_worst'],{f:max(m['relative'][f] for m in expected['native_metrics_per_output']) for f in actual['field_worst']})
            self.assertTrue(expected['managed_CPUrestore_certified'])
            self.assertEqual(actual['split'],'development')
        for mode,summary in fit['development'].items():
            selected=[r for r in rows if r['mode']==mode]
            self.assertEqual(len({r['identifier'] for r in selected}),12)
            self.assertEqual({f:max(r['field_worst'][f] for r in selected) for f in summary['worst_relative']},summary['worst_relative'])

    def test_minimal_guide_parity_chain_and_separate_paired_timings(self):
        pilots=self.data['temporal_pilots'];rows=[r for r in pilots['points'] if r['capture']=='minimal']
        self.assertEqual(len(rows),5760)
        self.assertEqual({r['mode'] for r in rows},{'folded_parent','minimal'})
        key=lambda r:(r['identifier'],r['start_fine_step'],r.get('cadence',r.get('output_steps_per_s')),r['K'])
        parent={key(r):r for r in pilots['points'] if r['capture']=='fineFull' and r['mode']=='preoutput_fine_fullstate_moment_tracer'}
        raw=json.loads((ROOT/rows[0]['source']).read_text());indexed={key(r):r for r in raw['rows']}
        folded=json.loads((ROOT/'../../runs/full-ai-data-diversity-v1/folded-fine-readout-parity-01/results.json').read_text());outputs={key(r):r['output_sha256'] for r in folded['rows']}
        for row in rows:
            record=indexed[key(row)]
            self.assertTrue(record['all_complete_output_bytes_equal'] and record['all_coarse_context_bytes_equal'])
            self.assertEqual(record['output_sha256'],outputs[key(row)])
            self.assertEqual(row['field_worst'],parent[key(row)]['field_worst'])
            self.assertEqual(row['accuracy_source'],parent[key(row)]['source'])
            self.assertEqual(row['wall_samples_ms'],record['block_wall_ms'][row['mode']])
            self.assertEqual(row['median_block_ms'],statistics.median(row['wall_samples_ms']))
        self.assertGreater(pilots['minimal_guide']['median_paired_GPU_speedup'],1)


if __name__=='__main__':unittest.main()
