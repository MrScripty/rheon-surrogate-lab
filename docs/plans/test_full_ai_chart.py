"""Admission and population checks for the newer full-state model charts."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from dashboard_build import DATA
from full_ai_chart import audit_passed, bind_temporal_timing, completed_temporal_specs

ROOT = Path(__file__).resolve().parent
NEW_TIMED = {f'K={k} {arm}' for k in (2, 3, 4) for arm in ('endpoint', 'reuse')} | {'K=2 coupled','K=2 continuation control','K=2 provisional supervision','K=2 displacement transport'}
NEW_UNTIMED = {
    'K=2 endpoint step-scaled', 'K=2 reuse step-scaled',
    'K=2 endpoint full-state MSE', 'K=2 endpoint increment MSE',
    'K=2 endpoint curl increment', 'K=2 pressure expert', 'K=2 decoder',
}


class TemporalAdmissionTests(unittest.TestCase):
    def test_nested_family_audits_require_every_control(self):
        self.assertTrue(audit_passed({'k1':{'parity':True},'k2':{'parity':True}}))
        self.assertFalse(audit_passed({'k1':{'parity':True},'k2':{'parity':False}}))
        self.assertFalse(audit_passed({'status':'failed','checks':{'parity':True}}))

    def test_followup_source_artifacts_are_hashed_without_json_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plans=root/'docs/plans'
            report_path=plans/'full-ai-next-priorities-v1/temporal-coupled-results.json'
            report_path.parent.mkdir(parents=True)
            source=root/'runs/study/control.py'
            source.parent.mkdir(parents=True)
            source.write_text('control = True\n')
            evaluation=root/'runs/full-ai-next-priorities-v1/temporal-coupled-evaluation-01/b2-coupled/summary.json'
            evaluation.parent.mkdir(parents=True)
            evaluation.write_text('{}')
            evidence={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [source,evaluation]}
            report_path.write_text(json.dumps(dict(status='completed',source_evidence=evidence)))
            load=lambda p,expected=None:json.loads(Path(p).read_text())
            self.assertEqual(len(completed_temporal_specs(plans,root,load)),7)
            source.write_text('control = False\n')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                completed_temporal_specs(plans,root,load)

    def test_flat_control_audit_must_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'full-ai-next-priorities-v1'
            path.mkdir()
            (path/'temporal-coupled-results.json').write_text(json.dumps(dict(
                status='completed',control_audit={'exact_CPU_artifact_parity':False},source_evidence={})))
            load=lambda p,expected=None:json.loads(Path(p).read_text())
            with self.assertRaisesRegex(ValueError,'control audit failed'):
                completed_temporal_specs(tmp,tmp,load)

    def test_tensor_only_timing_cannot_replace_complete_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab=Path(tmp)
            base=lab/'docs/plans'
            summary='runs/study/timing/summary.json'
            accuracy='runs/study/evaluation/summary.json'
            source='docs/plans/cost.json'
            row=dict(source=source,identifier='fit',regime='small',horizons=2,
                     full_block_ms=4,median_ms=2)
            report=dict(status='completed',CPU_latency_claim=True,
                        source_evidence={summary:'witness',accuracy:'witness'},
                        points=[{k:v for k,v in row.items() if k!='source'}])
            capture=dict(rows=[dict(identifier='fit',regime='small',horizons=2,
                variant='tensor',repeats=5,median_block_ms=4,median_ms_per_generated_step=2)])
            values={source:report,summary:capture,accuracy:{}}
            load=lambda p,expected=None:values[str(Path(p).relative_to(lab))]
            model=dict(evaluation_source='../../'+accuracy)
            with self.assertRaisesRegex(ValueError,'No unique complete-block'):
                bind_temporal_timing(row,model,base,lab,load,{})
            capture['rows'][0]['variant']='full'
            bound=bind_temporal_timing(row,model,base,lab,load,{})
            self.assertEqual(bound['execution_variant'],'full')
            self.assertEqual(bound['timing_source'],summary)
            report['reserved_scenes_consumed']=True
            with self.assertRaisesRegex(ValueError,'population differs'):
                bind_temporal_timing(row,model,base,lab,load,{})
            row['regime']='reserved_small'
            report['points']=[{k:v for k,v in row.items() if k!='source'}]
            self.assertEqual(bind_temporal_timing(row,model,base,lab,load,{})['timing_source'],summary)
            report['training_on_reserved']=True
            with self.assertRaisesRegex(ValueError,'Unqualified temporal cost'):
                bind_temporal_timing(row,model,base,lab,load,{})

    def test_pending_followup_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'full-ai-next-priorities-v1'
            path.mkdir()
            (path / 'temporal-sharing-results.json').write_text('{"status":"running"}')
            load = lambda p, expected=None: json.loads(Path(p).read_text())
            self.assertEqual(len(completed_temporal_specs(tmp, tmp, load)), 6)

    def test_completed_followup_requires_bound_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'full-ai-next-priorities-v1'
            path.mkdir()
            (path / 'temporal-sharing-results.json').write_text(
                '{"status":"completed","source_evidence":{}}')
            load = lambda p, expected=None: json.loads(Path(p).read_text())
            with self.assertRaisesRegex(ValueError, 'Unbound temporal follow-up'):
                completed_temporal_specs(tmp, tmp, load)

    def test_snapshot_models_match_recorded_evaluations_and_timings(self):
        data = json.loads(DATA.search((ROOT / 'cohort-progress.html').read_text())[2])
        study = data['full_ai_study']
        models = {m['label']: m for m in study['temporal_models']+study.get('temporal_executions',[])}
        self.assertTrue((NEW_TIMED | NEW_UNTIMED) <= models.keys())
        fits=[m for m in study['temporal_models'] if m.get('record_kind','fit')=='fit']
        self.assertEqual(len(fits),len({m['weights_sha256'] for m in fits}))
        for execution in study['temporal_executions']:
            self.assertEqual(execution['record_kind'],'execution')
            self.assertEqual(execution['optimizer_updates'],0)
            self.assertEqual(execution['weights_sha256'],models[execution['parent_label']]['weights_sha256'])
        self.assertEqual(len({m['bank_sha256'] for m in models.values() if m.get('evaluation_format')!='sampler-state-rows'}), 1)
        timed = {r['identifier'] for r in study['temporal_timing']}
        self.assertTrue(NEW_TIMED <= timed)
        self.assertFalse(NEW_UNTIMED & timed)
        for model in models.values():
            if model.get('evaluation_format')=='sampler-state-rows':
                continue  # The new format has per-state checks in test_full_ai_tight_chart.
            value = json.loads((ROOT / model['evaluation_source']).read_text())
            self.assertEqual(model['weights_sha256'], value['model']['weights_sha256'])
            rows = [r for r in study['temporal'] if r['label'] == model['label'] and r.get('population','development')=='development']
            self.assertEqual(len(rows), 8)
            for row in rows:
                scenes = [s for s in value['scenes']
                          if ('small' if max(s['counts_xyz']) <= 16 else '32cubed') == row['regime']]
                index = ['single_block', 'rollout_12', 'rollout_48', 'rollout_240'].index(row['interval'])
                self.assertEqual(row['scenes'], len(scenes))
                for field in ('equation', 'pressure', 'velocity', 'tracer'):
                    self.assertEqual(row['field_worst'][field],
                                     max(s['summaries'][index]['worst'][field] for s in scenes))
                self.assertEqual(row['accuracy_pass'],
                                 all(s['summaries'][index]['all_accuracy_criteria_pass'] for s in scenes))
        for label in ['K=2 pressure expert','K=2 decoder']:
            self.assertIn('inherited', models[label]['training_scope'])
            self.assertGreater(models[label]['parameters'], models[label]['trainable_parameters'])
            self.assertEqual(models[label]['pressure_expert_calls_per_block'],1)
        self.assertTrue(any(r['label']=='K=2 pressure expert' and not r['complete'] for r in study['temporal']))
        parent=models['K=2 coupled']
        self.assertEqual(parent['optimizer_updates'],10000)
        self.assertEqual(parent['pressure_expert_calls_per_block'],2)
        for label in ['K=2 continuation control','K=2 provisional supervision']:
            model=models[label]
            self.assertEqual(model['optimizer_updates'],4000)
            self.assertEqual(model['inherited_full_state_updates'],10000)
            self.assertEqual(model['parent_weights_sha256'],parent['weights_sha256'])
        displacement=models['K=2 displacement transport']
        self.assertEqual(displacement['optimizer_updates'],10000)
        self.assertEqual(displacement['inherited_point_sampling']['parameters'],972)
        self.assertEqual(displacement['inherited_point_sampling']['optimizer_updates'],1000)
        for label in ['K=2 coupled','K=2 provisional supervision']:
            captures={t['timing_source'] for t in study['temporal_timing'] if t['identifier']==label}
            self.assertTrue(any('temporal-provisional-timing-02' in c for c in captures))
            self.assertTrue(any('temporal-displacement-timing-01' in c for c in captures))
        identities=[(t['identifier'],t['regime'],t['interval'],t['timing_source'],t.get('raw_variant')) for t in study['temporal_timing']]
        self.assertEqual(len(identities),len(set(identities)))
        for timing in study['temporal_timing']:
            self.assertAlmostEqual(timing['median_ms'] * models[timing['identifier']]['horizons'],
                                   timing['full_block_ms'], places=12)
            self.assertTrue((ROOT / timing['timing_source']).exists())
        for k,updates in [(1,20000),(2,10000),(3,6667),(4,5000)]:
            fit=models[f'K={k} fixed viscosity']
            self.assertEqual(fit['optimizer_updates'],updates)
            self.assertEqual(fit['optimizer_work']['predicted_physical_states'],36000)
            self.assertIn(f'K={k} fixed viscosity · fast',timed)
        replica=models['K=1 fixed viscosity seed 2']
        self.assertNotEqual(replica['weights_sha256'],models['K=1 fixed viscosity']['weights_sha256'])
        self.assertNotIn(replica['label'],timed)
        composed=models['K=2 calibrated viscosity']
        self.assertEqual(composed['record_kind'],'composition')
        self.assertEqual(composed['optimizer_updates'],0)
        self.assertEqual(composed['component_optimizer_updates'],1000)
        for child,parent in [('K=2 viscosity · float32 map','K=2 calibrated viscosity'),
                             ('K=2 fixed viscosity · float32 map','K=2 fixed viscosity')]:
            self.assertNotEqual(models[child]['evaluation_source'],models[parent]['evaluation_source'])
            self.assertEqual(models[child]['weights_sha256'],models[parent]['weights_sha256'])
            self.assertEqual(models[child]['accuracy_inheritance'],'actual execution evaluation')
            own=next(r for r in study['temporal'] if r['label']==child and r['regime']=='small' and r['interval']=='single_block')
            prior=next(r for r in study['temporal'] if r['label']==parent and r['regime']=='small' and r['interval']=='single_block')
            self.assertNotEqual(own['field_worst'],prior['field_worst'])
        components = {c['name']: c for c in study['causal_components']}
        factored = components['b1-factored-request']
        self.assertEqual(factored['parameters'], 4385)
        self.assertEqual(factored['optimizer_updates'], 11000)
        self.assertTrue(factored['component_only'])
        self.assertFalse(factored['own_history_rollout'])
        self.assertFalse(any(c['weights_sha256'] in {m['weights_sha256'] for m in models.values()}
                             for c in data['cohort_points']))

    def test_replica_policies_and_reserved_populations_remain_separate(self):
        study=json.loads(DATA.search((ROOT/'cohort-progress.html').read_text())[2])['full_ai_study']
        models={m['label']:m for m in study['temporal_models']+study['temporal_executions']}
        for k in (1,2,4):
            for seed in (2,3):
                fit=models[f'K={k} fixed viscosity seed {seed}']
                policy=models[fit['label']+(' · float32 map' if k==2 else ' · float64 policy')]
                self.assertNotEqual(fit['weights_sha256'],models[f'K={k} fixed viscosity']['weights_sha256'])
                self.assertEqual(fit['weights_sha256'],policy['weights_sha256'])
                self.assertEqual(policy['optimizer_updates'],0)
                if k==2:
                    self.assertNotEqual(fit['evaluation_source'],policy['evaluation_source'])
                    self.assertEqual(policy['accuracy_inheritance'],'actual execution evaluation')
        reserved=[r for r in study['temporal'] if r.get('population')=='reserved']
        self.assertEqual(len(reserved),80)
        self.assertEqual(len({r['label'] for r in reserved}),10)
        self.assertTrue(any(not r['accuracy_pass'] for r in reserved))
        for row in reserved:
            model=models[row['label']]
            value=json.loads((ROOT/row['source']).read_text())
            self.assertNotEqual(row['bank_sha256'],model['bank_sha256'])
            self.assertEqual(row['bank_sha256'],value['bank_sha256'])
            self.assertEqual(value['optimizer_updates'],0)
            self.assertFalse(value['model_promotion'])
            self.assertTrue(all(s['split']=='reserved' for s in value['scenes']))
            scenes=[s for s in value['scenes'] if ('small' if max(s['counts_xyz'])<=16 else '32cubed')==row['regime']]
            interval=['single_block','rollout_12','rollout_48','rollout_240'].index(row['interval'])
            for field,error in row['field_worst'].items():
                self.assertEqual(error,max(s['summaries'][interval]['worst'][field] for s in scenes))
            timings=[t for t in study['temporal_timing'] if t.get('population')=='reserved'
                     and t['identifier']==row['label'] and t['regime']==row['regime'] and t['interval']==row['interval']]
            self.assertEqual(len(timings),1)
            timing=timings[0]
            capture=json.loads((ROOT/timing['timing_source']).read_text())
            self.assertEqual(capture['bank_sha256'],row['bank_sha256'])
            self.assertEqual(timing['accuracy_source'],row['source'])
            self.assertEqual(timing['field_worst'],row['field_worst'])
            self.assertEqual(timing['accuracy_pass'],row['accuracy_pass'])


if __name__ == '__main__':
    unittest.main()
