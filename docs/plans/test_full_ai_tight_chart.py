"""Recorded sampler coverage, horizon identity and timing admission invariants."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import statistics
import unittest

from dashboard_build import DATA
from full_ai_tight_chart import sampler_accuracy_rows, STUDY

ROOT=Path(__file__).resolve().parent


class SamplerChartTests(unittest.TestCase):
    def test_sampler_intervals_reject_missing_or_duplicate_states(self):
        rows=[]
        for counts in ([16,16,16],[32,32,32]):
            for step in range(1,241):
                rows.append(dict(identifier=str(counts),counts_xyz=counts,horizons=1,start=0,step=step,
                    metrics=dict(relative={f:step*1e-8 for f in ['equation','pressure','velocity','tracer']},all_accuracy_criteria_pass=True)))
        value=dict(own_history_without_teacher_resets=True,rows=rows)
        result=sampler_accuracy_rows(value,'sampler',1,'record','bank')
        self.assertAlmostEqual(result[0]['field_worst']['equation'],12e-8)
        self.assertEqual(result[2]['scored_states'],240)
        bad=deepcopy(value);bad['rows'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete or duplicated'):
            sampler_accuracy_rows(bad,'sampler',1,'record','bank')
        bad=deepcopy(value);bad['rows'].append(bad['rows'][0])
        with self.assertRaisesRegex(ValueError,'Incomplete or duplicated'):
            sampler_accuracy_rows(bad,'sampler',1,'record','bank')

    def test_recorded_sampler_variants_preserve_all_fields_and_populations(self):
        study=json.loads(DATA.search((ROOT/'cohort-progress.html').read_text())[2])['full_ai_study']
        records=[m for m in study['temporal_models']+study['temporal_executions'] if m.get('study_id')==STUDY]
        fits=[m for m in records if m['record_kind']=='fit']
        self.assertEqual(len(fits),1)
        self.assertEqual(fits[0]['optimizer_updates'],440)
        self.assertEqual(fits[0]['trainable_parameters'],48)
        models={m['label']:m for m in records}
        for k in (1,2,3,4):
            self.assertEqual(models[f'K={k} recurrent sampler']['weights_sha256'],fits[0]['weights_sha256'])
        self.assertEqual(len(study['tight_components']),4)
        self.assertTrue(all(c['component_only'] for c in study['tight_components']))
        self.assertEqual([c['optimizer_updates'] for c in study['tight_components']],[800,80,80,80])
        cache={}
        def read(source):
            if source not in cache:cache[source]=json.loads((ROOT/source).read_text())
            return cache[source]
        for model in models.values():
            self.assertEqual(model.get('execution_inventory',model['learned_inventory']),
                             read(model['evaluation_source'])['learned_inventory'])
        rows=[r for r in study['temporal'] if r.get('study_id')==STUDY]
        self.assertEqual(len([r for r in rows if r['population']=='sampler_fresh']),32)
        for row in rows:
            source=read(row['source'])
            selected=[r for r in source['rows'] if r['horizons']==row['horizons'] and ('small' if max(r['counts_xyz'])<=16 else '32cubed')==row['regime']]
            if row.get('scene_identifiers'):
                selected=[r for r in selected if r['identifier'] in row['scene_identifiers']]
            if row['interval']!='single_block':
                steps=int(row['interval'].split('_')[1]);selected=[r for r in selected if r['step']<=steps]
            self.assertEqual(row['scored_states'],len(selected))
            self.assertEqual(row['scenes'],len({r['identifier'] for r in selected}))
            self.assertEqual(row['accuracy_limit'],1e-5)
            self.assertEqual(row['population'] in ('sampler_fresh','decoder_fresh','history_fresh'),source['fresh_holdout_consumed'])
            for field,error in row['field_worst'].items():
                self.assertEqual(error,max(r['metrics']['relative'][field] for r in selected))
            self.assertEqual(row['accuracy_pass'],all(r['metrics']['all_accuracy_criteria_pass'] for r in selected))
        self.assertFalse(any(r['interval']=='single_block' for r in rows if 'refined pressure' in r['label']))
        self.assertTrue(any(not r['accuracy_pass'] for r in rows if r['population']=='sampler_fresh' and r['interval']=='rollout_240'))
        times=[t for t in study['temporal_timing'] if t.get('study_id')==STUDY]
        self.assertEqual(len(times),816)
        for timing in times:
            self.assertNotIn('refined pressure',timing['identifier'])
            capture=read(timing['timing_source'])
            samples=[ms for r in capture['rows'] if r['horizons']==timing['horizons'] and ('small' if max(r['counts_xyz'])<=16 else '32cubed')==timing['regime'] for ms in (r['raw_complete_block_ms'][timing['raw_variant']] if timing.get('raw_variant') else r['raw_complete_block_ms'])]
            self.assertAlmostEqual(timing['full_block_ms'],statistics.median(samples),places=12)
            self.assertAlmostEqual(timing['median_ms']*timing['horizons'],timing['full_block_ms'],places=12)
            if not timing.get('raw_variant'):
                self.assertEqual(timing['population']=='sampler_fresh',capture['fresh_holdout_consumed'])
            else:
                self.assertTrue(capture['all_intermediate_outputs_bitwise_equal'])
                self.assertIn(timing['core_count'],[1,2,4])
            self.assertTrue(any(r['label']==timing['identifier'] and r['source']==timing['accuracy_source']
                and r['interval']==timing['interval'] and r['regime']==timing['regime'] and r['population']==timing['population']
                and r['field_worst']==timing['field_worst'] for r in rows))

    def test_decoder_confirmation_and_consumed_repairs_are_not_pooled(self):
        study=json.loads(DATA.search((ROOT/'cohort-progress.html').read_text())[2])['full_ai_study']
        fresh=[r for r in study['temporal'] if r.get('population')=='decoder_fresh']
        subset=[r for r in study['temporal'] if r.get('population')=='decoder_consumed']
        self.assertEqual(len(fresh),32)
        self.assertEqual(len(subset),42)
        self.assertEqual(len({r['label'] for r in subset}),7)
        self.assertTrue(all(r['scenes']==1 and r['interval']!='single_block' for r in subset))
        self.assertFalse(any(t.get('population')=='decoder_consumed' for t in study['temporal_timing']))
        self.assertTrue(set(r['bank_sha256'] for r in fresh).isdisjoint(r['bank_sha256'] for r in subset))
        paired=[t for t in study['temporal_timing'] if t.get('raw_variant')]
        self.assertEqual(len(paired),720)
        self.assertEqual({t['core_count'] for t in paired},{1,2,4})
        for capture in [c for c in study['temporal_timing_captures'] if c.get('id')]:
            self.assertEqual(capture['id'].split('#')[0],capture['source'])
            value=json.loads((ROOT/capture['source']).read_text())
            protocol=ROOT/STUDY/(Path(capture['source']).name.replace('-cost-','-cost-protocol-'))
            reg=json.loads(protocol.read_text())
            self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(),value['protocol_sha256'])
            bank=hashlib.sha256((ROOT.parent.parent/reg['bank']).read_bytes()).hexdigest()
            self.assertTrue(all(r['bank_sha256']==bank for r in study['temporal']
                if r['label']==next(t['identifier'] for t in paired if t['capture_id']==capture['id'])
                and r.get('population')==capture['population']))
            self.assertEqual(capture['allocated_cpus'],value['cores'])
        self.assertEqual(len(study['temporal_applications']),3)
        for application in study['temporal_applications']:
            self.assertEqual(len(application['groups']),6)
            self.assertTrue(all(g['completed']==g['episodes']==6 for g in application['groups']))


if __name__=='__main__':unittest.main()
