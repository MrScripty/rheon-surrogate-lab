"""Check physical-time/case identity and native scalar provenance on tiny data."""
import copy
import unittest

import prepare_full_state_measurements as subject


class FullStatePacketTests(unittest.TestCase):
    def row(self):
        return dict(identifier='scene-a',counts_xyz=[16,32,48],cells_per_metre=100,split='development',
            start=32,K=4,lead=3,teacher_substeps=8,student_stages=4,actual_internal_advances_per_block=16,
            all_accuracy_pass=False,metric_error=None,nonfinite_counts=dict(pressure=0,vx=0,vy=0,vz=0,tracer=0),
            metrics=dict(relative=dict(pressure=.2,velocity=.3,tracer=.4,equation=.5),
                         absolute_rms=dict(pressure=.6,velocity=.7,tracer=.8)))

    def selection(self):
        return dict(counts_xyz=[16,32,48],cells_per_metre=100,reference_substeps=8,output_dt_s=1/60,
            receipt='receipt.json',receipt_sha256='1'*64,profile='pulsing',motion='stationary',
            states=[dict(step=n,time_s=n/60,path=f'state-{n}.npz',sha256=str(i)*64) for i,n in enumerate((32,35),2)])

    def test_density_domain_and_physical_horizon_are_separate(self):
        c=subject.case(self.row(),self.selection())
        self.assertEqual(c['dimensions']['cells_per_metre'],100)
        self.assertEqual([c['dimensions'][k] for k in ('grid_x','grid_y','grid_z')],[16,32,48])
        self.assertEqual(c['dimensions']['block_duration_s'],4/60)
        self.assertEqual(c['dimensions']['elapsed_s'],3/60)
        altered=self.selection();altered['states'][1]['sha256']='f'*64
        self.assertNotEqual(c['id'],subject.case(self.row(),altered)['id'])
        altered=self.selection();altered['cells_per_metre']=50
        with self.assertRaisesRegex(ValueError,'density'):subject.case(self.row(),altered)

    def test_temporal_case_does_not_change_when_execution_saves_internal_advances(self):
        r=dict(self.row(),start_fine_step=24,output_steps_per_s=60,output_dt_s=1/60,
               elapsed_s=3/60,block_duration_s=4/60,mode='fine_reference',groups_per_output=[1,1,1,1],actual_internal_advances=16)
        c=subject.case(r,temporal=True)
        self.assertEqual(c['dimensions']['reference_substeps'],4)
        r.update(mode='single_pass',groups_per_output=[4],actual_internal_advances=4)
        self.assertEqual(subject.case(r,temporal=True),c)
        self.assertEqual(subject.observation_work(r,True)['advances_block'],4)
        self.assertEqual(c['dimensions']['block_duration_s'],4/60)

    def test_selected_master_audits_do_not_claim_all_four_planned_pilots(self):
        passed='passed_completed_reference_labels_windows_and_late_replay'
        audits=[dict(status=passed,all_four_pilots_valid=False,cases=[dict(identifier=n)]) for n in ['a','b']]
        self.assertEqual(subject.audited_master_ids(audits,['a','b']),{'a','b'})
        with self.assertRaisesRegex(ValueError,'coverage'):subject.audited_master_ids(audits,['a','b','missing'])
        audits[0]['status']='failed'
        with self.assertRaisesRegex(ValueError,'qualified'):subject.audited_master_ids(audits,['a','b'])

    def test_native_table_keeps_threshold_failures_measured_and_missing_coverage_explicit(self):
        row=self.row();case=subject.case(row,self.selection());case['key']=list(subject.key(row));case['dimensions']['shared_anchor_case']=True
        norm=dict(relative_denominator_rms_floor=dict(pressure=1e-8,velocity=1e-10,tracer=1e-10))
        metrics=subject.definitions(norm,'a'*64,'b'*64)
        scope=dict(id='test-full-state',label='Tiny exact full-state fixture',population='consumed')
        context=dict(scope=scope,cases=[case],metrics=metrics,temporal=False)
        dims=dict.fromkeys(k for k,_,_,_ in subject.SUBJECT_COLUMNS);dims['inference_device']='cuda'
        data=dict(status='completed',model_identity={'name':'fixture'},latency_claim=False,rows=[row])
        subjects=[dict(id='seen',label='Seen',source_id='capture',dimensions=dims,rows={subject.key(row):(0,row)},residual=True),
                  dict(id='absent',label='Absent',source_id='capture',dimensions=dims,rows={},residual=False)]
        value=subject.packet(context,subjects,[subject.src('capture','Original tiny fixture',subject.encode(data))],metrics)
        table=subject.MeasurementTable(subject.encode(value))
        pressure=next(o for o in value['observations'] if o['subject_id']=='seen' and o['metric_id']=='pressure_relative_rms')
        self.assertEqual(pressure['reference']['pointer'],'/rows/0/metrics/relative/pressure')
        self.assertEqual(pressure['state'],'measured');self.assertEqual(pressure['value'],.2)
        self.assertFalse(pressure['dimensions']['diagnostic_pass'])
        self.assertEqual(pressure['dimensions']['advances_block'],16)
        self.assertEqual(pressure['dimensions']['residual_calls'],3)
        queried=table.query('pressure_relative_rms',filters={'cells_per_metre':100,'shared_anchor_case':True})
        self.assertEqual(queried['summary'][0]['maximum']['value'],.2)
        self.assertEqual(queried['summary'][1]['states']['unavailable'],1)
        self.assertTrue(all(o['state']=='unavailable' and o['value'] is None for o in value['observations'] if o['metric_id']=='per_output_ms'))
        wrong=copy.deepcopy(value);wrong['observations'][0]['reference']['pointer']='/rows/0/metrics/relative/velocity'
        with self.assertRaises(Exception):subject.MeasurementTable(subject.encode(wrong))


if __name__=='__main__':unittest.main()
