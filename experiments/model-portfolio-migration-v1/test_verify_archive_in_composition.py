"""Literal original observations and complete alternate-proof envelope binding."""
import copy
import hashlib
import json
import unittest

from verify_archive_in_composition import member, canonical, original_fact_checks, ORIGIN


class ArchiveCompositionTests(unittest.TestCase):
    def test_alternate_proof_uses_complete_original_document(self):
        doc=dict(run=dict(id='original',error='retained failure'),admission_proof=dict(grant='old'))
        envelope=dict(family='run_history',id='old-run',document=doc)
        membership=dict(records=dict(run_history={'old-run':hashlib.sha256(canonical(envelope)).hexdigest()}),
                        run_observations={'old-run':'alternate'})
        retained={('run_history','old-run'):{**doc,'admission_proof':dict(grant='other')},
                  ('run_observation','alternate'):dict(observation=doc)}
        self.assertEqual(member(retained,'run_history','old-run',membership),doc)
        retained[('run_observation','alternate')]['observation']=copy.deepcopy(doc)
        retained[('run_observation','alternate')]['observation']['admission_proof']['grant']='changed'
        with self.assertRaisesRegex(ValueError,'envelope binding'):member(retained,'run_history','old-run',membership)

    def original(self):
        run=dict(id='run-one',error='original failure',worker='local-only')
        snapshot=dict(tables=dict(workspace_runs=dict(columns=['id','document'],rows=[['run-one',json.dumps(run)]]),
                                  workspace_projects=dict(columns=['id','document'],rows=[])))
        records={('run_history',ORIGIN+'.run-one'):dict(origin_catalog_id=ORIGIN,run={k:v for k,v in run.items() if k!='worker'})}
        return snapshot,records

    def test_failed_history_values_are_compared_to_original_and_only_local_fields_excluded(self):
        original,records=self.original();checks,counts,_=original_fact_checks(original,records,all_archive=True)
        self.assertEqual(len(checks),1);self.assertEqual(counts['run_history'],1)
        records[('run_history',ORIGIN+'.run-one')]['run']['error']=None
        with self.assertRaisesRegex(ValueError,'observation changed'):original_fact_checks(original,records,all_archive=True)

    def test_missing_original_history_cannot_pass_as_empty_population(self):
        original,records=self.original();records.clear()
        with self.assertRaisesRegex(ValueError,'membership differs'):original_fact_checks(original,records,all_archive=True)


if __name__=='__main__':unittest.main()
