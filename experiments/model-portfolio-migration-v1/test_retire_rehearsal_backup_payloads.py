"""Retirement decision fixtures; actual native restore has separate real proof."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import retire_rehearsal_backup_payloads as subject

sys.path.insert(0,str(subject.SOURCE))
from backup_metadata import tree_manifest


class RetirementDecisions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/'runs/model-portfolio-migration-v1'
        self.reports = self.root/'docs/plans/model-portfolio-migration-v1/reports'
        self.reports.mkdir(parents=True)
        self.enterContext(patch.multiple(subject,ROOT=self.root,RUN=self.run,REPORTS=self.reports))
        self.source = self.run/'isolated-history-metadata-restore-01/private/extraction-exchange-native-history-extraction-rehearsal-03/stopped-copy'
        self.keeper = self.run/'native-backup-compaction-03/private/backup'
        (self.source/'catalog/nested').mkdir(parents=True)
        (self.source/'catalog/nested/payload.bin').write_bytes(b'Unique fixture payload remains in keeper')
        (self.source/'catalog/nested').chmod(0o555)
        (self.source/'metadata.json').write_bytes(b'{"private_fixture":"original opaque SQL"}\n')
        receipt={'metadata_sha256':subject.sha(self.source/'metadata.json'),
                 'files':tree_manifest(self.source/'catalog')}
        (self.source/'backup-receipt.json').write_text(json.dumps(receipt)+'\n')
        shutil.copytree(self.source,self.keeper)
        self.before=tree_manifest(self.source/'catalog')
        self.proof=dict(format='rheon_native_backup_compaction_restore_v1',attempt=3,
            status='qualified_compact_backup_and_native_restore',compact_copy_verified=True,
            native_restore_verified=True,postgres_stopped=True,temporary_restore_removed=True,
            source=str(self.source.relative_to(self.root)),compact_backup=str(self.keeper.relative_to(self.root)),
            source_receipt_sha256=subject.sha(self.source/'backup-receipt.json'),
            source_sha256={name:subject.sha(subject.SOURCE/name) for name in ('backup_metadata.py','metadata.py')})
        self.proof_path=self.reports/'native-backup-compaction-03.json'
        self.proof_path.write_text(json.dumps(self.proof)+'\n')
        self.report=self.reports/'retirement.json'

    def test_pending_restore_cannot_authorize_removal(self):
        self.proof['native_restore_verified']=False
        self.proof_path.write_text(json.dumps(self.proof)+'\n')
        with self.assertRaisesRegex(ValueError,'Completed compact copy, native restore'):
            subject.retire(3,self.report)
        self.assertEqual(tree_manifest(self.source/'catalog'),self.before)
        self.assertFalse((self.source/'COMPACTED-BACKUP.json').exists())

    def test_changed_keeper_payload_refuses_before_marker_or_deletion(self):
        (self.keeper/'catalog/nested/payload.bin').write_bytes(b'Changed keeper')
        with self.assertRaisesRegex(ValueError,'Every original and retained payload'):
            subject.retire(3,self.report)
        self.assertEqual(tree_manifest(self.source/'catalog'),self.before)
        self.assertFalse((self.source/'COMPACTED-BACKUP.json').exists())

    def test_other_source_root_is_never_accepted_from_proof(self):
        self.proof['source']='unrelated/original-live-catalog'
        self.proof_path.write_text(json.dumps(self.proof)+'\n')
        with self.assertRaisesRegex(ValueError,'Proof paths differ from fixed scope'):
            subject.retire(3,self.report)
        self.assertEqual(tree_manifest(self.source/'catalog'),self.before)

    def test_verified_copy_retires_payload_only_and_preserves_both_headers(self):
        headers={name:subject.sha(self.source/name) for name in ('metadata.json','backup-receipt.json')}
        subject.retire(3,self.report)
        self.assertFalse((self.source/'catalog').exists())
        self.assertEqual(tree_manifest(self.keeper/'catalog'),self.before)
        self.assertEqual(headers,{name:subject.sha(self.source/name) for name in headers})
        marker=json.loads((self.source/'COMPACTED-BACKUP.json').read_text())
        self.assertEqual(marker['status'],'duplicate_payloads_retired_use_complete_keeper')
        result=json.loads(self.report.read_text())
        self.assertTrue(result['completed'])
        self.assertFalse(result['unique_weights_deleted'])


if __name__=='__main__':
    unittest.main()
