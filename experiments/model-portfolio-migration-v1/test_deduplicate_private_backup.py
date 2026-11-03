"""Content-preserving backup deduplication and real native independent copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import deduplicate_private_backup as owner
from backup_metadata import copy_tree,tree_manifest,tree_modes


class PrivateBackupDeduplication(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.backup=self.root/'backup';self.catalog=self.backup/'catalog'
        self.catalog.mkdir(parents=True)
        for name,data,mode in [('a',b'same retained model bytes'*500,0o600),('b',b'same retained model bytes'*500,0o600),
                               ('different-mode',b'same retained model bytes'*500,0o640),('unique',b'unique original weights',0o600)]:
            p=self.catalog/name;p.write_bytes(data);p.chmod(mode)
        self.metadata=b'{"original_SQL_snapshot":"retained"}\n';(self.backup/'metadata.json').write_bytes(self.metadata)
        self.files=tree_manifest(self.catalog);self.modes=tree_modes(self.catalog,self.files)
        self.receipt=json.dumps(dict(format_version=1,kind='postgresql',metadata_sha256=hashlib.sha256(self.metadata).hexdigest(),files=self.files)).encode()
        (self.backup/'backup-receipt.json').write_bytes(self.receipt)

    def verify(self):
        self.assertEqual(tree_manifest(self.catalog),self.files)
        self.assertEqual(tree_modes(self.catalog,self.files),self.modes)
        self.assertEqual((self.backup/'metadata.json').read_bytes(),self.metadata)
        self.assertEqual((self.backup/'backup-receipt.json').read_bytes(),self.receipt)

    def test_exact_private_duplicates_keep_every_path_and_native_restore_files_are_independent(self):
        result=owner.deduplicate(self.backup,self.root/'operation');self.verify()
        self.assertEqual(result['status'],'complete_private_backup_bytes_modes_paths_exact')
        self.assertEqual(result['files_relinked'],1);self.assertGreater(result['allocated_bytes_reclaimed'],0)
        self.assertEqual((self.catalog/'a').stat().st_ino,(self.catalog/'b').stat().st_ino)
        self.assertNotEqual((self.catalog/'a').stat().st_ino,(self.catalog/'different-mode').stat().st_ino)
        restored=self.root/'restored';copy_tree(self.catalog,restored,self.files)
        self.assertEqual(tree_manifest(restored),self.files)
        self.assertEqual(tree_modes(restored,self.files),self.modes)
        inodes={(p.stat().st_dev,p.stat().st_ino) for p in restored.iterdir()}
        self.assertEqual(len(inodes),4)
        for p in restored.iterdir():
            self.assertEqual(p.stat().st_nlink,1)
            self.assertNotEqual(p.stat().st_ino,(self.catalog/p.name).stat().st_ino)

    def test_changed_same_length_backup_refuses_before_any_relink(self):
        path=self.catalog/'b';before=path.stat();path.write_bytes(b'x'*before.st_size)
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'content differs'):
            owner.deduplicate(self.backup,self.root/'operation')
        self.assertNotEqual((self.catalog/'a').stat().st_ino,path.stat().st_ino)
        self.assertEqual(path.read_bytes(),b'x'*before.st_size)

    def test_external_inode_alias_is_never_used_for_private_deduplication(self):
        outside=self.root/'original-catalog-file';os.link(self.catalog/'a',outside)
        identity=(outside.stat().st_ino,outside.stat().st_nlink)
        result=owner.deduplicate(self.backup,self.root/'operation');self.verify()
        self.assertEqual(result['files_relinked'],0)
        self.assertEqual((outside.stat().st_ino,outside.stat().st_nlink),identity)
        self.assertNotEqual((self.catalog/'b').stat().st_ino,outside.stat().st_ino)

    def test_another_process_open_file_refuses_before_relink(self):
        process=subprocess.Popen([sys.executable,'-c',
            'import sys; f=open(sys.stdin.readline().strip(),"rb"); print("ready",flush=True); sys.stdin.read()'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            process.stdin.write(str(self.catalog/'a')+'\n');process.stdin.flush()
            self.assertEqual(process.stdout.readline().strip(),'ready')
            with self.assertRaisesRegex(ValueError,'another process open file'):
                owner.deduplicate(self.backup,self.root/'operation')
        finally:
            process.communicate(timeout=5)
        self.verify();self.assertNotEqual((self.catalog/'a').stat().st_ino,(self.catalog/'b').stat().st_ino)

    def test_interruption_after_atomic_replace_preserves_complete_standalone_backup(self):
        original=os.replace
        def interrupted(src,dst,*args,**kwargs):
            original(src,dst,*args,**kwargs)
            if Path(dst).parent==self.catalog:raise InterruptedError('after exact atomic replacement')
        with patch.object(owner.os,'replace',side_effect=interrupted),self.assertRaises(InterruptedError):
            owner.deduplicate(self.backup,self.root/'interrupted')
        self.verify();self.assertFalse(list(self.backup.glob('.dedup-pending-*')))
        result=owner.deduplicate(self.backup,self.root/'reconciled')
        self.assertEqual(result['status'],'complete_private_backup_bytes_modes_paths_exact')
        self.assertEqual(result['files_relinked'],0);self.verify()

if __name__=='__main__':unittest.main()
