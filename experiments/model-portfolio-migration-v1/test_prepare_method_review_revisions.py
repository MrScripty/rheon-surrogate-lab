"""Review prose must preserve all original scientific evidence bytes."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import prepare_method_review_revisions as owner
from prepare_method_review_revisions import HEADING, revised_card


class MethodReviewTextTests(unittest.TestCase):
    def setUp(self):
        self.evidence = ('```json\n{"original":"évidence","metric":1.25e-15}\n```\n\n'
                         '## Exact original receipt bytes\n\n```json\n{"receipt":"exact"}\n```\n\n'
                         '## Exact training configuration\n\n```json\n{"seed":17}\n```\n').encode('utf-8')
        self.old = b'# Original title\n\nOriginal draft summary.\n\n'+HEADING+self.evidence
        self.review = dict(fit_item_id='fit-reviewed', original_receipt={'sha256':'a'*64},
                           history={'sha256':'b'*64}, sources=[{'sha256':'c'*64}])
        for field in ('method','purpose','experiment','outcome','lesson','reproduction'):
            self.review[field] = field+' reviewed from source'

    def test_review_preserves_complete_original_suffix_including_receipt_and_configuration(self):
        new, suffix = revised_card(self.old,self.review)
        self.assertEqual(suffix,self.evidence)
        self.assertTrue(new.endswith(self.evidence))
        self.assertTrue(new.startswith(b'# Original title\n'))
        self.assertIn(b'Retrospective source review',new)
        self.assertIn(b'Original retained fit facts before this review',new)
        for field in ('method','purpose','experiment','outcome','lesson','reproduction'):
            self.assertIn(self.review[field].encode(),new)

    def test_unknown_source_structure_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Expected the original captured card structure'):
            revised_card(b'# Changed card\n',self.review)

    def test_ambiguous_original_evidence_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Expected the original captured card structure'):
            revised_card(self.old+HEADING,self.review)

    def test_incomplete_authored_review_is_rejected(self):
        self.review['lesson'] = ''
        with self.assertRaisesRegex(ValueError,'Complete authored review paragraphs required'):
            revised_card(self.old,self.review)


class ReviewBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);p=patch.object(owner,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.code=b'# captured source, not executed\n';(self.root/'model.py').write_bytes(self.code)
        self.history=b'{"captured_updates":17}\n';(self.root/'history.json').write_bytes(self.history)
        self.population=b'{"population":"train_only"}\n';(self.root/'splits.json').write_bytes(self.population)
        self.receipt=dict(closure=dict(sources={'model.py':owner.sha(self.code)}),
            outputs={'history.json':dict(sha256=owner.sha(self.history)),
                     'splits.json':dict(sha256=owner.sha(self.population))})
        self.review=dict(history=dict(path='history.json',sha256=owner.sha(self.history)),
                         sources=[dict(path='model.py',sha256=owner.sha(self.code))],
                         training_population=dict(path='splits.json',sha256=owner.sha(self.population)))

    def record(self):
        raw=json.dumps(self.receipt).encode();(self.root/'receipt.json').write_bytes(raw)
        self.review['original_receipt']=dict(path='receipt.json',sha256=owner.sha(raw));return raw

    def witnessed(self):
        raw=json.dumps(dict(sources=self.receipt['closure'].pop('sources'))).encode()
        (self.root/'witness.json').write_bytes(raw)
        self.receipt['closure']['source_witness_sha256']=owner.sha(raw)
        self.review['source_witness']=dict(path='witness.json',sha256=owner.sha(raw))

    def test_direct_and_witness_bound_receipts_preserve_the_exact_original_bytes(self):
        raw=self.record();self.assertEqual(owner.verified_review_receipt(self.review),raw)
        self.witnessed();raw=self.record();self.assertEqual(owner.verified_review_receipt(self.review),raw)

    def test_changed_witness_or_member_cannot_support_a_review(self):
        self.witnessed();self.record();(self.root/'witness.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'witness'):owner.verified_review_receipt(self.review)
        raw=json.dumps(dict(sources={'model.py':owner.sha(self.code)})).encode();(self.root/'witness.json').write_bytes(raw)
        (self.root/'model.py').write_bytes(b'# changed source')
        with self.assertRaisesRegex(ValueError,'source differs'):owner.verified_review_receipt(self.review)

    def test_population_must_be_the_exact_original_fit_output(self):
        self.record();raw=b'{"population":"other"}\n';(self.root/'splits.json').write_bytes(raw)
        self.review['training_population']['sha256']=owner.sha(raw)
        with self.assertRaisesRegex(ValueError,'Training population'):owner.verified_review_receipt(self.review)

    def test_training_input_must_match_the_actual_receipt_slot_not_only_the_review(self):
        raw=b'original teacher bytes';(self.root/'teacher.npz').write_bytes(raw)
        self.receipt['closure']['inputs']={'teacher':dict(sha256=owner.sha(raw),bytes=len(raw))}
        self.review['training_inputs']=[dict(slot='teacher',path='teacher.npz',sha256=owner.sha(raw),bytes=len(raw))]
        receipt=self.record();self.assertEqual(owner.verified_review_receipt(self.review),receipt)
        changed=b'different teacher bytes';(self.root/'teacher.npz').write_bytes(changed)
        self.review['training_inputs'][0].update(sha256=owner.sha(changed),bytes=len(changed))
        with self.assertRaisesRegex(ValueError,'Training input'):owner.verified_review_receipt(self.review)

    def test_generator_must_match_the_verified_original_input_manifest(self):
        code=b'# offline teacher generator, not executed\n';(self.root/'generate.py').write_bytes(code)
        raw=json.dumps(dict(source_sha256=owner.sha(code))).encode();(self.root/'teacher-manifest.json').write_bytes(raw)
        self.receipt['closure']['inputs']={'teacher_manifest':dict(sha256=owner.sha(raw),bytes=len(raw))}
        self.review['training_inputs']=[dict(slot='teacher_manifest',path='teacher-manifest.json',sha256=owner.sha(raw),bytes=len(raw))]
        self.review['data_generation_source']=dict(path='generate.py',sha256=owner.sha(code),manifest_slot='teacher_manifest')
        receipt=self.record();self.assertEqual(owner.verified_review_receipt(self.review),receipt)
        changed=b'# a different generator\n';(self.root/'generate.py').write_bytes(changed)
        self.review['data_generation_source']['sha256']=owner.sha(changed)
        with self.assertRaisesRegex(ValueError,'Data-generation source'):owner.verified_review_receipt(self.review)

    def preparation_subject(self):
        receipt=self.record()
        project=self.root/'projects/current';metadata=project/'.tuldok'
        metadata.mkdir(parents=True)
        runs=self.root/'runs';runs.mkdir()
        selected=self.review | dict(fit_item_id='fit-new',note_path='notes/new.md')
        selected.update({key:key+' interpreted from captured sources' for key in
                         ('method','purpose','experiment','outcome','lesson','reproduction')})
        prior=dict(path='previous-review.json',sha256='a'*64)
        index=dict(fits=[dict(item_id='fit-previous',card_path='notes/previous.md',
                              card_sha256='b'*64,method_review_status='captured_source_and_receipt_reviewed'),
                         dict(item_id='fit-new',card_path='notes/new.md',card_sha256='c'*64)],
                   sources=dict(method_reviews=prior))
        files={'notes/new.md':b'# New fit\n\n'+HEADING+receipt,
               'notes/previous.md':b'# Previously reviewed fit\n\nKeep exact.\n',
               'notes/portfolio-index.json':owner.encode(index),
               'notes/portfolio-overview.md':b'# Portfolio\n'}
        inventory=[]
        for number,(name,raw) in enumerate(files.items()):
            target=project/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            document=dict(project_id='1'*32,path=name,current=dict(text=raw.decode(),sha256=owner.sha(raw),revision=2))
            encoded=owner.encode(document);path=f'record-{number}.json';(metadata/path).write_bytes(encoded)
            inventory.append(dict(family='project_file',path=path,sha256=owner.sha(encoded),bytes=len(encoded)))
        (metadata/'project.json').write_bytes(owner.encode(dict(project_id='1'*32,checkpoint='c'*64,inventory=inventory)))
        review=self.root/'review.json'
        review.write_bytes(owner.encode(dict(rows=[selected],overview_addition='\nNew authored review context.\n')))
        return project,runs,review,index,prior

    def test_incremental_preparation_uses_pinned_project_and_preserves_prior_reviews(self):
        project,runs,review,index,prior=self.preparation_subject()
        before={str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}
        with patch.object(owner,'RUN',runs):
            owner.prepare(runs/'prepared',review,project=project,checkpoint='c'*64)
        after={str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        prepared=json.loads((runs/'prepared/notes/portfolio-index.json').read_bytes())
        self.assertEqual(prepared['fits'][0],index['fits'][0])
        self.assertEqual(prepared['fits'][1]['method_review_status'],'captured_source_and_receipt_reviewed')
        current=dict(path='review.json',sha256=owner.sha(review.read_bytes()))
        self.assertEqual(prepared['sources']['method_review_batches'],[prior,current])
        self.assertEqual(prepared['sources']['method_reviews'],current)
        result=json.loads((runs/'prepared/preparation.json').read_bytes())
        self.assertEqual(result['source_checkpoint'],'c'*64)
        self.assertEqual({row['path'] for row in result['notes']},
                         {'notes/new.md','notes/portfolio-index.json','notes/portfolio-overview.md'})

    def test_wrong_pinned_checkpoint_refuses_before_preparation_output(self):
        project,runs,review,_index,_prior=self.preparation_subject()
        with patch.object(owner,'RUN',runs),self.assertRaisesRegex(ValueError,'checkpoint'):
            owner.prepare(runs/'prepared',review,project=project,checkpoint='d'*64)
        self.assertFalse((runs/'prepared').exists())

    def test_incremental_preparation_refuses_to_replace_a_prior_review(self):
        project,runs,review,_index,_prior=self.preparation_subject()
        document=json.loads(review.read_bytes())
        document['rows'][0].update(fit_item_id='fit-previous',note_path='notes/previous.md')
        review.write_bytes(owner.encode(document))
        previous=(project/'notes/previous.md').read_bytes()
        with patch.object(owner,'RUN',runs),self.assertRaisesRegex(ValueError,'already reviewed'):
            owner.prepare(runs/'prepared',review,project=project,checkpoint='c'*64)
        self.assertFalse((runs/'prepared').exists())
        self.assertEqual((project/'notes/previous.md').read_bytes(),previous)


if __name__ == '__main__':
    unittest.main()
