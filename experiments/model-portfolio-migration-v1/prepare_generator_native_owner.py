"""Adapt the accepted native note owner/audit to the qualified40-note batch."""
import ast
from pathlib import Path
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OLD = RUN/'method-review-native-adoption-12'
OUTPUT = RUN/'method-review-native-adoption-13'
REPORT = DOC/'reports/generator-native-owner-preparation-01.json'


def run():
    require(not OUTPUT.exists() and not REPORT.exists(), 'Preserve original40note native owner preparation')
    accepted = read(DOC/'reports/feedback-native-owner-preparation-01.json')
    originals = [binding(OLD/n) for n in ('run.py','audit.py')]
    require(originals == accepted['current'], 'Previously accepted native owners changed')
    qualification = read(DOC/'reports/generator-review-native-preparation-01.json')
    require(qualification['notes'] == 40 and qualification['reviewed_total'] == 116,
            'Independent complete note/guide qualification required')
    counts = {}

    def replace(text, old, new):
        count = text.count(old)
        require(count > 0, 'Required accepted-owner adaptation target absent: '+old)
        counts[old] = count
        return text.replace(old,new)

    owner = (OLD/'run.py').read_text()
    replacements = [
        ('thirteen','forty'),('with ten source reviews','with thirty-seven source reviews'),
        ('native-feedback-review-publication-01','native-generator-review-publication-01'),
        ('native-authored-portfolio-browser-01','native-postboot-project-browser-01'),
        ('project-read-performance-live-integration-06','native-server-reboot-recovery-01'),
        ('feedback-guide-revision-preparation-01.json','generator-review-native-preparation-01.json'),
        ("qualification['reviewed_total'] != 79","qualification['reviewed_total'] != 116"),
        ("len(preparation['notes']) != 13","len(preparation['notes']) != 40"),
        ("preparation['original_receipt_suffixes_preserved'] != 10","preparation['original_receipt_suffixes_preserved'] != 37"),
        ('original_receipts_preserved=10, prior_reviewed_cards_preserved=69','original_receipts_preserved=37, prior_reviewed_cards_preserved=79'),
        ('notes=13, project_id','notes=40, project_id'),
        ("value['changed_count'] != 13 or len(value['items']) != 13","value['changed_count'] != 40 or len(value['items']) != 40"),
        ('adopted_notes=13, authored_reviewed_fit_cards=10, total_reviewed_fit_cards=79',
         'adopted_notes=40, authored_reviewed_fit_cards=37, total_reviewed_fit_cards=116')]
    for old,new in replacements:
        owner = replace(owner,old,new)
    audit = (OLD/'audit.py').read_text()
    for old,new in [
        ("REPORT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/native-seventynine-method-review-adoption-01.json'",
         "REPORT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/native-116-method-review-adoption-01.json'"),
        ('native-authored-model-portfolio-adoption-01.json','native-seventynine-method-review-adoption-01.json'),
        ('native-feedback-review-publication-01','native-generator-review-publication-01'),
        ('thirteen','forty'),('Thirteen','Forty'),('len(entries) == 13','len(entries) == 40'),
        ("for fit in index['fits']) == 79","for fit in index['fits']) == 116"),
        ('Expected79native','Expected116native'),('changed_note_records=13','changed_note_records=40'),
        ('reviewed_fit_cards=79','reviewed_fit_cards=116')]:
        audit = replace(audit,old,new)
    ast.parse(owner)
    ast.parse(audit)
    OUTPUT.mkdir()
    (OUTPUT/'run.py').write_text(owner)
    (OUTPUT/'audit.py').write_text(audit)
    result = dict(status='qualified40note_owner_adapted_from_accepted13note_owner',created_unix=time.time(),
        originals=originals,current=[binding(OUTPUT/n) for n in ('run.py','audit.py')],replacement_occurrences=counts,
        expected_notes=40,new_reviewed_cards=37,total_reviewed_cards=116,guide_revision=3,
        native_requests_started=0,current_project_mutated=False,new_model_forwards=0,source_and_data_copies=0,
        producer=binding(Path(__file__)),evidence=[binding(DOC/'reports/feedback-native-owner-preparation-01.json'),
            binding(DOC/'reports/generator-review-native-preparation-01.json'),
            binding(RUN/'native-generator-review-publication-01/preparation.json')])
    REPORT.write_bytes(canonical(result)+b'\n')
    print('40-note native owner/audit prepared from exact accepted owners; no dispatch or project edit.')


if __name__ == '__main__':
    run()
