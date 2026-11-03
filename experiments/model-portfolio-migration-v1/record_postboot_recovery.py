"""Record completed recovery from the original Open and custody/UI evidence."""
from pathlib import Path
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC/'reports/native-server-prior-boot-recovery-02.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original completed recovery report')
    base = RUN/'native-server-reboot-recovery-01'
    opened = read(base/'open-status.json')
    before, after = [read(base/f'{stage}-custody.json') for stage in ('before','after')]
    browser_path = RUN/'native-postboot-project-browser-01/report.json'
    browser = read(browser_path)
    require(opened['status'] == 'native_open_acknowledged' and opened['project_left_open'],
            'Original native Open acknowledgement required')
    require(before['facts'] == after['facts'] and before['canonical_bodies_freshly_hashed']
            and after['canonical_bodies_freshly_hashed'], 'Exact same-boot custody required')
    require(opened['checkpoint'] == after['facts']['checkpoint'], 'Open/custody checkpoint differs')
    require(browser['status'] == 'passed_native_project_UI_actions' and not browser['errors']
            and browser['browser_stopped'] and len(browser['steps']) == 28,
            'Completed original recovered-project UI check required')
    requests = browser['api_requests']
    require(len(requests) == 8 and all(r['status'] == 200 and r['complete'] for r in requests),
            'All original UI reads must succeed')
    screenshots = [s['name'] for s in browser['steps'] if s['kind'] == 'screenshot']
    require(set(screenshots) == {'recovered-guide2.png','recovered-C107-summary.png'}
            and not browser['page']['error'] and opened['checkpoint'] in browser['page']['body'],
            'Recovered guide/model actions and saved project page required')
    screenshots.append('native-project.png')  # Automatic final-page capture by the browser owner.
    prior = read(DOC/'reports/native-server-prior-boot-recovery-01.json')
    value = dict(prior)
    value.update(status='same_catalog_and_project_recovered_Open_custody_and_actual_UI_passed',
        created_unix=time.time(),native_Open_status=opened['status'],
        native_Open_seconds=opened['native_call_wall_seconds'],
        owner_wrapper_Open_seconds=opened['owner_wrapper_wall_seconds'],
        post_Open_custody_pending=False,actual_recovered_project_UI_pending=False,
        postboot_after_Open_custody=after['facts'],actual_UI_actions=28,actual_UI_reads=8,
        all_UI_reads_HTTP200=True,project_left_open=True,near_instant_Open_achieved=False,
        observation='Original request completed; no replay. Device-inclusive payload identities compared within this boot. All canonical bodies freshly hashed; payload bodies were not rehashed.',
        producer=binding(Path(__file__)),evidence=[binding(p) for p in [
            DOC/'reports/native-server-prior-boot-recovery-01.json',base/'open-status.json',
            base/'before-custody.json',base/'after-custody.json',browser_path,
            *[browser_path.parent/name for name in screenshots]]])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Original Open/custody/28 UI actions recorded; near-instant opening remains unresolved.')


if __name__ == '__main__':
    run()
