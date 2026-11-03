"""Bind native note/HTTP, browser and decoder checks to one source proposal."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1/reports'
SOURCE = RUN/'sources/Tuldok-authored-model-portfolio'
OUTPUT = RUN/'authored-model-portfolio-source-checks-01'
REPORT = DOC/'authored-model-portfolio-source-qualification-01.json'
FILES = ['app.py','static/project-files.css','static/project-files.html','static/project-files.js','static/project-model-portfolio.js',
    'docs/contracts/project-model-portfolio.md','tests/model_portfolio_fixture.cjs','tests/test_project_model_portfolio.cjs',
    'tests/browser_model_portfolio.cjs','tests/test_project_model_portfolio_http.py',
    'tests/test_workspace_project_files.cjs','tests/test_workspace_project_files_m4.cjs']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def qualify():
    require(not OUTPUT.exists() and not REPORT.exists(),'Fresh proposal qualification outputs required')
    files = set(subprocess.check_output(['git','-C',str(SOURCE),'diff','--name-only'],text=True).splitlines())
    files |= set(subprocess.check_output(['git','-C',str(SOURCE),'ls-files','--others','--exclude-standard'],text=True).splitlines())
    require(files == set(FILES),'Proposal differs from its explicit owned write set')
    before = {name:sha(SOURCE/name) for name in FILES}
    browser_path = RUN/'authored-portfolio-component-browser-04/report.json'
    browser = json.loads(browser_path.read_bytes())
    require(browser['status'] == 'passed_authored_portfolio_real_browser_component'
            and len(browser['checks']) == 19 and not browser['errors'] and browser['browser_stopped']
            and browser['source_stable_through_tests'] and browser['source_sha256'] == before,
            'Real browser/source/lifecycle evidence differs')
    sql_path = DOC/'authored-model-portfolio-SQL-01.json'
    sql = json.loads(sql_path.read_bytes())
    log_path = RUN/'authored-model-portfolio-SQL-01/tests.log'
    require(sql['status'] == 'passed_disposable_native_project_SQL_fixture'
            and sql['postgres_stopped'] and sql['source_stable_through_tests']
            and sql['cold_role_provisioned'] and sql['cold_role_source_CONNECT_denied']
            and sha(log_path) == sql['tests_log_sha256']
            and re.search(r'Ran 21 tests in [\d.]+s\n\nOK\s*$',log_path.read_text()),
            'Real21-test native/HTTP/cold-role evidence required')
    require(all(sha(SOURCE/name) == digest for name,digest in sql['source_sha256'].items()),
            'Native-tested source changed')
    guide_path = DOC/'authored-model-portfolio-note-qualification-01.json'
    guide = json.loads(guide_path.read_bytes())
    require(guide['fits'] == 205 and guide['source_reviewed'] == 69 and guide['exact_review_paragraphs'] == 414,
            'Complete native guide binding required')
    require(all(sha(ROOT/item['path']) == item['sha256'] for item in guide['evidence']),
            'Guide/decoder qualification binding changed')
    OUTPUT.mkdir()
    checks = []
    for name,script in [('document','test_project_model_portfolio.cjs'),('host','test_workspace_project_files.cjs'),('host-m4','test_workspace_project_files_m4.cjs')]:
        result = subprocess.run(['node','tests/'+script],cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
        path = OUTPUT/(name+'.log');path.write_text(result.stdout)
        require(result.returncode == 0,'Current frozen-source controller/document check failed')
        checks.append(dict(command=['node','tests/'+script],log=str(path.relative_to(ROOT)),log_sha256=sha(path),returncode=0))
    require(before == {name:sha(SOURCE/name) for name in FILES},'Proposal changed during final qualification')
    value = dict(status='authored_portfolio_source_qualified_for_serial_native_rollout',created_unix=time.time(),
        base_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        source=str(SOURCE.relative_to(ROOT)),source_sha256=before,native_SQL_HTTP_checks=21,
        document_contract_checks=31,real_browser_component_checks=19,both_existing_host_controller_suites_passed=True,
        all_source_files_stable=True,source_only_no_schema_or_SQL_migration=True,original_Trial_history_unchanged=True,
        current_project_mutated=False,actual_full_project_UI_accepted=False,installed=False,
        full_portfolio_complete=False,new_model_forwards=0,payload_copies=0,checks=checks,
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in (browser_path,sql_path,log_path,guide_path)],
        producer_sha256=sha(Path(__file__)))
    REPORT.write_text(json.dumps(value,indent=2)+'\n')
    print(value['status'])


if __name__ == '__main__':
    qualify()
