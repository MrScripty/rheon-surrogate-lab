"""Install only the validated, reviewable book diff; protect concurrent edits."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/book-staging'
BASELINE=ROOT/'runs/book-staging-incomplete'
BOOK=ROOT.parent.parent/'BCAI/training-your-own-models'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    validation=json.loads((STAGE/'qa/validation.json').read_text())
    assert validation['errors']==[]
    changes=[]
    for p in sorted(STAGE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        relative=p.relative_to(STAGE)
        expected=digest(BASELINE/relative)
        after=digest(p)
        if expected==after:continue
        actual=digest(BOOK/relative)
        if actual not in (expected,after):
            raise RuntimeError(f'Concurrent book edit; preserve it: {relative}')
        changes.append({'path':str(relative),'before_sha256':actual,'after_sha256':after})
    # No mutation occurs until every target has passed the concurrent-edit check.
    for change in changes:
        destination=BOOK/change['path'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(STAGE/change['path'],destination)
    receipt={'status':'installed','book':str(BOOK),'changes':changes,'validation':validation}
    (ROOT/'experiments/trajectory-v2/book-installation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'Installed {len(changes)} validated book files; original PDF and companion archive preserved.')


if __name__=='__main__':main()
