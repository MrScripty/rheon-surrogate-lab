"""Prepare a reviewable book update inside the lab before installing it.

The two original PDF/ZIP archives remain available and are labelled historical.
The current Markdown/web edition and separate physics companion cover chapter 41.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT.parent.parent / "BCAI/training-your-own-models"
STAGE = ROOT / "runs/book-staging"


def main():
    if STAGE.exists():
        raise FileExistsError("book staging already exists")
    shutil.copytree(BOOK, STAGE, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__'))
    (STAGE / 'qa').mkdir(exist_ok=True)
    chapter = (ROOT / 'docs/book-physics-chapter.md').read_text()
    original = (STAGE / 'source/manuscript.md').read_text()
    marker = '<a id="glossary"></a>'
    if marker not in original:
        raise ValueError('glossary insertion anchor changed')
    manuscript = original.replace(marker, chapter + '\n' + marker, 1)
    toc = '- [40  Companion guide and verification record](#companion-guide-and-verification-record)'
    if toc not in manuscript:
        raise ValueError('chapter 40 contents anchor changed')
    manuscript = manuscript.replace(toc, toc + '\n- [41  Project train a 3D fluid pressure solver](#project-train-a-3d-fluid-pressure-solver)', 1)
    manuscript = manuscript.replace('Edition dated 2 October 2026',
        'Original edition dated 2 October 2026; chapter 41 added 12 October 2026. The first 40 chapters retain their original verification scope. Chapter 41 records a separate measured CUDA fluid-model experiment.', 1)
    warning = 'GPU recipes were not trained on the reader\'s RTX card while preparing this book'
    manuscript = manuscript.replace(warning, 'The GPU recipes in chapters 1–40 were not trained on the reader\'s RTX card while preparing the original edition', 1)
    marker40 = '<a id="what-was-actually-executed"></a>\n## What was actually executed'
    if marker40 in manuscript:
        manuscript = manuscript.replace(marker40, marker40 + '\n\nThis section records the original 40-chapter edition. The measured CUDA pressure-model fit and complete 3D rollout experiment added later are documented in [chapter 41](#project-train-a-3d-fluid-pressure-solver).', 1)
    (STAGE / 'source/manuscript.md').write_text(manuscript)
    (STAGE / 'docs/downloads/training-your-own-models.md').write_text(manuscript)
    (STAGE / 'docs/downloads/ai-3d-physics-solver.md').write_text(chapter)
    build = (STAGE / 'build.py').read_text()
    build = build.replace('40 chapters', '41 chapters').replace("==40", "==41").replace("'chapters':40", "'chapters':41")
    build = build.replace("' / 40'", "' / 41'").replace('<strong>40</strong>', '<strong>41</strong>')
    build = build.replace('Across all 40 chapters', 'Across all 41 chapters')
    build = build.replace('No GPU training or 24 GB peak-memory measurements were performed for this edition.',
        'The original 40 chapters did not perform GPU training. Chapter 41 adds a measured CUDA fluid-model fit, full 3D rollout comparison and recorded memory use.')
    build = build.replace("'Complete PDF','247 pages, including the front and back covers.'",
        "'Original PDF archive','Original 40-chapter edition: 247 pages including covers. The physics chapter is a separate download below.'")
    build = build.replace("'Editable Markdown','The complete manuscript, unchanged. Its five figures are in the companion archive.'",
        "'Current editable Markdown','Complete current 41-chapter manuscript, including the 3D fluid solver project.'")
    build = build.replace("'Companion archive','57 scripts, source manuscript, figures, configurations, and the companion README.'",
        "'Original companion archive','Original 57 scripts and supporting resources for chapters 1–40. Chapter 41 has its own companion below.'")
    needle="checks={}"
    build = build.replace(needle, "files += [('ai-3d-physics-solver.md','3D physics chapter','Chapter 41 as standalone Markdown.'), ('rheon-physics-companion.zip','3D physics companion','Eight Python scripts, native probes, viewer, frozen model and reproduction guide.') ]\n"+needle,1)
    build = build.replace('The same book, in the format that suits your work.',
        'The current web and Markdown edition includes chapter 41. The original PDF and original companion remain historical archives; the physics chapter has separate current downloads.')
    build = build.replace('GPU recipes are proposed experiments, not measured results from your card.',
        'Chapters 1–40 retain their original execution labels. Chapter 41 records the measured fluid experiment and its supported scope.')
    (STAGE / 'build.py').write_text(build)
    readme = (STAGE / 'README.md').read_text().replace('complete 40-chapter manuscript','complete 41-chapter manuscript')
    readme = readme.replace('No GPU training or 24 GB peak-memory measurements were performed for this edition.',
        'The original 40 chapters performed no GPU training. Added chapter 41 covers a measured PyTorch 2.14 CUDA pressure-model fit, 3D rollouts, memory and complete simulation timings; its exact supported scope is stated there.')
    readme += '\n## 3D physics solver project\n\nChapter 41 teaches a reproducible learned pressure operator inside Rheon 3D flow. Read `source/manuscript.md`, or build `docs/chapters/41.html`. The current Markdown includes this chapter. The original PDF and original companion ZIP remain clearly labelled archives; `docs/downloads/rheon-physics-companion.zip` contains the new executable project. No website publication is performed by this update.\n'
    (STAGE / 'README.md').write_text(readme)
    archive = STAGE / 'docs/downloads/rheon-physics-companion.zip'
    selected = [ROOT/'rust-toolchain.toml', ROOT/'LICENSE', ROOT/'THIRD_PARTY.md',
        *sorted((ROOT/'probes/trajectory').rglob('*')), *sorted((ROOT/'probes/trajectory-parallel').rglob('*')),
        *sorted((ROOT/'viewer').glob('*')),
        *[ROOT/'scripts'/name for name in ('trajectory_model.py','trajectory_experiment.py',
          'pumas_trajectory.py','package_trajectory_viewer.py','trajectory_parallel.py','test_trajectory_model.py',
          'verify_trajectory_results.py','verify_parallel_results.py')]]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in selected:
            if p.is_file(): z.write(p,'rheon-surrogate-lab/'+p.relative_to(ROOT).as_posix())
        for name in ('model.pt','freeze.json'):
            z.write(ROOT/'runs/trajectory-v2'/name,'rheon-surrogate-lab/experiments/trajectory-v2/'+name)
        z.writestr('rheon-surrogate-lab/README.md',chapter)
    manifest = {}
    for p in (STAGE/'docs/downloads').iterdir():
        if p.is_file() and p.suffix in ('.md','.pdf','.zip'):
            manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (STAGE/'source/download-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['python3','build.py'],cwd=STAGE,check=True)
    subprocess.run(['python3','tests/validate.py'],cwd=STAGE,check=True)
    print(f'Prepared validated update: {STAGE}')


if __name__ == '__main__': main()
