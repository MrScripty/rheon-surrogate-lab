"""Record completed real chart actions separately from immutable import custody."""
from pathlib import Path
import time

from review_water_learning_sequence import ROOT, binding, canonical, read, require


def run():
    docs = ROOT/'docs/plans/model-portfolio-migration-v1'
    runs = ROOT/'runs/model-portfolio-migration-v1'
    browser = runs/'native-common-generator-GPU-charts-01'
    job = runs/'native-common-generator-GPU-import-01'
    destination = docs/'reports/native-common-generator-GPU-charts-UI-01.json'
    require(not destination.exists(), 'Preserve original UI acceptance')
    report = read(browser/'report.json')
    custody = read(docs/'reports/native-common-generator-GPU-import-custody-01.json')
    actions = read(job/'browser-actions.json')
    expected = read(job/'browser-expectations.json')
    requests = report['api_requests']
    require(report['status'] == 'passed_native_project_UI_actions' and report['browser_stopped']
            and not report['errors'] and report['page']['error'] is None
            and report['steps'] == actions and len(actions) == 516, 'Complete original browser required')
    require(len(requests) == 54 and all(r['status'] == 200 and r['complete'] for r in requests),
            'All actual HTTP reads must complete successfully')
    require(len(expected) == 40 and len(list(browser.glob('*.png'))) == 45,
            'All requested comparisons and captures required')
    require(custody['status'] == 'four_common_generator_GPU_tables_native_custody_exact'
            and custody['checkpoint'] == actions[0]['text'], 'Exact imported checkpoint required')
    viewed = ['K1-all-pressure_relative_l2.png','K1-all-per_output_median_ms.png',
        'K2-all-pressure_relative_l2.png','K2-all-per_output_median_ms.png',
        'K3-all-pressure_relative_l2.png','K4-all-per_output_median_ms.png',
        'K4-all-pressure_relative_l2.png','K3-cpm16-grid32-time.png',
        'K4-GPU-source-inspection.png','K1-cpm8-grid16-pressure.png']
    result = dict(status='four_native_generator_GPU_tables_real_chart_UI_accepted',created_unix=time.time(),
        checkpoint=custody['checkpoint'],models=10,measured_scalars=6400,native_sources=25,
        actual_UI_actions=516,requested_comparisons=40,HTTP200_complete_reads=54,
        comparison_HTTP_reads_including_four_initial_loads=44,captured_screenshots=45,
        visually_inspected=[binding(browser/name) for name in viewed],
        visual_result='Representative K1–4 pressure/time charts, two density/grid selections and exact source inspection render readable values and explicit GPU/full-K labels. Some table header words wrap; no new blocking defect observed.',
        statistic='Displayed maxima of per-case statistics over matching cases; not median across cases.',
        numerical_scope='Import custody verifies all6400scalars; original owner checks12unfiltered native maxima. Real40comparison actions assert counts,labels and filters. Browser does not retain all response bodies; no claim that all displayed values were independently recomputed from those responses.',
        timing_scope='Historical ten-model GPU timings include the CUDA forward-validation hook inside measured calls (I47). Production-call speed and hook overhead remain unqualified. Original packets preserved; native scope correction pending.',
        API_compare_seconds=dict(minimum=min(r['total_wall_ms']/1000 for r in requests if r['path'].endswith('/compare')),
            maximum=max(r['total_wall_ms']/1000 for r in requests if r['path'].endswith('/compare'))),
        evidence=[binding(browser/'report.json'),binding(job/'browser-actions.json'),
            binding(job/'browser-expectations.json'),binding(docs/'reports/native-common-generator-GPU-import-custody-01.json'),
            binding(docs/'reports/generator-timing-hook-scope-01.json')],
        producer=binding(Path(__file__)),project_left_open=True,full_portfolio_complete=False)
    destination.write_bytes(canonical(result)+b'\n')
    print(result['status'], '516actions/40comparisons/54complete HTTP200reads;10of45captures inspected.')


if __name__ == '__main__':
    run()
