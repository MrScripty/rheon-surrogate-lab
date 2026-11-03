"""Test the profiled velocity-advection bottleneck by inlining hot sampling calls."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

def prepare(output):
    if output.exists(): raise ValueError('Fresh optimized build required')
    source = (HERE/'prepare_profile.py').read_text()
    seam = "    (output/'target').symlink_to(prior/'target', target_is_directory=True)"
    addition = '''    sampling = output/'rheon/src/sampling.rs'
    text = sampling.read_text()
    for name in ['    pub fn sample(', '    pub fn backtrace(', '    fn departure(']:
        text = text.replace(name, '    #[inline]\\n'+name)
    sampling.write_text(text)
'''
    if source.count(seam) != 1: raise ValueError('Build seam changed')
    source = source.replace(seam, addition+seam)
    source = source.replace('[engine, scene, driver, Path(__file__)]', '[engine, scene, driver, sampling, Path(__file__)]')
    capture = output.with_name(output.name+'-builder.py')
    if capture.exists(): raise ValueError('Fresh builder capture required')
    capture.write_text(source)
    spec = importlib.util.spec_from_file_location('optimized_profile_builder', capture)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.build(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
