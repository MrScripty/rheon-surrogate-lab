"""Build and qualify exact-sample interpolation on the profiled native path."""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent

def prepare(output):
    if output.exists():raise ValueError('Fresh aligned sampler build required')
    source=(HERE/'prepare_profile.py').read_text();seam="    (output/'target').symlink_to(prior/'target', target_is_directory=True)"
    addition='''    sys.path.insert(0,str(ROOT/'experiments/pressure-methods-v3'))
    import sampling_aligned
    sampling = output/'rheon/src/sampling.rs'
    sampling.write_text(sampling_aligned.patch(sampling.read_text()))
'''
    if source.count(seam)!=1:raise ValueError('Profile builder seam changed')
    source=source.replace(seam,addition+seam)
    source=source.replace('[engine, scene, driver, Path(__file__)]',"[engine, scene, driver, sampling, ROOT/'experiments/pressure-methods-v3/sampling_aligned.py', ROOT/'experiments/pressure-methods-v3/sampling_coordinates.py', Path(__file__)]")
    source=source.replace("'--lib', 'diagnostic::'","'--lib'")
    capture=output.with_name(output.name+'-builder.py')
    if capture.exists():raise ValueError('Fresh builder capture required')
    capture.write_text(source)
    spec=importlib.util.spec_from_file_location('aligned_sampler_builder',capture)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.build(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
