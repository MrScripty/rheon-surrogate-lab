"""Run selected Tuldok tests only against the private performance test database."""
import importlib.util,os,subprocess,sys
from pathlib import Path
root=Path.cwd();source=root/'runs/tuldok-performance-v1/Tuldok'
spec=importlib.util.spec_from_file_location('private_postgres',source/'scripts/local_postgres.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
env=dict(os.environ);env['TULDOK_DATABASE_URL']=module.configuration(Path('/tmp/tuldok-perf-pg-20261018'));env['PYTHONPATH']=str(source)+os.pathsep+str(source/'tests');env['PYTHONDONTWRITEBYTECODE']='1'
command=[str(root/'runs/dependency-migration-v1/app-env/bin/python3.12'),'-B','-m','unittest','-v',*sys.argv[1:]]
raise SystemExit(subprocess.call(command,cwd=source,env=env))
