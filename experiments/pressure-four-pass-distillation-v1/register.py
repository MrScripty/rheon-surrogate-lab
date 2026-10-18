"""Open immutable prospective Projects in the existing visible Tuldok catalog."""
import argparse
from pathlib import Path
import sys

from common import ROOT, read, sha, write


def register(args):
    for name in ['config','projects','tuldok','session_file','output']:
        setattr(args,name,getattr(args,name).resolve())
    config=read(args.config)
    source=ROOT/config['source_lab']/'experiments/dependency-migration-v1'
    sys.path[:0]=[str(source),str(args.tuldok)]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP,Journal
    if args.output.exists():
        raise ValueError('Fresh registration journal required; inspect prior dispatch rather than repeat it')
    args.output.mkdir()
    (args.output/'mutations').mkdir()
    journal=Journal(args.output/'mutations')
    owner=OwnerHTTP(agent_api,args.endpoint,args.catalog_id,args.session_file)
    projects=[]
    for arm in config['arms']:
        directory=args.projects/(arm['id']+'-project-01')
        assembly=read(directory/'assembly.json')
        qualification=read(args.projects/(arm['id']+'-qualification.json'))
        if qualification['status']!='passed' or qualification['project_sha256']!=assembly['project']['sha256']:
            raise ValueError('Qualified project identity required')
        body=dict(root=args.project_alias,manifest_path=assembly['project']['path'])
        project=journal.mutation('open-'+arm['id'],'project_snapshots.open',body,
                                  lambda:owner.call('project_snapshots.open',body))
        if project['id']!=assembly['project']['id'] or project['manifest_sha256']!=assembly['project']['sha256']:
            raise ValueError('Visible Project identity differs')
        projects.append(dict(arm=arm['id'],project_id=project['id'],manifest_sha256=project['manifest_sha256']))
    write(args.output/'receipt.json',dict(status='passed',endpoint=args.endpoint,catalog_id=args.catalog_id,
                                         projects=projects,optimizer_updates=0,source_sha256=sha(__file__)))
    print('Registered',len(projects),'qualified prospective training projects',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','tuldok','session-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ['endpoint','catalog-id','project-alias']:
        parser.add_argument('--'+name,required=True)
    register(parser.parse_args())
