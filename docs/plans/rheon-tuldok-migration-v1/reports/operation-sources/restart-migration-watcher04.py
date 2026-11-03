from pathlib import Path
import os,sys,json,hashlib,subprocess,time,signal,resource,select,re,struct
root=Path.cwd();m=root/'runs/dependency-migration-v1';logs=m/'receipts/live-migration';source=m/'lab/experiments/dependency-migration-v1/live_migration.py';argv=[str(root/'.venv/bin/python'),'-B',str(source),'--root',str(root),'--interval','2'];oldpid=532121;oldticks=6056249;receipt=logs/'watcher-04.json';newlog=logs/'watcher-04.stderr.log';started=time.monotonic();sha=lambda b:hashlib.sha256(b).hexdigest();row={'status':'replacement_preflight','previous_pid':oldpid,'previous_linux_start_ticks':oldticks,'retry_policy':'one authorized replacement; no automatic retry'}
def write():receipt.write_text(json.dumps(row,indent=2)+'\n')
def stat(pid):return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
def children(pid):
 result=set()
 for p in Path('/proc',str(pid),'task').glob('*/children'):
  try:result.update(map(int,p.read_text().split()))
  except FileNotFoundError:pass
 return sorted(result)
def read_public(path):
 b=path.read_bytes();assert len(b)<=2097152;return json.loads(b),b
def fields(raw):
 text=raw.decode();result={}
 for key in ('pressure','source'):
  hits=list(re.finditer('"'+key+'"\\s*:\\s*',text));assert len(hits)==1
  start=hits[0].end();value,end=json.JSONDecoder().raw_decode(text[start:]);result[key]=text[start:start+end].encode()
 return result
assert not receipt.exists() and not newlog.exists()
try:
 assert sha(source.read_bytes())=='ab6d937d3c4723f4d48d738306d39dfff70f6a394e7d856c58b9d5702004481b'
 old=Path('/proc',str(oldpid));s=stat(oldpid);assert int(s[19])==oldticks and s[0]!='Z';assert old.stat().st_uid==os.getuid();assert old.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==argv;assert old.joinpath('cwd').resolve()==root.resolve();assert not children(oldpid)
 assert old.joinpath('fd/1').resolve()==logs/'watcher-03.stderr.log' and old.joinpath('fd/2').resolve()==logs/'watcher-03.stderr.log'
 old_log=(logs/'watcher-03.stderr.log').read_bytes();old_receipt=(logs/'watcher-03.json').read_bytes();assert len(old_log)==0
 before,raw=read_public(m/'live-migration/frame.json');before_fields=fields(raw);raw_machine,_=read_public(m/'typed-pressure-live-01/receipt.json');assert raw_machine['status']=='machine_qualified_browser_pending'
 row.update(previous_log_sha256=sha(old_log),previous_log_bytes=len(old_log),previous_receipt_sha256=sha(old_receipt),source_sha256=sha(source.read_bytes()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m/'lab',text=True).strip(),command=argv,cwd=str(root),raw_machine_status=raw_machine['status'],field_fragment_sha256_before={k:sha(v) for k,v in before_fields.items()});write()
 pidfd=os.pidfd_open(oldpid,0)
 try:
  assert int(stat(oldpid)[19])==oldticks
  signal.pidfd_send_signal(pidfd,signal.SIGTERM)
  assert select.select([pidfd],[],[],8)[0],'old publisher did not exit after SIGTERM'
 finally:os.close(pidfd)
 row['status']='previous_exact_owner_exited';write()
 def limits():
  os.nice(10);os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64));resource.setrlimit(resource.RLIMIT_FSIZE,(2097152,2097152))
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
 with newlog.open('xb') as log:child=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True,preexec_fn=limits)
 s=stat(child.pid);row.update(status='new_detached_owner_started',pid=child.pid,linux_start_ticks=int(s[19]),detached_session=int(s[3]),nice=10,cpu_affinity=[2],address_space_limit_bytes=536870912,open_file_limit=64,regular_file_limit_bytes=2097152,numerical_library_threads=1,gpu_visible=False,log=str(newlog));write()
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  assert child.poll() is None,('new publisher exit',child.returncode)
  a,_=read_public(m/'live-migration/activity.json')
  if a.get('state')=='running' and a.get('progress')=={'completed':74,'total':74} and 'machine_qualified_browser_pending' in a.get('message',''):break
  time.sleep(.1)
 else:raise AssertionError('new publisher did not publish actual healthy74/current machine status')
 first=a;time.sleep(3.2);assert child.poll() is None;second,_=read_public(m/'live-migration/activity.json');assert second['sequence']>first['sequence'] and second['state']=='running' and second['progress']=={'completed':74,'total':74};assert '74/74 current regression case receipts passed; 0 failed, 0 unreadable.' in second['message'] and 'machine_qualified_browser_pending' in second['message']
 after,raw=read_public(m/'live-migration/frame.json');after_fields=fields(raw);assert before_fields==after_fields;assert not children(child.pid);assert newlog.stat().st_size<8192
 assert (logs/'watcher-03.stderr.log').read_bytes()==old_log and (logs/'watcher-03.json').read_bytes()==old_receipt
 consumer=m/'Tuldok-deployed/current-03'
 code='import sys,json;sys.path.insert(0,sys.argv[1]);import project_activity;v=project_activity.activity(json.load(open(sys.argv[2])));print(json.dumps({k:v[k] for k in ("available","authority","progress")}))'
 decoded=subprocess.run([str(root/'.venv/bin/python'),'-B','-c',code,str(consumer),str(m/'live-migration/activity.json')],capture_output=True,text=True,timeout=5)
 assert decoded.returncode==0,decoded.stderr
 projection=json.loads(decoded.stdout);assert projection['available'] is True and projection['progress']=={'completed':74,'total':74}
 assert len(second['message'])<=1000 and 'Cutover: m3_complete=false' in second['message'] and 'Final M3 completion: absent' in second['message']
 row.update(message_characters=len(second['message']),actual_consumer={'path':str(consumer/'project_activity.py'),'sha256':sha((consumer/'project_activity.py').read_bytes()),'exit_code':decoded.returncode,'projection':projection})
 row.update(status='healthy_current_observation',elapsed_s=time.monotonic()-started,heartbeat_sequence=[first['sequence'],second['sequence']],progress=second['progress'],failed_scene_receipts=0,unreadable_scene_receipts=0,current_machine_marker_visible=True,pressure_and_source_fragments_byteexact=True,field_fragment_sha256_after={k:sha(v) for k,v in after_fields.items()},active_child_pids=[],log_bytes=newlog.stat().st_size,old_log_and_receipt_preserved=True,activity_updated_at=second['updated_at'],activity_message=second['message'],observer_pid493890_untouched=True);write();print(json.dumps({k:row[k] for k in ['status','pid','linux_start_ticks','source_commit','source_sha256','heartbeat_sequence','progress','log_bytes','pressure_and_source_fragments_byteexact','elapsed_s']},indent=2))
except BaseException as error:
 row.update(status='replacement_failed_no_retry',error_type=type(error).__name__,error_detail=str(error)[:400],elapsed_s=time.monotonic()-started);write();raise
