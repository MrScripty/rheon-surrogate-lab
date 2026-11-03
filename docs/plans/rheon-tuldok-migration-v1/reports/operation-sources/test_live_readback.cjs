/* Pure request-policy checks. Requiring the helper never loads Playwright. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const {validateEnvelope,allowImport,allowEvidenceRead,validateFiles}=require('./live_readback.cjs');
const ENDPOINT='http://127.0.0.1:8091',CATALOG='a7da4a8de53541a2a0e2aad08005bb71';
const config={format:'rheon_live_browser_readback_config_v1',root_trigger:true,endpoint:ENDPOINT,catalog_id:CATALOG,
 machine:{status:'machine_qualified_browser_pending',catalog_id:CATALOG,endpoint:ENDPOINT,optimizer_updates_executed:1,restore_optimizer_updates:0,probe:{status:'exact',field_api_cli_exact:true,cuda_initialized:false,fresh_process:true,optimizer_updates:0},ordinary_restore_bound_profile:false,scientific_acceptance:'not_asserted',model_promotion:false},
 launch:{endpoint_owned_by_selected_pid:true,deployment_manifest_sha256:'a'.repeat(64)},deployment_manifest_sha256:'a'.repeat(64),
 gates:[{},{},{},{}],provenance:{project_id:'f'.repeat(32)},import_reference:'cubic-fields',import_sha256:'b'.repeat(64),allow_project_import:true,existing_record:null,
 required_evidence:['dependency-migration-plan','dependency-migration-results','dependency-migration-scientific-parity']};
validateEnvelope(config);
for(const change of [c=>c.root_trigger=false,c=>c.machine.status='prepared',c=>c.machine.optimizer_updates_executed=2,c=>c.machine.probe.cuda_initialized=true,c=>c.machine.model_promotion=true,c=>c.existing_record='also',c=>c.endpoint='http://127.0.0.1:8100',c=>c.required_evidence.pop()]){const c=structuredClone(config);change(c);assert.throws(()=>validateEnvelope(c));}
const request={url:ENDPOINT+'/api/workspace/projects/'+config.provenance.project_id+'/import-data/cubic-fields',method:'POST',body:JSON.stringify({expected_sha256:config.import_sha256,parents:[],name:'Current migration readback · Actual source',rights:'unknown',groups:['dependency-migration-v1 readback']})};
assert(allowImport(request,config,true));assert(!allowImport(request,config,false));assert(!allowImport(request,{...config,allow_project_import:false},true));
for(const change of [r=>r.method='GET',r=>r.url+='/approve',r=>r.url+='?dispatch=true',r=>r.url=r.url.replace(':8091',':8092'),r=>r.body='{}',r=>{const b=JSON.parse(r.body);b.expected_sha256='c'.repeat(64);r.body=JSON.stringify(b);},r=>{const b=JSON.parse(r.body);b.approve=true;r.body=JSON.stringify(b);},r=>{const b=JSON.parse(r.body);b.parents=['invented'];r.body=JSON.stringify(b);},r=>{const b=JSON.parse(r.body);b.rights='accepted';r.body=JSON.stringify(b);}]){const r={...request};change(r);assert(!allowImport(r,config,true));}
config.requests={restore:{path:'/api/workspace/runs/'+('5'.repeat(32))},field:{path:'/api/workspace/runs/'+('4'.repeat(32))+'/model-evidence',sha256:'d'.repeat(64),body:{checkpoint_sha256:'e'.repeat(64),member_ordinal:0,field_id:'full_pressure',axis:'z',index:0,split:'validation'}}};
const fieldRequest={url:ENDPOINT+config.requests.field.path,method:'POST',pin:'runs.model_evidence/2/'+('d'.repeat(64)),body:JSON.stringify(config.requests.field.body)};
assert(allowEvidenceRead(fieldRequest,config));assert(allowEvidenceRead({...fieldRequest,body:'{}'},config));assert(allowEvidenceRead({...fieldRequest,body:'{"split":"validation"}'},config));assert(allowEvidenceRead({...fieldRequest,url:ENDPOINT+config.requests.restore.path+'/model-evidence',body:'{}'},config));
for(const change of [r=>r.url=r.url.replace('4'.repeat(32),'5'.repeat(32)),r=>r.method='GET',r=>r.pin='runs.model_evidence/1/legacy',r=>{const b=JSON.parse(r.body);b.split='test';r.body=JSON.stringify(b);},r=>{const b=JSON.parse(r.body);b.checkpoint_sha256='f'.repeat(64);r.body=JSON.stringify(b);},r=>{const b=JSON.parse(r.body);b.index=1;r.body=JSON.stringify(b);},r=>r.body='{"approve":true}']){const r={...fieldRequest};change(r);assert(!allowEvidenceRead(r,config));}
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rheon-live-gates-'));
try{const digest=raw=>crypto.createHash('sha256').update(raw).digest('hex'),c=structuredClone(config);c.application=dir;
 const raw=Buffer.from('{"fixture":true}\n');fs.writeFileSync(path.join(dir,'deployment-manifest.json'),raw);c.deployment_manifest_sha256=digest(raw);c.launch.deployment_manifest_sha256=c.deployment_manifest_sha256;
 const app=Buffer.from('# inert fixture\n');fs.writeFileSync(path.join(dir,'app.py'),app);c.app_sha256=digest(app);
 const docs=[{schema:'rheon-tuldok-migration-cutover-v1',m3_service_switched:true,m3_complete:false,smoke_authorized:{kind:'typed_pressure_one_update_cpu_restore',max_optimizer_updates:1},ordinary_admissions_open:false,endpoint:ENDPOINT,catalog_id:CATALOG},c.launch,c.machine,c.provenance];
 c.gates=docs.map((value,i)=>{const raw=Buffer.from(JSON.stringify(value)),file=path.join(dir,i+'.json');fs.writeFileSync(file,raw);return {path:file,sha256:digest(raw)};});validateFiles(c);
 fs.appendFileSync(c.gates[2].path,' ');assert.throws(()=>validateFiles(c));
}finally{fs.rmSync(dir,{recursive:true,force:true});}
console.log('PASS local live readback gates, exact one-import policy, disk-witness drift; no browser/network/authority access');
