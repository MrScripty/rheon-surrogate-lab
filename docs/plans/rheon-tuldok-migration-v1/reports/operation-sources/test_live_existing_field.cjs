/* Selection policy fixture only: no browser, endpoint, authority or catalog. */
'use strict';
const assert=require('node:assert/strict');
const {openExistingField,allowImport}=require('./live_readback.cjs');
const digest='d'.repeat(64),id='7'.repeat(32),record={kind:'field_sequence',id,content_hash:digest};
async function fixture(change){
  const actions=[],value={total:1,items:[{id,content_hash:digest}]};if(change)change(value);
  const url='http://127.0.0.1:8091/api/workbench/records?'+new URLSearchParams({kind:'field_sequence',group:'field-sequence-source:'+digest,offset:'0',limit:'40'});
  const response={url:()=>url,request:()=>({method:()=> 'GET'}),ok:()=>true,body:async()=>Buffer.from(JSON.stringify(value))};
  let selected=false,submitted=false,predicate;
  const editor={locator:selector=>({selectOption:async value=>actions.push({selector,action:'select',value}),fill:async value=>actions.push({selector,action:'fill',value}),click:async()=>{actions.push({selector,action:'click'});if(selector==='#filters button')submitted=true;else{assert(submitted);assert.equal(selector,'[data-record-id="'+id+'"]');selected=true;}}})};
  const waitForResponse=check=>{predicate=check;assert(check(response));assert(!check({...response,request:()=>({method:()=> 'POST'})}));assert(!check({...response,url:()=>url.replace(digest,'e'.repeat(64))}));return Promise.resolve(response);};
  let result,error;try{result=await openExistingField(editor,record,digest,waitForResponse);}catch(caught){error=caught;}
  return {result,error,actions,selected,submitted,predicate};
}
(async()=>{
  const positive=await fixture();assert(!positive.error);assert(positive.selected);assert.equal(positive.result.collection_writes,0);assert.equal(positive.result.record_id,id);
  assert.deepEqual(positive.actions.slice(0,3),[{selector:'#kind',action:'select',value:'field_sequence'},{selector:'#query',action:'fill',value:''},{selector:'#group-filter',action:'fill',value:'field-sequence-source:'+digest}]);
  assert.deepEqual(positive.actions.filter(row=>row.action==='click').map(row=>row.selector),['#filters button','[data-record-id="'+id+'"]']);
  for(const change of [v=>v.total=2,v=>v.items=[],v=>v.items[0].id='8'.repeat(32),v=>v.items[0].content_hash='f'.repeat(64)]){const negative=await fixture(change);assert(negative.error);assert(!negative.selected);}
  assert(!allowImport({method:'POST',url:'http://127.0.0.1:8091/api/workspace/projects/'+id+'/import-data/cubic-fields',body:'{}'},{allow_project_import:false,provenance:{project_id:id},import_reference:'cubic-fields'},true));
  console.log('PASS existing-field exact GET search/selection; conflicting identities fail before record click; import disabled');
})().catch(error=>{console.error(error);process.exitCode=1;});
