'use strict';
const assert=require('node:assert/strict'),test=require('node:test');
const {modelQuery}=require('./matched_model_query.cjs');
test('prefix benchmark alias selects unique full model label',()=>{
  const model={id:'base',label:'Full AI b2 train',aliases:['GPU benchmark: b2']};
  const guide={models:[model,{id:'endpoint',label:'Full AI b2-endpoint train',aliases:['GPU benchmark: b2-endpoint']}]};
  assert.equal(modelQuery(guide,model),model.label);
});
test('unique benchmark alias remains preferred',()=>{
  const model={id:'base',label:'Full AI b2 train',aliases:['GPU benchmark: b2-v1']};
  assert.equal(modelQuery({models:[model]},model),'GPU benchmark: b2-v1');
});
test('an ambiguous guide cannot silently select the first fit',()=>{
  const model={id:'base',label:'Duplicate',aliases:['GPU benchmark: same']};
  assert.throws(()=>modelQuery({models:[model,{...model,id:'other'}]},model));
});
