'use strict';
const assert=require('node:assert/strict');
function modelQuery(guide,model){
  const benchmark=model.aliases.find(alias=>alias.startsWith('GPU benchmark: '));
  const candidates=[benchmark,model.label,...model.aliases].filter(value=>typeof value==='string'&&value.length&&value.length<=500);
  for(const query of candidates){
    const matching=guide.models.filter(item=>[item.label,...item.aliases].some(value=>value.toLocaleLowerCase().includes(query.toLocaleLowerCase())));
    if(matching.length===1&&matching[0].id===model.id)return query;
  }
  assert.fail('No unique native substring query for '+model.id);
}
module.exports={modelQuery};
