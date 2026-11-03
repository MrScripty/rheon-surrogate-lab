'use strict';
// Reuse the exact qualified205card codec checks;change only revision-specific expectations.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const originalPath=path.join(__dirname,'qualify_pressure_outcome_notes.cjs');
const original=fs.readFileSync(originalPath,'utf8');
const hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
assert.equal(hash(original),'7e479d6c772b1bd65d57f62b800dc4aa7cdbe80fc43f725a88d6d2bcaf1f9be9');
let text=original;
const replacements=[
  ["'native-pressure-outcome-notes-01'","'native-pressure-reproduction-status-notes-01'"],
  ["reports/pressure-outcome-notes-qualification-01.json","reports/pressure-reproduction-status-notes-qualification-01.json"],
  ["['method','purpose','experiment']","['method','purpose','experiment','outcome','lesson']"],
  ["assert.ok(model.summary.reproduction.startsWith(before.summary.reproduction));",
   "assert.ok(model.summary.reproduction.includes('GPU evaluation and timing are now qualified')||model.summary.reproduction.includes('common GPU pressure accuracy and latency are qualified'));assert.ok(!model.summary.reproduction.includes('fresh qualified latency'));assert.ok(!model.summary.reproduction.includes('a current common benchmark for this unnamed fit remain unqualified'));"],
  ["for(const name of ['outcome','lesson','reproduction'])assert.ok(text.includes(JSON.stringify(before.summary[name])));",
   "assert.ok(text.includes(JSON.stringify(before.summary.reproduction)));"],
  ["assert.equal(model.aliases.length,before.aliases.length+1);","equal(model.aliases,before.aliases);"],
  ["assert.ok(overview.endsWith(oldOverview.slice('# Rheon model portfolio\\n\\n'.length)));",
   String.raw`assert.equal(overview.replace(/\s+/g,' ').trim(),oldOverview.replace(/\s+/g,' ').trim());`],
  ["assert.ok(overview.includes(text));",String.raw`assert.ok(overview.replace(/\s+/g,' ').includes(text));`]
];
for(const [before,after]of replacements){assert.equal(text.split(before).length,2,'Exact old qualifier expectation required: '+before);text=text.replace(before,after);}
new Function('require','__dirname','__filename',text)(require,__dirname,__filename);
const root=path.resolve(__dirname,'../..');
const output=path.join(root,'docs/plans/model-portfolio-migration-v1/reports/pressure-reproduction-status-notes-qualification-01.json');
const result=JSON.parse(fs.readFileSync(output,'utf8'));
result.five_scientific_summary_fields_unchanged=23;
result.aliases_groups_relationships_and_measurement_bindings_unchanged=true;
result.overview_words_and_metric_values_unchanged=true;
result.current_reproduction_status_coherent=true;
result.delegated_original_qualifier={path:path.relative(root,originalPath),sha256:hash(original)};
result.evidence.push(result.delegated_original_qualifier);
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
console.log('I57status correction qualified:all23five-field scientific summaries,aliases,measurement bindings andoverview words unchanged.');
