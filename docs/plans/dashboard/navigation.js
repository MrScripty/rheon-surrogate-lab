function setupAnalysisTabs(){
 const tablist=$('analysis-tabs'),tabs=[...tablist.querySelectorAll('[role="tab"]')],panels=[...document.querySelectorAll('.analysis-panel')],scrollPositions=new Map(),jump=$('analysis-jump');let active='overview';
 function updateSummary(panel){
  const id=panel.id.slice(6),scope=panel.dataset.scope;
  if(id==='precision'){$('tab-state-precision').textContent=$('precision-scene').selectedOptions[0]?.textContent||'Precision controls';return}
  const state=$('tab-state-'+id);if(!state)return;
  state.textContent=(scope?$(scope)?.selectedOptions[0]?.textContent:null)||panel.dataset.question||'Independent controls';
  if(scope)state.textContent+=cohortFilterSummary(id);
 }
 function activate(id,{focus=false,scroll=true}={}){
  const tab=tabs.find(t=>t.dataset.analysis===id);if(!tab)return;
  if(active!==id){scrollPositions.set(active,window.scrollY);active=id}
  for(const t of tabs){const selected=t===tab;t.setAttribute('aria-selected',String(selected));t.tabIndex=selected?0:-1}
  for(const panel of panels)panel.hidden=panel.id!=='panel-'+id;
  for(let parent=tab.parentElement;parent&&parent!==tablist;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;
  jump.value=id;$('tooltip').hidden=true;
  try{history.replaceState(null,'','#'+id)}catch{}
  if(focus)tab.focus({preventScroll:true});
  if(scroll)window.scrollTo({top:scrollPositions.get(id)??0,behavior:'instant'});
 }
 for(const tab of tabs)tab.addEventListener('click',()=>activate(tab.dataset.analysis));
 jump.addEventListener('change',()=>activate(jump.value));
 tablist.addEventListener('keydown',event=>{const index=tabs.indexOf(document.activeElement);if(index<0)return;let next;if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else if(event.key==='ArrowDown')next=(index+1)%tabs.length;else if(event.key==='ArrowUp')next=(index+tabs.length-1)%tabs.length;else return;event.preventDefault();activate(tabs[next].dataset.analysis,{focus:true})});
 for(const panel of panels){updateSummary(panel);panel.addEventListener('change',()=>updateSummary(panel))}
 window.addEventListener('hashchange',()=>activate(location.hash.slice(1)));
 activate(tabs.some(t=>t.dataset.analysis===location.hash.slice(1))?location.hash.slice(1):'overview',{scroll:false});
}
function setupFilterMemory(){
 const key='rheon-evaluation-filters-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
 function controls(panel){return [...panel.querySelectorAll('select[id],input[id]')].filter(e=>e.type!=='file')}
 function values(panel){return Object.fromEntries(controls(panel).map(e=>[e.id,e.type==='checkbox'?e.checked:e.value]))}
 function save(panel){saved[panel.id]=values(panel);try{localStorage.setItem(key,JSON.stringify(saved))}catch{}}
 for(const panel of document.querySelectorAll('.analysis-panel')){
  const defaults=values(panel),stored=saved[panel.id]||{};
  // Cohort changes populate model selectors. Repeat once to restore dependent
  // selections after their parents, and ignore options absent in a new snapshot.
  for(let pass=0;pass<2;pass++)for(const e of controls(panel))if(Object.hasOwn(stored,e.id)){
   const value=stored[e.id];if(e.tagName==='SELECT'&&![...e.options].some(o=>o.value===value&&!o.disabled))continue;
   if(e.type==='checkbox'){if(e.checked===value)continue;e.checked=value}else{if(e.value===String(value))continue;e.value=value}
   e.dispatchEvent(new Event(e.type==='range'?'input':'change',{bubbles:true}));
  }
  const reset=document.createElement('button');reset.type='button';reset.className='reset-tab';reset.textContent='Reset this tab';reset.addEventListener('click',()=>{for(let pass=0;pass<2;pass++)for(const e of controls(panel))if(Object.hasOwn(defaults,e.id)){if(e.type==='checkbox')e.checked=defaults[e.id];else e.value=defaults[e.id];e.dispatchEvent(new Event(e.type==='range'?'input':'change',{bubbles:true}))}save(panel)});
  const note=panel.querySelector('.tab-filter-note');if(note)note.append(reset);else panel.prepend(reset);
  panel.addEventListener('change',()=>save(panel));panel.addEventListener('input',()=>save(panel));
 }
}
