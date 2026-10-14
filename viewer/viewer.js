/* Dependency-free orthographic 3D voxel renderer over actual accepted fields. */
"use strict";
const query = new URLSearchParams(location.search);
const run = query.get("run") || "trajectory-v2";
if (!/^[A-Za-z0-9_-]+$/.test(run)) throw new Error("Invalid experiment name");
const mode = query.get("compute") || "cpu";
if (!["cpu", "gpu"].includes(mode)) throw new Error("Invalid compute mode");
const base = `../runs/${run}/${mode === "cpu" ? "cpu-only-8threads" : "evaluation"}/`;
const views = ["reference", "learned"];
const canvases = views.map(v => document.getElementById(v));
const slider = document.getElementById("timeline");
const camera = {yaw: -.65, pitch: -.22, zoom: 1, panX: 0, panY: 0};
let data, frame = 0, playing = false, last = 0;
function project(x,y,z,w,h) {
  x -= .5; y -= .5; z -= .5;
  const u = x*Math.cos(camera.yaw) + z*Math.sin(camera.yaw);
  const depth = -x*Math.sin(camera.yaw) + z*Math.cos(camera.yaw);
  const v = y*Math.cos(camera.pitch) - depth*Math.sin(camera.pitch);
  const d = y*Math.sin(camera.pitch) + depth*Math.cos(camera.pitch);
  const scale = Math.min(w,h)*.68*camera.zoom;
  return [w*.5 + camera.panX + u*scale, h*.52 + camera.panY - v*scale, d];
}
function draw(index) {
  const canvas=canvases[index], rect=canvas.getBoundingClientRect(), ratio=devicePixelRatio||1;
  canvas.width=Math.round(rect.width*ratio); canvas.height=Math.round(rect.height*ratio);
  const ctx=canvas.getContext("2d"); ctx.scale(ratio,ratio);
  const w=rect.width,h=rect.height;
  const edge=(a,b,color)=>{const p=project(...a,w,h),q=project(...b,w,h);ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();};
  for(let a=0;a<3;a++) for(let b=0;b<2;b++) for(let c=0;c<2;c++) {
    const lo=[0,0,0],hi=[0,0,0];lo[(a+1)%3]=hi[(a+1)%3]=b;lo[(a+2)%3]=hi[(a+2)%3]=c;hi[a]=1;edge(lo,hi,"#455670");
  }
  if(!data)return;
  const n=data.n, current=data[views[index]][frame], points=[];
  const voxels=current.voxels || current.tracer.map((v,i)=>[i,v]);
  for(const [cell,value] of voxels){
    if(value<.025)continue;
    const x=cell%n,y=Math.floor(cell/n)%n,z=Math.floor(cell/(n*n));
    const p=project((x+.5)/n,(y+.5)/n,(z+.5)/n,w,h);
    points.push({p,value});
  }
  points.sort((a,b)=>a.p[2]-b.p[2]);
  const size=Math.min(w,h)*.68*camera.zoom/n*.85;
  for(const {p,value} of points){
    const alpha=Math.min(.85,.12+value*.68), light=35+value*35;
    ctx.fillStyle=index===0?`hsla(32,85%,${light}%,${alpha})`:`hsla(174,70%,${light}%,${alpha})`;
    ctx.beginPath();ctx.moveTo(p[0],p[1]-size*.7);ctx.lineTo(p[0]+size*.7,p[1]-size*.3);ctx.lineTo(p[0]+size*.7,p[1]+size*.4);ctx.lineTo(p[0],p[1]+size*.8);ctx.lineTo(p[0]-size*.7,p[1]+size*.4);ctx.lineTo(p[0]-size*.7,p[1]-size*.3);ctx.closePath();ctx.fill();
  }
  ctx.fillStyle="#94a4be";ctx.font="11px system-ui";ctx.fillText("X / Y / Z · 1 m³",16,25);
  const r=data[views[index]][frame].report;
  document.getElementById(`${views[index]}Metrics`).innerHTML=`<span>Calculation<b>${r.total_ms.toFixed(2)} ms</b></span><span>Pressure stage<b>${r.pressure_ms.toFixed(2)} ms</b></span><span>Max divergence<b>${r.actual_divergence_max.toExponential(2)} s⁻¹</b></span>`;
}
function render(){canvases.forEach((_,i)=>draw(i));if(data){slider.value=frame;document.getElementById("frame").textContent=`Frame ${frame+1} / ${data.reference.length} · ${data.reference[frame].report.time.toFixed(3)} s`;}}
function playToggle(){playing=!playing;document.getElementById("play").textContent=playing?"❚❚ Pause":"▶ Play";document.getElementById("play").setAttribute("aria-label",playing?"Pause simulation":"Play simulation");}
document.getElementById("play").onclick=playToggle;
document.getElementById("backend").value=mode;
document.getElementById("backend").onchange=e=>{const url=new URL(location.href);url.searchParams.set("compute",e.target.value);location.href=url.href;};
document.getElementById("reset").onclick=()=>{frame=0;render();};
slider.oninput=()=>{frame=Number(slider.value);render();};
for(const canvas of canvases){
  let drag;
  canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener("pointermove",e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};if(e.shiftKey){camera.panX+=dx;camera.panY+=dy;}else{camera.yaw+=dx*.008;camera.pitch=Math.max(-1.3,Math.min(1.3,camera.pitch+dy*.008));}render();});
  canvas.addEventListener("pointerup",()=>drag=null);
  canvas.addEventListener("pointercancel",()=>drag=null);
  canvas.addEventListener("wheel",e=>{e.preventDefault();camera.zoom=Math.max(.3,Math.min(4,camera.zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
  canvas.addEventListener("keydown",e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){camera.yaw+=e.key==="ArrowLeft"?-.1:.1;render();}if(e.key===" "){e.preventDefault();playToggle();}});
}
window.addEventListener("resize",render);
function animate(now){if(playing&&data&&now-last>1000/Number(document.getElementById("rate").value)){frame=(frame+1)%data.reference.length;last=now;render();}requestAnimationFrame(animate);}
requestAnimationFrame(animate);
async function load(){
  try{
    let response=await fetch(base+"viewport-display.json");
    if(!response.ok)response=await fetch(base+"viewport.json");
    if(!response.ok)throw new Error(`Fields unavailable (${response.status})`);data=await response.json();
    slider.max=data.reference.length-1;frame=data.reference.length-1;
    document.getElementById("domain").textContent=`${data.n} × ${data.n} × ${data.n} cells · seed ${data.seed}`;
    document.getElementById("comparisonLabel").textContent=mode==="cpu"?"CPU vs CPU · 8 threads":"CPU vs GPU pressure";
    document.getElementById("referenceLabel").textContent=`Jacobi PCG · ${mode==="cpu"?8:1} native threads`;
    document.getElementById("modelLabel").textContent=`PyTorch ${data.torch || "2.14"} · ${mode==="cpu"?"CPU":"GPU"} inference · zero PCG`;
    document.getElementById("accuracy").textContent=data.accuracy.max_relative_velocity_l2.toExponential(2);
    document.getElementById("duration").textContent=`${(data.reference.length*data.dt).toFixed(2)} seconds`;
    document.getElementById("status").textContent="Full 3D fields loaded. Timeline replays accepted simulation frames.";
    render();
    const receipt=await fetch(base+"result.json");
    if(receipt.ok){const result=await receipt.json();document.getElementById("speedup").textContent=`${result.speedup_vs_jacobi.toFixed(2)}× vs Jacobi · ${result.speedup_vs_sgs.toFixed(2)}× vs SGS`;document.getElementById("receipt").href=base+"result.json";}
    if(mode==="cpu"){document.getElementById("timing").textContent="All methods: 8 CPU threads · model inference on CPU";}
    else{const parallel=await fetch(`../runs/${run}/parallel/result.json`);if(parallel.ok){const result=await parallel.json();document.getElementById("timing").textContent=`Parallel GPU-model comparison: ${result.speedup_vs_jacobi.toFixed(2)}× / ${result.speedup_vs_sgs.toFixed(2)}× speedup`;}}
  }catch(error){document.getElementById("status").textContent=`${error.message}. Run the trajectory evaluation and serve the lab repository root over HTTP.`;}
}
render();load();
