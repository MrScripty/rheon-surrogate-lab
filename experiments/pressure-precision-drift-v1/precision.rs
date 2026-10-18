//! Research-only storage-precision instrumentation, never a deployed solver.
#![allow(clippy::useless_conversion)] // Mechanical f32 -> f64 storage specialization.
use crate::{Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings, PressureWorkspace};
use crate::diagnostic::{Scene, DT, MU, RHO, PRESSURE_ITERATION_LIMIT};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
mod sampling64;
use sampling64::{SamplingError, ScalarSampler, VelocitySampler};
mod advection64;
use advection64::{advect_tracer, advect_velocity};
#[allow(dead_code)] // Retain the measured kernel's unused API/error variants unchanged.
mod viscosity64;
use viscosity64::ViscosityWorkspace;
const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
type Velocity = [Vec<f64>; 3];
fn refs(v: &Velocity) -> [&[f64]; 3] { [&v[0], &v[1], &v[2]] }
fn muts(v: &mut Velocity) -> [&mut [f64]; 3] { let [a,b,c]=v; [a,b,c] }
fn refs32(v: &[Vec<f32>;3]) -> [&[f32];3] { [&v[0],&v[1],&v[2]] }
fn muts32(v: &mut [Vec<f32>;3]) -> [&mut [f32];3] { let [a,b,c]=v; [a,b,c] }
fn error(e: impl std::fmt::Display) -> String { e.to_string() }
fn promote(v: &[Vec<f32>;3]) -> Velocity { std::array::from_fn(|d|v[d].iter().copied().map(f64::from).collect()) }
fn zeros(grid: &GridGeometry) -> Velocity { std::array::from_fn(|d|vec![0.0;grid.face_len(AXES[d])]) }

#[derive(Clone)]
pub struct Track { pub velocity: Velocity, pub tracer: Vec<f64>, pub pressure: Vec<f64> }
struct Prepared { velocity: Velocity, tracer: Vec<f64>, rhs: Vec<f64> }
#[derive(Clone,Debug,Serialize)]
pub struct Quality {
    pub relative_residual: Option<f64>, pub residual_l2:f64,
    pub predicted_divergence:f64, pub actual_divergence:f64,
    pub courant:f64, pub net_flux:f64, pub qualified:bool, pub hard_stop:bool,
    pub flux_limit:f64, pub walls_closed:bool,
}

pub struct Track64 {
    scene: Scene, grid: GridGeometry, pub state: Track,
    solver: PressureWorkspace, viscosity: ViscosityWorkspace, step:usize,
}
impl Track64 {
    pub fn new(scene:Scene,initial_tracer:&[f32])->Result<Self,String>{
        scene.validate()?;
        let grid=GridGeometry::new(scene.counts.map(|n|n as u64),[scene.h();3],[0.0;3]).map_err(error)?;
        if initial_tracer.len()!=grid.cell_len(){return Err("Initial tracer shape".into())}
        let method=if scene.sgs {PressureImplementation::SymmetricGaussSeidelPcgV1}else{PressureImplementation::JacobiPcgV1};
        let solver=PressureWorkspace::with_implementation(&grid,128*1024*1024,method).map_err(error)?;
        let viscosity=ViscosityWorkspace::new(grid.clone(),128*1024*1024).map_err(error)?;
        let state=Track{velocity:zeros(&grid),tracer:initial_tracer.iter().copied().map(f64::from).collect(),pressure:vec![0.0;grid.cell_len()]};
        Ok(Self{scene,grid,state,solver,viscosity,step:0})
    }
    pub fn advance(&mut self,cancel:&AtomicBool)->Result<Quality,String>{
        if self.step>=self.scene.steps{return Err("64-bit run already complete".into())}
        let prepared=prepare(&self.scene,&self.grid,&self.state,&mut self.viscosity,self.step as f64*DT,cancel)?;
        let (next,quality)=native_solve(&self.grid,&prepared,&mut self.solver,cancel)?;
        if !quality.qualified{return Err(format!("64-bit physical gate failed: {quality:?}"))}
        self.state=next; self.step+=1; Ok(quality)
    }
}

#[derive(Clone,Debug,Serialize)]
pub struct CastError {pub rms:f64,pub maximum:f64,pub maximum_bound:f64,pub values:usize,pub cast_mismatches:usize}
fn cast_error(rounded:&[f32],unrounded:&[f64])->Result<CastError,String>{
    if rounded.len()!=unrounded.len(){return Err("Cast observation shape".into())}
    let mut square=0.0;let mut maximum=0.0_f64;let mut bound=0.0_f64;let mut mismatch=0;
    for (&q,&v)in rounded.iter().zip(unrounded){
        if !v.is_finite(){return Err("Nonfinite unrounded cast observation".into())}
        mismatch+=usize::from((v as f32).to_bits()!=q.to_bits());
        let delta=f64::from(q)-v;square+=delta*delta;maximum=maximum.max(delta.abs());
        let u=2.0_f64.powi(-24);
        bound=bound.max((u*f64::from(q).abs()+2.0_f64.powi(-150))/(1.0-u));
    }
    if mismatch!=0{return Err(format!("Float64 stage did not reproduce {mismatch} original float32 casts"))}
    Ok(CastError{rms:(square/rounded.len()as f64).sqrt(),maximum,maximum_bound:bound,values:rounded.len(),cast_mismatches:mismatch})
}
fn vector_cast_error(rounded:&[Vec<f32>;3],unrounded:&Velocity)->Result<CastError,String>{
    let rows=(0..3).map(|d|cast_error(&rounded[d],&unrounded[d])).collect::<Result<Vec<_>,_>>()?;
    let values=rows.iter().map(|r|r.values).sum::<usize>();
    Ok(CastError{rms:(rows.iter().map(|r|r.rms*r.rms*r.values as f64).sum::<f64>()/values as f64).sqrt(),
        maximum:rows.iter().fold(0.0_f64,|a,r|a.max(r.maximum)),maximum_bound:rows.iter().fold(0.0_f64,|a,r|a.max(r.maximum_bound)),values,cast_mismatches:0})
}
#[derive(Clone,Debug,Serialize)]
pub struct CastObservations {
    pub advection:CastError,pub tracer_advection:CastError,pub force:CastError,pub viscosity:CastError,pub projection:CastError,
    pub projection_divergence_before_cast:f64,pub projection_divergence_after_cast:f64,
    pub projection_cast_divergence:f64,
}
pub struct Snapshot32 {
    pub velocity:[Vec<f32>;3],pub tracer:Vec<f32>,pub pressure:Vec<f64>,
    pub quality:crate::diagnostic::Quality,pub casts:CastObservations,
}
type Force32=fn(&Scene,&GridGeometry,&mut[Vec<f32>;3],f64,&AtomicBool)->Result<(),String>;

#[allow(clippy::too_many_arguments)] // Explicit immutable states prevent comparing different inputs.
pub(crate) fn observe_casts(scene:&Scene,grid:&GridGeometry,old:&[Vec<f32>;3],tracer:&[f32],
    prepared:&[Vec<f32>;3],prepared_tracer:&[f32],pressure:&[f64],corrected:&[Vec<f32>;3],
    time:f64,cancel:&AtomicBool,force32:Force32)->Result<CastObservations,String>{
    let mut advected32: [Vec<f32>;3]=std::array::from_fn(|d|vec![0.0;grid.face_len(AXES[d])]);
    let old64=promote(old);let mut advected64=zeros(grid);
    crate::advect_velocity(grid,refs32(old),DT,muts32(&mut advected32),||cancel.load(Ordering::Relaxed)).map_err(error)?;
    advect_velocity(grid,refs(&old64),DT,muts(&mut advected64),||cancel.load(Ordering::Relaxed)).map_err(error)?;
    let advection=vector_cast_error(&advected32,&advected64)?;
    let mut tracer32=vec![0.0;grid.cell_len()];let mut tracer64=vec![0.0;grid.cell_len()];
    crate::advect_tracer(grid,tracer,refs32(old),DT,&mut tracer32,||cancel.load(Ordering::Relaxed)).map_err(error)?;
    advect_tracer(grid,&tracer.iter().copied().map(f64::from).collect::<Vec<_>>(),refs(&old64),DT,&mut tracer64,||cancel.load(Ordering::Relaxed)).map_err(error)?;
    let tracer_advection=cast_error(&tracer32,&tracer64)?;
    if tracer32!=prepared_tracer{return Err("Replayed native tracer differs".into())}
    let mut forced64=promote(&advected32);
    force32(scene,grid,&mut advected32,time,cancel)?;force(scene,grid,&mut forced64,time,cancel)?;
    let force=vector_cast_error(&advected32,&forced64)?;
    let forced64=promote(&advected32);let mut viscous64=zeros(grid);
    let mut viscosity64=ViscosityWorkspace::new(grid.clone(),128*1024*1024).map_err(error)?;
    viscosity64.update(refs(&forced64),RHO,MU,DT,muts(&mut viscous64),|_|cancel.load(Ordering::Relaxed)).map_err(error)?;
    let viscosity=vector_cast_error(prepared,&viscous64)?;
    let operator=PressureOperator::new(grid,RHO).map_err(error)?;
    let pre64=promote(prepared);let mut corrected64=zeros(grid);
    operator.correct_velocity_f64(refs(&pre64),pressure,DT,muts(&mut corrected64)).map_err(error)?;
    let projection=vector_cast_error(corrected,&corrected64)?;
    let stored=promote(corrected);
    let delta:Velocity=std::array::from_fn(|d|stored[d].iter().zip(&corrected64[d]).map(|(a,b)|a-b).collect());
    let max_divergence=|v:&Velocity|->Result<f64,String>{
        let mut out=vec![0.0;grid.cell_len()];operator.divergence_f64(refs(v),&mut out).map_err(error)?;
        Ok(out.iter().fold(0.0_f64,|a,v|a.max(v.abs())))
    };
    Ok(CastObservations{advection,tracer_advection,force,viscosity,projection,
        projection_divergence_before_cast:max_divergence(&corrected64)?,projection_divergence_after_cast:max_divergence(&stored)?,
        projection_cast_divergence:max_divergence(&delta)?})
}
