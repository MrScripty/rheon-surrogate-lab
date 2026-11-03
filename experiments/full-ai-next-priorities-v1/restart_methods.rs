// Isolated research restart API; the existing advance/finish pressure gates are unchanged.
impl SingleTrackSession {
    pub fn state_step(&self) -> usize { self.step }
    pub fn restore_state_fields(&mut self, step: usize, pressure: Vec<f64>,
        velocity: [Vec<f32>;3], tracer: Vec<f32>) -> Result<(),String> {
        if step>self.scene.steps || pressure.len()!=self.grid.cell_len() || tracer.len()!=self.grid.cell_len()
            || pressure.first()!=Some(&0.) || pressure.iter().any(|v|!v.is_finite())
            || tracer.iter().any(|v|!v.is_finite())
            || (0..3).any(|d|velocity[d].len()!=self.grid.face_len(AXES[d]) || velocity[d].iter().any(|v|!v.is_finite())) {
            return Err("Invalid complete research restart state".into());
        }
        let solver=PressureWorkspace::new(&self.grid,2*1024*1024*1024).map_err(error)?;
        let viscosity=ViscosityWorkspace::new(self.grid.clone(),2*1024*1024*1024).map_err(error)?;
        self.track=Track {pressure,velocity,tracer};self.step=step;self.halted=false;
        self.solver=solver;self.viscosity=viscosity;
        Ok(())
    }
}
#[cfg(test)]
mod temporal_restart_tests {
    use super::*;
    #[test]
    fn eight_future_native_steps_are_bitwise_equal_after_complete_restart() {
        for counts in [[8,7,5],[16,16,16]] {
            let scene=Scene {counts,steps:16,warmup_steps:0,..Default::default()};
            let mut original=SingleTrackSession::new(scene.clone()).unwrap();
            for _ in 0..4 { assert!(original.advance(true,1e-8,|_,_|unreachable!()).unwrap().advanced); }
            let mut restarted=SingleTrackSession::new(scene).unwrap();
            restarted.restore_state_fields(original.step,original.track.pressure.clone(),
                original.track.velocity.clone(),original.track.tracer.clone()).unwrap();
            for _ in 0..8 {
                assert!(original.advance(true,1e-8,|_,_|unreachable!()).unwrap().advanced);
                assert!(restarted.advance(true,1e-8,|_,_|unreachable!()).unwrap().advanced);
                assert_eq!(original.step,restarted.step);
                assert_eq!(original.track.pressure,restarted.track.pressure);
                assert_eq!(original.track.velocity,restarted.track.velocity);
                assert_eq!(original.track.tracer,restarted.track.tracer);
            }
        }
    }
    #[test]
    fn invalid_restart_preserves_the_prior_state() {
        let mut session=SingleTrackSession::new(Scene {steps:16,warmup_steps:0,..Default::default()}).unwrap();
        let before=session.track.clone();
        let mut bad=before.pressure.clone();bad[0]=f64::NAN;
        assert!(session.restore_state_fields(0,bad,before.velocity.clone(),before.tracer.clone()).is_err());
        assert!(session.restore_state_fields(17,before.pressure.clone(),before.velocity.clone(),before.tracer.clone()).is_err());
        assert_eq!(session.track.pressure,before.pressure);assert_eq!(session.step,0);
    }
}
