// Included inside diagnostic::engine in the isolated experimental Rheon copy.
// No reference track, same-request oracle solve, frame history or fallback.
pub struct SingleTrackSession {
    scene: Scene,
    grid: GridGeometry,
    track: Track,
    solver: PressureWorkspace,
    viscosity: ViscosityWorkspace,
    step: usize,
    halted: bool,
}
#[derive(Serialize)]
pub struct SingleStep {
    pub step: usize,
    pub preparation_seconds: f64,
    pub pressure_seconds: f64,
    pub projection_seconds: f64,
    pub frame_seconds: f64,
    pub quality: Quality,
    pub advanced: bool,
}
impl SingleTrackSession {
    pub fn new(scene: Scene) -> Result<Self, String> {
        // Reuse the qualified initializer, discarding its unused reference before
        // stepping. This temporary duplicate initialization is included in setup.
        let session = DiagnosticSession::new(scene)?;
        Ok(Self { scene: session.scene, grid: session.grid, track: session.ai,
            solver: session.solver, viscosity: session.viscosity, step: 0, halted: false })
    }
    pub fn advance(
        &mut self,
        native: bool,
        native_tolerance: f64,
        mut predict: impl FnMut(&[f64], &[f64]) -> Result<Vec<f64>, String>,
    ) -> Result<SingleStep, String> {
        if self.halted || self.step >= self.scene.steps {
            return Err("Single track already stopped".into());
        }
        let start = std::time::Instant::now();
        let cancel = AtomicBool::new(false);
        let prepared = prepare(&self.scene, &self.grid, &self.track,
            &mut self.viscosity, self.step as f64 * DT, &cancel)?;
        let preparation_seconds = start.elapsed().as_secs_f64();
        let pressure_start = std::time::Instant::now();
        let pressure = if native || self.step < self.scene.warmup_steps {
            let operator = PressureOperator::new(&self.grid, RHO).map_err(error)?;
            self.solver.solve_rhs(&operator, &prepared.rhs, DT,
                PressureSettings { relative_residual: native_tolerance,
                    absolute_residual: 1e-12, divergence_limit: 1e-7,
                    max_iterations: PRESSURE_ITERATION_LIMIT }, || false).map_err(error)?;
            self.solver.pressure().to_vec()
        } else {
            let operator = PressureOperator::new(&self.grid, RHO).map_err(error)?;
            let mut residual = vec![0.0; self.grid.cell_len()];
            operator.apply_full(&self.track.pressure, &mut residual).map_err(error)?;
            for (r, b) in residual.iter_mut().zip(&prepared.rhs) { *r = b - *r; }
            predict(&self.track.pressure, &residual)?
        };
        let pressure_seconds = pressure_start.elapsed().as_secs_f64();
        let projection_start = std::time::Instant::now();
        let (track, quality) = finish(&self.grid, &prepared, pressure, 1e-6)?;
        let projection_seconds = projection_start.elapsed().as_secs_f64();
        let advanced = quality.qualified;
        if advanced { self.track = track; self.step += 1; } else { self.halted = true; }
        Ok(SingleStep { step: self.step, preparation_seconds, pressure_seconds,
            projection_seconds, frame_seconds: start.elapsed().as_secs_f64(), quality, advanced })
    }
}
#[cfg(test)]
mod single_track_tests {
    use super::*;
    #[test]
    fn native_single_track_matches_existing_reference_fields_exactly() {
        let scene = Scene { counts: [8, 7, 5], steps: 8, warmup_steps: 2,
            acceleration: 0.125, ..Default::default() };
        let mut paired = DiagnosticSession::new(scene.clone()).unwrap();
        let mut single = SingleTrackSession::new(scene).unwrap();
        for _ in 0..8 {
            paired.advance(&AtomicBool::new(false), |r| Ok(r.initial_guess.to_vec())).unwrap();
            let step = single.advance(true, 1e-8, |_, _| panic!("native callback")).unwrap();
            assert!(step.advanced);
            assert_eq!(single.track.pressure, paired.reference.pressure);
            assert_eq!(single.track.velocity, paired.reference.velocity);
            assert_eq!(single.track.tracer, paired.reference.tracer);
        }
    }
    #[test]
    fn rejected_candidate_never_advances_and_no_oracle_or_fallback_is_called() {
        let mut single = SingleTrackSession::new(Scene { counts: [8, 7, 5],
            steps: 2, warmup_steps: 0, ..Default::default() }).unwrap();
        let prior = single.track.clone();
        let mut calls = 0;
        let step = single.advance(false, 1e-8, |base, _| { calls += 1; Ok(vec![0.0; base.len()]) }).unwrap();
        assert!(!step.advanced);
        assert_eq!(calls, 1);
        assert_eq!(single.step, 0);
        assert_eq!(single.track.pressure, prior.pressure);
        assert_eq!(single.track.velocity, prior.velocity);
        assert_eq!(single.track.tracer, prior.tracer);
        assert!(single.advance(true, 1e-8, |_, _| unreachable!()).is_err());
    }
}
