// Included only in the isolated research engine for independent forcing tests.
impl SingleTrackSession {
    pub fn preview_prescribed_force(&self, step: usize, mut velocity: [Vec<f32>; 3])
        -> Result<[Vec<f32>; 3], String>
    {
        if step >= self.scene.steps || velocity.iter().enumerate().any(|(d, values)| {
            values.len() != self.grid.face_counts(AXES[d]).iter().product::<usize>()
                || values.iter().any(|v| !v.is_finite())
        }) {
            return Err("Bounded finite prescribed-force preview required".into());
        }
        force(&self.scene, &self.grid, &mut velocity, step as f64 * DT, &AtomicBool::new(false))?;
        Ok(velocity)
    }
}
