#[cfg(test)]
mod finite_f32_viscosity_tests {
    use super::*;

    fn fixture(value: f32) -> (ViscosityWorkspace, [Vec<f32>; 3]) {
        let grid = GridGeometry::new([5; 3], [0.04; 3], [0.0; 3]).unwrap();
        let mut velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(Axis::ALL[d])]);
        let index = grid.face_index(Axis::Z, [2, 2, 2]).unwrap();
        velocity[2][index] = value;
        (ViscosityWorkspace::new(grid, 1024 * 1024).unwrap(), velocity)
    }

    #[test]
    fn finite_subnormal_input_preserves_energy_gates_without_flushing() {
        for tiny in [1.0e-40_f32, f32::from_bits(1)] {
            let (mut workspace, velocity) = fixture(tiny);
            let mut output = velocity.clone();
            let [x, y, z] = &mut output;
            let report = workspace.update(
                [&velocity[0], &velocity[1], &velocity[2]], 1000.0, 0.001,
                1.0 / 240.0, [x, y, z], |_| false,
            ).unwrap();
            assert!(output.iter().flatten().all(|v| v.is_finite()));
            assert!(output.iter().flatten().any(|v| v.is_subnormal()));
            assert!(report.kinetic_after - report.kinetic_before <= report.energy_rounding_budget);
            assert!(report.stability_number <= 0.25);
        }
    }

    #[test]
    fn nonfinite_input_remains_transactionally_rejected() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (mut workspace, velocity) = fixture(invalid);
            let mut output = std::array::from_fn::<_, 3, _>(|d| vec![0.125; velocity[d].len()]);
            let saved = output.clone();
            let [x, y, z] = &mut output;
            let result = workspace.update(
                [&velocity[0], &velocity[1], &velocity[2]], 1000.0, 0.001,
                1.0 / 240.0, [x, y, z], |_| false,
            );
            assert_eq!(result, Err(ViscosityError::ArithmeticFailure));
            assert_eq!(output, saved);
        }
    }

    #[test]
    fn diffusion_stability_limit_remains_transactionally_rejected() {
        let (mut workspace, velocity) = fixture(0.0);
        let mut output = velocity.clone();
        let saved = output.clone();
        let [x, y, z] = &mut output;
        let result = workspace.update(
            [&velocity[0], &velocity[1], &velocity[2]], 1000.0, 0.001,
            1000.0, [x, y, z], |_| false,
        );
        assert!(matches!(result, Err(ViscosityError::StabilityLimit { .. })));
        assert_eq!(output, saved);
    }
}
