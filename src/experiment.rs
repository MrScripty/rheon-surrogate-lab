use anyhow::{Result, ensure};
use rheon::{
    Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace,
};
use serde::{Deserialize, Serialize};

pub const RHEON_COMMIT: &str = "34235d33c886aa05b5bdf95d920b863d1fb2ad6f";
pub const COUNTS: [usize; 3] = [8, 7, 6];
pub const SPACING: [f64; 3] = [0.25, 0.3, 0.4];
pub const DENSITY: f64 = 1.25;
pub const DT: f64 = 0.03125;
pub const ACTUAL_DIVERGENCE_LIMIT: f64 = 1e-5;
pub type Velocity = [Vec<f32>; 3];

pub fn grid() -> Result<GridGeometry> {
    Ok(GridGeometry::new(
        COUNTS.map(|n| n as u64),
        SPACING,
        [0.0; 3],
    )?)
}

pub fn settings() -> PressureSettings {
    PressureSettings {
        relative_residual: 1e-10,
        absolute_residual: 1e-12,
        divergence_limit: 1e-9,
        max_iterations: 1000,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseSpec {
    pub id: String,
    pub family: String,
    pub seed: u64,
    pub split: String,
    /// Each case is a standalone preprojection field, never a neighboring frame.
    pub trajectory: String,
}

pub fn case_plan(per_family: usize) -> Result<Vec<CaseSpec>> {
    ensure!(
        (1..=64).contains(&per_family),
        "case_count: per-family must be 1..=64"
    );
    let mut cases = Vec::new();
    for (family, split, base) in [
        ("white_noise", "train", 1000),
        ("polynomial_mix", "train", 2000),
        ("localized_blob", "validation", 3000),
        ("oscillatory_modes", "test", 4000),
        ("zero", "check", 0),
        ("manufactured_gradient", "check", 1),
    ] {
        let n = if split == "check" { 1 } else { per_family };
        for index in 0..n {
            let seed = base + index as u64;
            let id = format!("{family}-{seed}");
            cases.push(CaseSpec {
                trajectory: id.clone(),
                id,
                family: family.into(),
                seed,
                split: split.into(),
            });
        }
    }
    Ok(cases)
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn signed(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        2.0 * ((x >> 40) as f64 / 16_777_216.0) - 1.0
    }
}

pub fn velocity(spec: &CaseSpec) -> Result<Velocity> {
    let g = grid()?;
    let mut out: Velocity =
        std::array::from_fn(|d| vec![0.0; g.face_len([Axis::X, Axis::Y, Axis::Z][d])]);
    if spec.family == "zero" {
        return Ok(out);
    }
    if spec.family == "manufactured_gradient" {
        // A gradient input: reference pressure is still solved from the actual f32 faces.
        let exact: Vec<f64> = (0..g.cell_len())
            .map(|i| {
                let x = (i % COUNTS[0]) as f64;
                let y = ((i / COUNTS[0]) % COUNTS[1]) as f64;
                let z = (i / (COUNTS[0] * COUNTS[1])) as f64;
                -(x * x + 0.7 * y * z + 0.3 * z * z + 0.2 * x * y * z)
            })
            .collect();
        let zero = out.clone();
        let [x, y, z] = &mut out;
        PressureOperator::new(&g, DENSITY)?.correct_velocity(refs(&zero), &exact, DT, [x, y, z])?;
        return Ok(out);
    }
    ensure!(
        [
            "white_noise",
            "polynomial_mix",
            "localized_blob",
            "oscillatory_modes"
        ]
        .contains(&spec.family.as_str()),
        "family: unsupported"
    );
    let mut rng = SplitMix64(spec.seed);
    for (d, field) in out.iter_mut().enumerate() {
        let c = [rng.signed(), rng.signed(), rng.signed(), rng.signed()];
        let shape = g.face_counts([Axis::X, Axis::Y, Axis::Z][d]);
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let ijk = [i, j, k];
                    if ijk[d] == 0 || ijk[d] == COUNTS[d] {
                        continue;
                    }
                    let pos: [f64; 3] = std::array::from_fn(|a| {
                        (ijk[a] as f64 + if a == d { 0.0 } else { 0.5 }) / COUNTS[a] as f64
                    });
                    let [x, y, z] = pos;
                    let envelope = 4.0 * pos[d] * (1.0 - pos[d]);
                    let value = match spec.family.as_str() {
                        "white_noise" => rng.signed(),
                        "polynomial_mix" => {
                            envelope
                                * (c[0] * (2.0 * x - 1.0) * (2.0 * y - 1.0)
                                    + c[1] * (2.0 * y - 1.0) * (2.0 * z - 1.0)
                                    + c[2] * x * y * z
                                    + c[3])
                        }
                        "localized_blob" => {
                            let r2 = (x - 0.5 - 0.15 * c[0]).powi(2)
                                + (y - 0.5 - 0.15 * c[1]).powi(2)
                                + (z - 0.5 - 0.15 * c[2]).powi(2);
                            envelope * c[3] * (1.0 - 4.0 * r2).max(0.0).powi(3)
                        }
                        _ => {
                            envelope
                                * (std::f64::consts::TAU * (x + 2.0 * y + z) + c[0]).sin()
                                * (c[1] + c[2] * z + c[3] * x * y)
                        }
                    };
                    field[g.face_index([Axis::X, Axis::Y, Axis::Z][d], ijk).unwrap()] =
                        (0.5 * value) as f32;
                }
            }
        }
    }
    Ok(out)
}

pub fn refs(v: &Velocity) -> [&[f32]; 3] {
    [&v[0], &v[1], &v[2]]
}
fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub rhs_l2: f64,
    pub rhs_sum: f64,
    pub compatibility_budget: f64,
    pub full_residual_l2: f64,
    pub full_residual_max: f64,
    pub predicted_divergence_max: f64,
    pub corrected_f32_divergence_max: f64,
    pub initial_divergence_max: f64,
}

pub struct Qualified {
    pub velocity: Velocity,
    pub rhs: Vec<f64>,
    pub pressure: Vec<f64>,
    pub corrected: Velocity,
    pub metrics: Metrics,
    pub iterations: usize,
}

/// Recompute physical gates from data, without trusting the solver's success flag.
pub fn qualify(v: &Velocity, b: &[f64], p: &[f64]) -> Result<(Metrics, Velocity)> {
    let g = grid()?;
    let op = PressureOperator::new(&g, DENSITY)?;
    ensure!(
        b.len() == g.cell_len() && p.len() == g.cell_len(),
        "shape: cell array"
    );
    ensure!(
        b.iter().chain(p).all(|x| x.is_finite()),
        "finite: cell array"
    );
    ensure!(p[0] == 0.0, "gauge: pressure[0] must equal zero");
    let mut rebuilt = vec![0.0; b.len()];
    op.build_rhs(refs(v), DT, &mut rebuilt)?;
    ensure!(rebuilt == b, "rhs: does not match preprojection velocity");
    let sum = b.iter().sum::<f64>();
    let budget = 64.0 * f64::EPSILON * b.iter().map(|x| x.abs()).sum::<f64>();
    ensure!(
        sum.abs() <= budget,
        "compatibility: sealed RHS sum exceeds rounding budget"
    );
    let mut ap = vec![0.0; b.len()];
    op.apply_full(p, &mut ap)?;
    let residual: Vec<f64> = b.iter().zip(ap).map(|(b, ap)| b - ap).collect();
    let residual_l2 = l2(&residual);
    let residual_max = max_abs(&residual);
    let rhs_l2 = l2(b);
    let predicted = residual_max * DT / g.cell_volume();
    ensure!(
        residual_l2
            <= settings()
                .absolute_residual
                .max(settings().relative_residual * rhs_l2),
        "residual: full b-Ap exceeds tolerance"
    );
    ensure!(
        predicted <= settings().divergence_limit,
        "predicted_divergence: exceeds tolerance"
    );
    let mut corrected = v.clone();
    let [x, y, z] = &mut corrected;
    op.correct_velocity(refs(v), p, DT, [x, y, z])?;
    let mut div = vec![0.0; b.len()];
    op.divergence(refs(&corrected), &mut div)?;
    let actual = max_abs(&div);
    ensure!(
        actual <= ACTUAL_DIVERGENCE_LIMIT,
        "actual_divergence: corrected f32 faces exceed tolerance"
    );
    op.divergence(refs(v), &mut div)?;
    Ok((
        Metrics {
            rhs_l2,
            rhs_sum: sum,
            compatibility_budget: budget,
            full_residual_l2: residual_l2,
            full_residual_max: residual_max,
            predicted_divergence_max: predicted,
            corrected_f32_divergence_max: actual,
            initial_divergence_max: max_abs(&div),
        },
        corrected,
    ))
}

pub fn solve(spec: &CaseSpec, method: PressureImplementation) -> Result<Qualified> {
    let g = grid()?;
    let op = PressureOperator::new(&g, DENSITY)?;
    let v = velocity(spec)?;
    let mut b = vec![0.0; g.cell_len()];
    op.build_rhs(refs(&v), DT, &mut b)?;
    let mut ws = PressureWorkspace::with_implementation(&g, g.cell_len() * 48, method)?;
    let report = ws.solve_rhs(&op, &b, DT, settings(), || false)?;
    let p = ws.pressure().to_vec();
    let (metrics, corrected) = qualify(&v, &b, &p)?;
    Ok(Qualified {
        velocity: v,
        rhs: b,
        pressure: p,
        corrected,
        metrics,
        iterations: report.iterations,
    })
}
