use anyhow::{Context, Result, ensure};
use rheon::{Axis, GridGeometry, PressureOperator};
use serde_json::{Value, json};
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut text = String::new();
    io::stdin().take(64 * 1024 * 1024 + 1).read_to_string(&mut text)?;
    ensure!(text.len() <= 64 * 1024 * 1024, "bounded teacher request");
    let packet: Value = serde_json::from_str(&text)?;
    let cases = packet["cases"].as_array().context("cases required")?;
    ensure!(!cases.is_empty() && cases.len() <= 16, "bounded teacher cases");
    let mut result = Vec::new();
    for case in cases {
        let mut counts = [0_u64; 3];
        for (d, n) in counts.iter_mut().enumerate() {
            *n = case["counts"][d].as_u64().context("counts required")?;
        }
        let h = case["cell_size_m"].as_f64().context("spacing required")?;
        let dt = case["dt_s"].as_f64().context("timestep required")?;
        let rho = case["rho"].as_f64().context("density required")?;
        let grid = GridGeometry::new(counts, [h; 3], [0.; 3])?;
        let mut velocity: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for (d, field) in velocity.iter_mut().enumerate() {
            let input = case["velocity"][d].as_array().context("faces required")?;
            ensure!(input.len() == grid.face_len([Axis::X, Axis::Y, Axis::Z][d]), "face length");
            for value in input {
                field.push(value.as_f64().context("finite face value required")? as f32);
            }
        }
        let operator = PressureOperator::new(&grid, rho)?;
        let mut rhs = vec![0.; grid.cell_len()];
        operator.build_rhs([&velocity[0], &velocity[1], &velocity[2]], dt, &mut rhs)?;
        result.push(json!({"rhs":rhs,"counts":counts,"native_steps_executed":0,"pressure_solves":0}));
    }
    println!("{}", serde_json::to_string(&json!({"cases":result}))?);
    Ok(())
}
