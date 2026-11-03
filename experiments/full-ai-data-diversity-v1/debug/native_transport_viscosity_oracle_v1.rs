//! DEBUG ONLY: bounded native primitives for stage diagnosis, never inference.
use rheon::{Axis, GridGeometry, ViscosityWorkspace, advect_tracer, advect_velocity};
use std::{error::Error, fs, path::Path};

fn read(path: &Path, length: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != 4 * length {
        return Err("Exact bounded native float layout required".into());
    }
    let values: Vec<f32> = bytes.chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Finite native input required".into());
    }
    Ok(values)
}

fn write(path: &Path, values: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
    fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 9 { return Err("mode nx ny nz h dt input output required".into()); }
    let counts: [u64; 3] = [args[2].parse()?, args[3].parse()?, args[4].parse()?];
    if counts.iter().any(|n| !(5..=128).contains(n)) {
        return Err("Registered diagnostic grid cap exceeded".into());
    }
    let h: f64 = args[5].parse()?;
    let dt: f64 = args[6].parse()?;
    if !h.is_finite() || h <= 0. || !dt.is_finite() || dt <= 0. {
        return Err("Finite positive spacing and timestep required".into());
    }
    let input = Path::new(&args[7]);
    let output = Path::new(&args[8]);
    fs::create_dir(output)?;
    let grid = GridGeometry::new(counts, [h; 3], [0.; 3])?;
    let names = ["vx", "vy", "vz"];
    let axes = [Axis::X, Axis::Y, Axis::Z];
    let old: [Vec<f32>; 3] = [
        read(&input.join("vx.f32"), grid.face_len(axes[0]))?,
        read(&input.join("vy.f32"), grid.face_len(axes[1]))?,
        read(&input.join("vz.f32"), grid.face_len(axes[2]))?];
    let mut vx = vec![0.; old[0].len()];
    let mut vy = vec![0.; old[1].len()];
    let mut vz = vec![0.; old[2].len()];
    let refs = [&old[0][..], &old[1][..], &old[2][..]];
    match args[1].as_str() {
        "transport" => {
            advect_velocity(&grid, refs, dt, [&mut vx, &mut vy, &mut vz], || false)?;
            let old_tracer = read(&input.join("tracer.f32"), grid.cell_len())?;
            let mut tracer = vec![0.; grid.cell_len()];
            advect_tracer(&grid, &old_tracer, refs, dt, &mut tracer, || false)?;
            write(&output.join("tracer.f32"), &tracer)?;
        }
        "viscosity" => {
            let mut workspace = ViscosityWorkspace::new(grid, 1024 * 1024 * 1024)?;
            workspace.update(refs, 1000., 0.001, dt,
                             [&mut vx, &mut vy, &mut vz], |_| false)?;
        }
        _ => return Err("Registered native primitive required".into()),
    }
    for (name, values) in names.iter().zip([vx, vy, vz]) {
        write(&output.join(format!("{name}.f32")), &values)?;
    }
    println!("Native diagnostic primitive completed: {}", args[1]);
    Ok(())
}
