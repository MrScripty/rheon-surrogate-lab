//! Bounded binary transport for one-track pressure requests through 256^3.
use anyhow::{Result, ensure};
use rheon::diagnostic::Scene;
use rheon::diagnostic::engine::SingleTrackSession;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

fn send(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
fn field(out: &mut impl Write, values: &[f64]) -> Result<()> {
    let mut block = Vec::with_capacity(65536);
    for chunk in values.chunks(8192) {
        block.clear();
        for value in chunk { block.extend_from_slice(&value.to_le_bytes()); }
        out.write_all(&block)?;
    }
    Ok(())
}
fn pressure(input: &mut impl Read, count: usize) -> Result<Vec<f64>> {
    ensure!((1..=256usize.pow(3)).contains(&count), "bounded pressure field");
    let mut values = Vec::with_capacity(count);
    let mut block = [0_u8; 65536];
    while values.len() < count {
        let n = (count-values.len()).min(8192);
        input.read_exact(&mut block[..8*n])?;
        for bytes in block[..8*n].chunks_exact(8) {
            values.push(f64::from_le_bytes(bytes.try_into()?));
        }
    }
    Ok(values)
}
fn main() -> Result<()> {
    let native = std::env::args().any(|arg| arg == "--native");
    let mut input = BufReader::with_capacity(1024*1024, io::stdin());
    let mut out = BufWriter::with_capacity(1024*1024, io::stdout());
    let mut raw = String::new();
    input.by_ref().take(65537).read_line(&mut raw)?;
    ensure!(raw.len() <= 65536 && raw.ends_with('\n'), "bounded scene");
    let scene: Scene = serde_json::from_str(&raw)?;
    ensure!(scene.steps <= 4, "bounded large-grid coupling probe, not a long rollout");
    let tick = std::time::Instant::now();
    let mut session = SingleTrackSession::new(scene.clone()).map_err(anyhow::Error::msg)?;
    send(&mut out, &json!({"kind":"ready", "setup_seconds":tick.elapsed().as_secs_f64()}))?;
    for _ in 0..scene.steps {
        let step = session.advance(native, 1e-8, |base, residual| {
            let exchange = (|| -> Result<Vec<f64>> {
                send(&mut out, &json!({"kind":"predict", "cells":base.len(),
                    "counts_xyz":scene.counts, "cell_size_m":scene.h(), "rho":1000., "dt":1./60.,
                    "dtype":"<f8", "fields":["base","native_input_residual"]}))?;
                field(&mut out, base)?;
                field(&mut out, residual)?;
                out.flush()?;
                pressure(&mut input, base.len())
            })();
            exchange.map_err(|error| error.to_string())
        }).map_err(anyhow::Error::msg)?;
        let advanced = step.advanced;
        send(&mut out, &json!({"kind":"frame", "frame":step}))?;
        if !advanced { break; }
    }
    send(&mut out, &json!({"kind":"complete"}))?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_fields_roundtrip_across_block_boundary_without_precision_loss() {
        let values: Vec<f64> = (0..9001).map(|i| (i as f64-4500.)/17.).collect();
        let mut encoded = Vec::new();
        field(&mut encoded, &values).unwrap();
        assert_eq!(encoded.len(), values.len()*8);
        assert_eq!(pressure(&mut encoded.as_slice(), values.len()).unwrap(), values);
        assert!(pressure(&mut encoded[..8].as_ref(), 2).is_err());
    }
}
