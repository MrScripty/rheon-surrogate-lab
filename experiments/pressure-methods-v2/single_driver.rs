use anyhow::{Result, ensure};
use rheon::diagnostic::Scene;
use rheon::diagnostic::engine::SingleTrackSession;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
fn send(value: &Value) -> Result<()> {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?; out.flush()?; Ok(())
}
fn main() -> Result<()> {
    let native = std::env::args().any(|x| x == "--native");
    let loose = std::env::args().any(|x| x == "--native-1e-6");
    let mut input = io::stdin().lock();
    let mut line = String::new();
    input.read_line(&mut line)?;
    ensure!(line.len() <= 65536, "bounded scene");
    let scene: Scene = serde_json::from_str(&line)?;
    let tick = std::time::Instant::now();
    let mut session = SingleTrackSession::new(scene.clone()).map_err(anyhow::Error::msg)?;
    send(&json!({"kind":"ready", "setup_seconds": tick.elapsed().as_secs_f64()}))?;
    for _ in 0..scene.steps {
        let step = session.advance(native, if loose {1e-6} else {1e-8}, |base, residual| {
            send(&json!({"kind":"request", "base":base, "residual":residual}))
                .map_err(|e| e.to_string())?;
            let mut raw = String::new();
            input.read_line(&mut raw).map_err(|e| e.to_string())?;
            if raw.is_empty() || raw.len() > 32*1024*1024 { return Err("bounded pressure reply".into()); }
            serde_json::from_str(&raw).map_err(|e| e.to_string())
        }).map_err(anyhow::Error::msg)?;
        let advanced = step.advanced;
        send(&json!({"kind":"frame", "frame":step}))?;
        if !advanced { break; }
    }
    send(&json!({"kind":"complete"}))?;
    Ok(())
}
