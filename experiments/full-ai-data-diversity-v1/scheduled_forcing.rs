//! Canonical prescribed controls, independent of the numerical pressure solver.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForceFrame {
    pub center: [f64; 3],
    pub pulse: f64,
}

static CONTROLS: OnceLock<Vec<ForceFrame>> = OnceLock::new();

pub fn validate(frames: &[ForceFrame], steps: usize) -> Result<(), String> {
    if steps == 0 || steps > 600 || frames.len() != steps {
        return Err("One prescribed force control per actual step, at most600 steps".into());
    }
    if frames.iter().any(|f| {
        !f.pulse.is_finite()
            || !(0.0..=2.0).contains(&f.pulse)
            || f.center.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    }) {
        return Err("Finite bounded pulse and normalized center required".into());
    }
    Ok(())
}

pub fn configure(frames: Vec<ForceFrame>, steps: usize) -> Result<(), String> {
    validate(&frames, steps)?;
    CONTROLS.set(frames).map_err(|_| "Force controls already configured".into())
}

pub fn sample(time: f64) -> Result<Option<&'static ForceFrame>, String> {
    let Some(frames) = CONTROLS.get() else { return Ok(None); };
    let step = (time * 60.0).round();
    if !time.is_finite() || time < 0.0 || (time * 60.0 - step).abs() > 1e-8 {
        return Err("Force controls require an actual60Hz integration boundary".into());
    }
    frames.get(step as usize).map(Some).ok_or_else(|| "Force control exhausted".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_have_explicit_count_and_finite_bounded_values() {
        let mut frames = vec![ForceFrame { center: [0.5; 3], pulse: 1.0 }; 2];
        assert!(validate(&frames, 2).is_ok());
        assert!(validate(&frames, 3).is_err());
        frames[0].pulse = f64::NAN;
        assert!(validate(&frames, 2).is_err());
        frames[0].pulse = 1.0;
        frames[1].center[2] = 1.001;
        assert!(validate(&frames, 2).is_err());
        assert!(validate(&[], 0).is_err());
    }
}
