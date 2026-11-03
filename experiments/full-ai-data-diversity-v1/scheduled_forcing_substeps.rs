//! Reference-only internal timesteps with canonical60Hz output controls.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForceFrame {
    pub center: [f64; 3],
    pub pulse: f64,
}

struct Controls {
    frames: Option<Vec<ForceFrame>>,
    substeps: usize,
}
static CONTROLS: OnceLock<Controls> = OnceLock::new();

pub fn substeps() -> usize {
    CONTROLS.get().map_or(1, |c| c.substeps)
}

pub fn dt() -> f64 {
    1.0 / (60.0 * substeps() as f64)
}

fn validate(frames: Option<&[ForceFrame]>, steps: usize, factor: usize) -> Result<(), String> {
    if ![1, 2, 4, 8].contains(&factor) || !(1..=600).contains(&steps)
        || steps.checked_mul(factor).is_none_or(|v| v > 3600)
    {
        return Err("Bounded macro count and one/two/four/eight reference substeps required".into());
    }
    if let Some(frames) = frames {
        if frames.len() != steps || frames.iter().any(|f| {
            !f.pulse.is_finite() || !(0.0..=2.0).contains(&f.pulse)
                || f.center.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        }) {
            return Err("One finite prescribed control per macro step required".into());
        }
    }
    Ok(())
}

pub fn configure(frames: Option<Vec<ForceFrame>>, steps: usize, factor: usize) -> Result<(), String> {
    validate(frames.as_deref(), steps, factor)?;
    CONTROLS.set(Controls { frames, substeps: factor })
        .map_err(|_| "Reference controls already configured".into())
}

fn sample_at(controls: &Controls, time: f64) -> Result<Option<&ForceFrame>, String> {
    let Some(frames) = &controls.frames else { return Ok(None); };
    let fine_step = (time * (60.0 * controls.substeps as f64)).round();
    if !time.is_finite() || time < 0.0
        || (time * (60.0 * controls.substeps as f64) - fine_step).abs() > 1e-7
    {
        return Err("Reference forcing requires an actual fine-step boundary".into());
    }
    frames.get(fine_step as usize / controls.substeps).map(Some)
        .ok_or_else(|| "Reference force controls exhausted".into())
}

pub fn sample(time: f64) -> Result<Option<&'static ForceFrame>, String> {
    match CONTROLS.get() {
        Some(controls) => sample_at(controls, time),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_persist_across_fine_steps_and_change_at_macro_boundary() {
        let controls = Controls { frames: Some(vec![
            ForceFrame { center: [0.25; 3], pulse: 0.0 },
            ForceFrame { center: [0.75; 3], pulse: 1.0 }]), substeps: 4 };
        assert_eq!(sample_at(&controls, 3.0 / 240.0).unwrap().unwrap().pulse, 0.0);
        assert_eq!(sample_at(&controls, 4.0 / 240.0).unwrap().unwrap().pulse, 1.0);
        assert!(sample_at(&controls, 8.0 / 240.0).is_err());
        assert!(validate(controls.frames.as_deref(), 2, 4).is_ok());
        assert!(validate(controls.frames.as_deref(), 3, 4).is_err());
        assert!(validate(None, 600, 8).is_err());
        assert!(validate(None, 240, 8).is_ok());
    }
}
