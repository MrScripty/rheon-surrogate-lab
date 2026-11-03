//! Offline reference clock with explicit macro-held or fine-time controls.
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
    control_substeps: usize,
}
static CONTROLS: OnceLock<Controls> = OnceLock::new();

pub fn substeps() -> usize {
    CONTROLS.get().map_or(1, |c| c.substeps)
}

pub fn dt() -> f64 {
    1.0 / (60.0 * substeps() as f64)
}

fn validate(frames: Option<&[ForceFrame]>, steps: usize, factor: usize,
            control_factor: usize) -> Result<(), String> {
    if ![1, 2, 4, 8, 16, 32, 64].contains(&factor) || !(1..=1200).contains(&steps)
        || steps.checked_mul(factor).is_none_or(|v| v > 38400)
        || !(control_factor == 1 || control_factor == factor)
    {
        return Err("Bounded reference steps and macro-held or fine-time controls required".into());
    }
    if let Some(frames) = frames {
        if frames.len() != steps * control_factor || frames.iter().any(|f| {
            !f.pulse.is_finite() || !(0.0..=2.0).contains(&f.pulse)
                || f.center.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        }) {
            return Err("Finite prescribed controls at the declared cadence required".into());
        }
    } else if control_factor != 1 {
        return Err("Fine controls must be supplied explicitly".into());
    }
    Ok(())
}

pub fn configure(frames: Option<Vec<ForceFrame>>, steps: usize, factor: usize,
                 control_factor: usize) -> Result<(), String> {
    validate(frames.as_deref(), steps, factor, control_factor)?;
    CONTROLS.set(Controls { frames, substeps: factor, control_substeps: control_factor })
        .map_err(|_| "Reference controls already configured".into())
}

fn sample_at(controls: &Controls, time: f64) -> Result<Option<&ForceFrame>, String> {
    let Some(frames) = &controls.frames else { return Ok(None); };
    let scaled = time * (60.0 * controls.substeps as f64);
    let fine_step = scaled.round();
    if !time.is_finite() || time < 0.0 || (scaled - fine_step).abs() > 1e-7 {
        return Err("Reference forcing requires an actual fine-step boundary".into());
    }
    let stride = controls.substeps / controls.control_substeps;
    frames.get(fine_step as usize / stride).map(Some)
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

    fn frame(pulse: f64) -> ForceFrame {
        ForceFrame { center: [0.5; 3], pulse }
    }

    #[test]
    fn fine_controls_change_inside_an_output_interval() {
        let controls = Controls { frames: Some(vec![frame(0.), frame(1.), frame(0.), frame(0.)]),
                                  substeps: 4, control_substeps: 4 };
        assert!(validate(controls.frames.as_deref(), 1, 4, 4).is_ok());
        assert_eq!(sample_at(&controls, 0.).unwrap().unwrap().pulse, 0.);
        assert_eq!(sample_at(&controls, 1. / 240.).unwrap().unwrap().pulse, 1.);
        assert!(sample_at(&controls, 1. / 60.).is_err());
        assert!(sample_at(&controls, 0.5 / 240.).is_err());
    }

    #[test]
    fn macro_controls_remain_held_until_the_next_output_interval() {
        let controls = Controls { frames: Some(vec![frame(0.), frame(1.)]),
                                  substeps: 4, control_substeps: 1 };
        assert!(validate(controls.frames.as_deref(), 2, 4, 1).is_ok());
        assert_eq!(sample_at(&controls, 3. / 240.).unwrap().unwrap().pulse, 0.);
        assert_eq!(sample_at(&controls, 4. / 240.).unwrap().unwrap().pulse, 1.);
    }

    #[test]
    fn counts_cadence_and_finite_values_are_bounded() {
        assert!(validate(None, 600, 64, 1).is_ok());
        assert!(validate(None, 1200, 4, 1).is_ok());
        assert!(validate(None, 1201, 4, 1).is_err());
        assert!(validate(Some(&vec![frame(1.); 4800]), 1200, 4, 4).is_ok());
        assert!(validate(None, 601, 64, 1).is_err());
        assert!(validate(None, 2, 128, 1).is_err());
        assert!(validate(None, 2, 8, 8).is_err());
        assert!(validate(Some(&[frame(1.)]), 1, 4, 2).is_err());
        assert!(validate(Some(&[frame(f64::NAN)]), 1, 1, 1).is_err());
        assert!(validate(Some(&[frame(1.)]), 1, 4, 4).is_err());
    }
}
