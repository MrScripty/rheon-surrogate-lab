//! Bounded CPU wire around Rheon's public stop-on-reject diagnostic session.
use anyhow::{Context, Result, ensure};
use rheon::diagnostic::{
    AdvanceStop, DiagnosticSession, ExecutionPolicy, InputSemantic, OutputRole, PredictionFailure,
    RHO, Scene,
};
use rheon::recording::diagnostic::{DiagnosticAttempt, NoCandidateReason, PredictionOutcome};
use rheon::{GridGeometry, PressureOperator, PressureSettings, PressureWorkspace};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Read, Write};
use std::sync::atomic::AtomicBool;
const LIMIT: u64 = 8 * 1024 * 1024;
fn receive(input: &mut impl BufRead) -> Result<String> {
    let mut line = String::new();
    input.take(LIMIT + 1).read_line(&mut line)?;
    ensure!(
        !line.is_empty() && line.len() as u64 <= LIMIT && line.ends_with('\n'),
        "bounded complete input"
    );
    Ok(line)
}
fn send(value: &Value) -> Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
fn attempt_summary(attempt: &DiagnosticAttempt, stopped: bool) -> Value {
    let request = &attempt.request;
    let comparison = match attempt.pressure_comparison() {
        Ok(comparison) => {
            json!({"status":"available", "relative_l2":comparison.metrics.relative_l2,
            "l2":comparison.metrics.l2, "rmse":comparison.metrics.rmse, "linf":comparison.metrics.linf})
        }
        Err(reason) => json!({"status":"unavailable", "reason":format!("{reason:?}")}),
    };
    let outcome = match &attempt.outcome {
        PredictionOutcome::Candidate(candidate) => json!({"kind":"candidate",
            "qualified":candidate.observation.qualified, "hard_stop":candidate.observation.hard_stop,
            "relative_residual":candidate.observation.relative_residual,
            "residual_l2":candidate.observation.residual_l2,
            "actual_divergence":candidate.observation.actual_divergence,
            "predicted_divergence":candidate.observation.predicted_divergence,
            "courant":candidate.observation.courant, "net_flux":candidate.observation.net_flux,
            "walls_closed":candidate.observation.walls_closed,
            "relative_limit":candidate.observation.relative_limit,
            "absolute_limit":candidate.observation.absolute_limit,
            "predicted_limit":candidate.observation.predicted_limit,
            "actual_limit":candidate.observation.actual_limit,
            "courant_limit":candidate.observation.courant_limit,
            "flux_limit":candidate.observation.flux_limit}),
        PredictionOutcome::NoCandidate {
            reason,
            detail,
            invalid_payload_sha256,
        } => json!({
            "kind":"no_candidate", "reason":format!("{reason:?}"), "detail":detail,
            "invalid_payload_sha256":invalid_payload_sha256}),
    };
    let mut result = json!({"attempt_index":attempt.attempt_index, "generation":request.generation(),
        "interval_start_s":request.interval_start(), "interval_end_s":request.interval_end(),
        "request_sha256":request.digest(), "dispatched":attempt.dispatched,
        "history_transition":format!("{:?}",attempt.transition), "outcome":outcome,
        "same_request_pressure_comparison":comparison, "native_pressure_refinement":false,
        "native_fallback":false});
    if stopped {
        result["raw_request"] = json!({"counts_xyz":request.grid().counts(),
            "spacing_m":request.grid().spacing(), "origin_m":request.grid().origin(),
            "density_kg_m3":request.density(), "dt_s":request.dt(), "viscosity_pa_s":request.viscosity(),
            "rhs":request.rhs(), "native_input_residual":request.initial_residual(),
            "prior_pressure":request.prior().map(|prior| &prior.pressure)});
        result["raw_candidate_pressure"] = match &attempt.outcome {
            PredictionOutcome::Candidate(candidate) => json!(candidate.pressure),
            PredictionOutcome::NoCandidate { .. } => Value::Null,
        };
        result["same_request_native_pressure"] = json!(
            attempt
                .native_reference
                .as_ref()
                .map(|native| &native.sample.pressure)
        );
    }
    result
}
fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let (control, adapter_inputs, output_role): (
        bool,
        BTreeMap<String, InputSemantic>,
        OutputRole,
    ) = match arguments.as_slice() {
        [flag] if flag == "--reference-control" => {
            (true, BTreeMap::new(), OutputRole::FullPressure)
        }
        [flag, raw, role_flag, role]
            if flag == "--adapter-inputs"
                && raw.len() <= 65536
                && role_flag == "--adapter-output-role" =>
        {
            (
                false,
                serde_json::from_str(raw)?,
                serde_json::from_value(json!(role))?,
            )
        }
        _ => anyhow::bail!(
            "select native reference control or current adapter inputs and output role"
        ),
    };
    let mut input = io::stdin().lock();
    let scene: Scene = serde_json::from_str(&receive(&mut input)?)?;
    scene.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        scene.warmup_steps == 15,
        "migration requires 15 native warmup steps"
    );
    let grid = GridGeometry::new(scene.counts.map(|n| n as u64), [scene.h(); 3], [0.0; 3])?;
    let operator = PressureOperator::new(&grid, RHO)?;
    let mut solver = PressureWorkspace::new(&grid, 128 * 1024 * 1024)?;
    let mut session = DiagnosticSession::with_policy(
        scene.clone(),
        ExecutionPolicy::IndependentStopOnReject,
        format!("lab-residual:{}", std::process::id()),
    )
    .map_err(anyhow::Error::msg)?;
    let cancel = AtomicBool::new(false);
    let mut ai_steps = 0;
    let mut qualified = 0;
    let mut stopped = false;
    for _ in 0..scene.steps {
        let mut invalid_reply = None;
        let result = session
            .advance_typed(&cancel, |request| {
                if control {
                    let prediction = (|| -> Result<Vec<f64>> {
                        // Retain the bound control's exact RHS reconstruction arithmetic.
                        let mut rhs = vec![0.0; grid.cell_len()];
                        operator.apply_full(request.initial_guess, &mut rhs)?;
                        for (value, residual) in rhs.iter_mut().zip(request.residual) {
                            *value += residual;
                        }
                        solver.solve_rhs(
                            &operator,
                            &rhs,
                            request.context.dt(),
                            PressureSettings {
                                relative_residual: 1e-8,
                                absolute_residual: 1e-12,
                                divergence_limit: 1e-7,
                                max_iterations: 3000,
                            },
                            || false,
                        )?;
                        Ok(solver.pressure().to_vec())
                    })();
                    return prediction.map_err(|error| PredictionFailure {
                        reason: NoCandidateReason::RuntimeFailure,
                        detail: format!("native control: {error:#}"),
                        dispatched: false,
                        invalid_payload_sha256: None,
                    });
                }
                let packet = request.adapter_request(&adapter_inputs)?;
                send(&packet).map_err(|error| PredictionFailure {
                    reason: NoCandidateReason::RuntimeFailure,
                    detail: format!("send model request: {error:#}"),
                    dispatched: false,
                    invalid_payload_sha256: None,
                })?;
                let raw = receive(&mut input).map_err(|error| PredictionFailure {
                    reason: NoCandidateReason::RuntimeFailure,
                    detail: format!("receive model reply: {error:#}"),
                    dispatched: true,
                    invalid_payload_sha256: None,
                })?;
                invalid_reply = Some(raw.clone());
                request.adapter_reply(raw.as_bytes(), output_role)
            })
            .map_err(anyhow::Error::msg)
            .context("public diagnostic advance")?;
        stopped = result.stop.is_some();
        let diagnostic = result.attempt.as_ref().map(|attempt| {
            ai_steps += 1;
            if let PredictionOutcome::Candidate(candidate) = &attempt.outcome {
                qualified += usize::from(candidate.observation.qualified);
            }
            attempt_summary(attempt, stopped)
        });
        if let Some(frame) = result.frame {
            let pressure_error = frame
                .ai_pressure
                .iter()
                .zip(&frame.reference_pressure)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let reference_norm = frame
                .reference_pressure
                .iter()
                .map(|p| p * p)
                .sum::<f64>()
                .sqrt();
            send(
                &json!({"kind":"frame", "phase":frame.phase,"step":frame.step,"time_s":frame.time_s,
                "reference_quality":frame.reference_quality,"ai_quality":frame.ai_quality,
                "relative_pressure_difference":pressure_error/reference_norm.max(1e-30),
                "pressure_comparison_scope":"independent_trajectories",
                "diagnostic_attempt":diagnostic,
                "reference_seconds":frame.reference_seconds,"ai_seconds":frame.ai_seconds}),
            )?;
        } else if let Some(mut diagnostic) = diagnostic {
            if matches!(
                result.attempt.as_ref().map(|a| &a.outcome),
                Some(PredictionOutcome::NoCandidate { .. })
            ) {
                diagnostic["raw_reply"] = json!(invalid_reply);
            }
            send(&json!({"kind":"diagnostic_attempt", "attempt":diagnostic,
                "stop":format!("{:?}",result.stop)}))?;
        }
        if let Some(stop) = result.stop {
            ensure!(
                matches!(
                    stop,
                    AdvanceStop::Rejected | AdvanceStop::ProviderFailure(_)
                ),
                "diagnostic cancelled"
            );
            break;
        }
    }
    send(
        &json!({"kind":"complete","ai_steps":ai_steps,"qualified_ai_steps":qualified,"hard_stop":stopped,
        "reference_control":control,"warmup_steps":scene.warmup_steps,
        "execution_policy":"independent_stop_on_reject", "native_pressure_refinement":false,"native_fallback":false}),
    )
}
