use anyhow::{Context, Result, ensure};
use rheon_surrogate_lab::dataset;
use std::{path::Path, process::Command};

fn provenance() -> Result<String> {
    let git = |args: &[&str]| -> Result<String> {
        let output = Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR")])
            .args(args)
            .output()?;
        ensure!(output.status.success(), "provenance: git command failed");
        Ok(String::from_utf8(output.stdout)?.trim().to_owned())
    };
    let head = git(&["rev-parse", "HEAD"])?;
    ensure!(
        git(&["status", "--porcelain"])?.is_empty(),
        "provenance: commit source changes before generating a dataset"
    );
    ensure!(
        head == env!("LAB_COMMIT") && env!("LAB_DIRTY") == "false",
        "provenance: rebuild from the clean current commit"
    );
    Ok(head)
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: rheon-surrogate-lab generate DIR [PER_FAMILY=4] | verify DIR | evaluate DIR PREDICTIONS_DIR";
    let command = args.get(1).context(usage)?;
    let root = Path::new(args.get(2).context(usage)?);
    let mut result = match command.as_str() {
        "generate" if (3..=4).contains(&args.len()) => {
            let count = args.get(3).map(|n| n.parse()).transpose()?.unwrap_or(4);
            dataset::summary(&dataset::generate(root, count, &provenance()?)?)
        }
        "verify" if args.len() == 3 => dataset::summary(&dataset::verify(root)?),
        "evaluate" if args.len() == 4 => dataset::evaluate_predictions(root, Path::new(&args[3]))?,
        _ => anyhow::bail!(usage),
    };
    result["tool"] = serde_json::json!({
        "experiment_commit": env!("LAB_COMMIT"),
        "source_dirty": env!("LAB_DIRTY") != "false",
        "schema": dataset::SCHEMA,
        "rheon_commit": rheon_surrogate_lab::experiment::RHEON_COMMIT
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
