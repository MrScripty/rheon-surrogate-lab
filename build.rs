use std::process::Command;

fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn main() {
    for path in [
        "src",
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        ".git/HEAD",
        ".git/index",
        ".git/refs/heads/main",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rustc-env=LAB_COMMIT={}", git(&["rev-parse", "HEAD"]));
    println!(
        "cargo:rustc-env=LAB_DIRTY={}",
        !git(&["status", "--porcelain"]).is_empty()
    );
}
