use std::{path::Path, process::Command};
fn stamp(path: &Path, prefix: &str) {
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .arg("-C")
            .arg(path)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success());
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    };
    println!(
        "cargo:rustc-env={prefix}_COMMIT={}",
        git(&["rev-parse", "HEAD"])
    );
    println!(
        "cargo:rustc-env={prefix}_DIRTY={}",
        !git(&["status", "--porcelain"]).is_empty()
    );
    for file in [".git/HEAD", ".git/index", "src", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", path.join(file).display());
    }
    let head = git(&["rev-parse", "--symbolic-full-name", "HEAD"]);
    println!(
        "cargo:rerun-if-changed={}",
        path.join(".git").join(head).display()
    );
}
fn main() {
    let here = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let here = Path::new(&here);
    stamp(&here.join("../.."), "LAB");
    stamp(&here.join("../../../rheon-warmstart"), "PATCH");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
}
