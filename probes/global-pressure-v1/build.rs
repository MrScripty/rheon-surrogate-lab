//! Only the lab-owned wrapper supplies a verified actual Cargo invocation.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-env-changed=RHEON_BUILD_BINDING_SHA256");
    println!("cargo:rerun-if-env-changed=RHEON_BUILD_BINDING");
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lab = root.parent().unwrap().parent().unwrap();
    let binding_path = PathBuf::from(
        env::var("RHEON_BUILD_BINDING").expect("direct Cargo is unsupported: use capture.py build"),
    );
    let bytes = fs::read(&binding_path).expect("build binding");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        env::var("RHEON_BUILD_BINDING_SHA256").expect("binding hash")
    );
    let binding: serde_json::Value = serde_json::from_slice(&bytes).expect("binding JSON");
    let prepared = PathBuf::from(binding["prepared_source"].as_str().unwrap())
        .canonicalize()
        .unwrap();
    let helper = lab.join("experiments/global-pressure-v1/prepare_dependency.py");
    let result = Command::new("python3")
        .arg(&helper)
        .arg("--verify")
        .arg(&prepared)
        .output()
        .expect("dependency verification helper");
    assert!(
        result.status.success(),
        "prepared dependency invalid: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&result.stdout).expect("receipt JSON");
    assert_eq!(receipt, binding["native_receipt"]);
    assert_eq!(receipt["schema_version"].as_u64(), Some(1));
    assert_eq!(
        receipt["base_commit"],
        "2209012bc5f902693d0f695ec1c2d854126af317"
    );
    assert_eq!(
        binding["resolved_native_manifest"].as_str().unwrap(),
        prepared.join("Cargo.toml").to_str().unwrap()
    );
    let identity = serde_json::json!({"binding":binding,"native_receipt":receipt,"native_threads":1,"parallel":env::var_os("CARGO_FEATURE_PARALLEL").is_some(),"source_sha256":format!("{:x}",Sha256::digest(fs::read(root.join("src/main.rs")).unwrap())),"config_sha256":format!("{:x}",Sha256::digest(fs::read(lab.join("experiments/global-pressure-v1/capture_config.json")).unwrap())),"reservation_sha256":format!("{:x}",Sha256::digest(fs::read(lab.join("docs/plans/global-pressure-correction-v1/reservation.json")).unwrap()))});
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("identity.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();
    println!("cargo:rerun-if-changed={}", binding_path.display());
    // Watch every qualified native file, including directories for insertion.
    fn watch(path: &std::path::Path) {
        println!("cargo:rerun-if-changed={}", path.display());
        if path.is_dir() {
            for child in fs::read_dir(path).unwrap() {
                watch(&child.unwrap().path());
            }
        }
    }
    watch(&prepared);
    for path in [
        root.join("src/main.rs"),
        helper,
        lab.join("experiments/global-pressure-v1/capture_config.json"),
        lab.join("docs/plans/global-pressure-correction-v1/reservation.json"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
