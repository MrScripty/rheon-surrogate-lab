//! Bind the actual prepared dependency and build invocation, without modifying it.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    for name in ["RHEON_BUILD_BINDING", "RHEON_BUILD_BINDING_SHA256"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lab = root.parent().unwrap().parent().unwrap();
    let path =
        PathBuf::from(env::var("RHEON_BUILD_BINDING").expect("use evaluate_native.py build"));
    let bytes = fs::read(&path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        env::var("RHEON_BUILD_BINDING_SHA256").unwrap()
    );
    let binding: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let prepared = PathBuf::from(binding["prepared_source"].as_str().unwrap())
        .canonicalize()
        .unwrap();
    let helper = lab.join("experiments/global-pressure-v1/prepare_dependency.py");
    let checked = Command::new("python3")
        .arg(&helper)
        .arg("--verify")
        .arg(&prepared)
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "invalid prepared source: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(receipt, binding["native_receipt"]);
    assert_eq!(
        receipt["base_commit"],
        "2209012bc5f902693d0f695ec1c2d854126af317"
    );
    assert_eq!(
        binding["resolved_native_manifest"],
        prepared.join("Cargo.toml").to_str().unwrap()
    );
    let identity = serde_json::json!({"schema":"rheon-global-pressure-training-native-v1","binding":binding,"native_receipt":receipt,"native_threads":1,"source_sha256":format!("{:x}",Sha256::digest(fs::read(root.join("src/main.rs")).unwrap()))});
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("identity.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();
    fn watch(p: &std::path::Path) {
        println!("cargo:rerun-if-changed={}", p.display());
        if p.is_dir() {
            for entry in fs::read_dir(p).unwrap() {
                watch(&entry.unwrap().path());
            }
        }
    }
    watch(&prepared);
    for p in [path, helper, root.join("src/main.rs")] {
        println!("cargo:rerun-if-changed={}", p.display());
    }
}
