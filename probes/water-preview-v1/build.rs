//! Bind configurable reservations to the same serial development producer.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn main() {
    for key in [
        "RHEON_BUILD_BINDING",
        "RHEON_BUILD_BINDING_SHA256",
        "RHEON_CAPTURE_RESERVATION",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let path = env::var("RHEON_BUILD_BINDING").expect("use pipeline.py capture-build");
    let bytes = fs::read(path).expect("build binding");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        env::var("RHEON_BUILD_BINDING_SHA256").unwrap()
    );
    let binding: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let reservation = env::var("RHEON_CAPTURE_RESERVATION").expect("explicit reservation");
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&reservation).unwrap())),
        binding["reservation_sha256"].as_str().unwrap()
    );
    println!("cargo:rerun-if-changed={reservation}");
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for (relative, expected) in binding["source_sha256"].as_object().unwrap() {
        let source = root.parent().unwrap().parent().unwrap().join(relative);
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(&source).unwrap())),
            expected.as_str().unwrap()
        );
        println!("cargo:rerun-if-changed={}", source.display());
    }
    let identity = serde_json::json!({"native_receipt":binding["native_receipt"], "reservation_sha256":binding["reservation_sha256"], "binding":binding, "native_threads":1, "parallel":env::var_os("CARGO_FEATURE_PARALLEL").is_some()});
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("identity.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();
}
