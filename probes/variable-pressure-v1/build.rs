//! Bind the unchanged diagnostic wire to the selected prepared dependency.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn main() {
    for key in ["RHEON_BUILD_BINDING", "RHEON_BUILD_BINDING_SHA256"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let bytes = fs::read(env::var("RHEON_BUILD_BINDING").expect("use native.py build")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        env::var("RHEON_BUILD_BINDING_SHA256").unwrap()
    );
    let binding: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for (relative, expected) in binding["source_sha256"].as_object().unwrap() {
        let path = root.parent().unwrap().parent().unwrap().join(relative);
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
            expected.as_str().unwrap()
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let identity = serde_json::json!({"schema":"rheon-variable-pressure-native-v1", "native_receipt":binding["native_receipt"], "source_sha256":binding["source_sha256"], "binding":binding, "native_threads":1});
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("identity.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();
}
