//! Include the selected existing Rheon comparison engine without editing its checkout.
use std::{env, fs, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=RHEON_COMPARISON_SOURCE");
    let root =
        PathBuf::from(env::var("RHEON_COMPARISON_SOURCE").expect("explicit comparison source"));
    let mut modules = String::new();
    for name in ["scene", "engine"] {
        let path = root
            .join(format!("src/bin/desktop/ai_comparison/{name}.rs"))
            .canonicalize()
            .unwrap();
        println!("cargo:rerun-if-changed={}", path.display());
        modules.push_str(&format!(
            "#[path={:?}] pub mod {name};\n",
            path.to_str().unwrap()
        ));
    }
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("selected.rs"),
        modules,
    )
    .unwrap();
}
