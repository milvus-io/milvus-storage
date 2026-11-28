use std::sync::OnceLock;

use anyhow::{Result, anyhow, ensure};
use rustls::crypto::{CryptoProvider, default_fips_provider};

static INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

fn initialize() -> Result<()> {
    aws_lc_rs::try_fips_mode().map_err(|error| anyhow!("AWS-LC FIPS mode: {error:?}"))?;
    let version = aws_lc_rs::awslc_version();
    ensure!(
        version == "4.0.0",
        "expected AWS-LC FIPS 4.0.0, found {version}"
    );

    // Another bridge client may already have initialized a FIPS provider.
    if CryptoProvider::get_default().is_none() {
        let _ = default_fips_provider().install_default();
    }
    ensure!(
        CryptoProvider::get_default().is_some_and(|provider| provider.fips()),
        "Rustls default provider is outside FIPS mode"
    );
    Ok(())
}

pub(crate) fn require() -> Result<()> {
    INITIALIZED
        .get_or_init(|| initialize().map_err(|error| format!("{error:#}")))
        .as_ref()
        .copied()
        .map_err(|error| anyhow!("Rust TLS FIPS initialization failed: {error}"))
}
