# Talon telemetry compatibility patch

Source: `milvus-io/talon` at `c6ba5c97d09c2cec2d58eebec3c7d661521bb99b`, matching the Storage Talon client and transport revision.

This crate backports the request-local suppression change and regression test from [milvus-io/talon#594](https://github.com/milvus-io/talon/pull/594), commit `89caf4a69a3292981c50dc4b26dbbd65a3432778`. It adds `Operation::disabled()` and preserves a disabled request scope after process telemetry has been initialized. Storage needs this when the host removes its provider or supplies no parent; a zero root sampling ratio alone still creates a carrier and can enable v2 wire propagation.

The upstream source is otherwise unchanged. Workspace-inherited package and dependency settings are made explicit in Cargo.toml. The Git-source patch in the bridge manifest ensures the client, transport and Storage adapter share this one telemetry crate and its process configuration.

Remove this directory and the Cargo patch when the pinned Talon revision includes the suppression fix. Keep the provider-removal and missing-parent integration tests.
