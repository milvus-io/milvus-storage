.PHONY: build test clean check examples

# Build the library
build:
	cargo build --release

# Run tests
test:
	cargo test

# Check compilation without building
check:
	cargo check

# Build examples
examples:
	cargo build --examples

# Build everything
all: build examples

# Clean build artifacts
clean:
	cargo clean

# Install dependencies and build
install: build

# Format code
format:
	cargo fmt

# Run clippy linter
lint:
	cargo clippy -- -D warnings

# Build documentation
docs:
	cargo doc --no-deps --open

# Run a specific example
run-example:
	@echo "Usage: make run-example EXAMPLE=basic_usage"
	@echo "Available examples: basic_usage, vector_search"
	cargo run --example $(EXAMPLE)

# Help
help:
	@echo "Available targets:"
	@echo "  build     - Build the library"
	@echo "  test      - Run tests"
	@echo "  check     - Check compilation"
	@echo "  examples  - Build examples"
	@echo "  clean     - Clean build artifacts"
	@echo "  format    - Format code"
	@echo "  lint      - Run clippy linter"
	@echo "  docs      - Build and open documentation"
	@echo "  run-example EXAMPLE=name - Run a specific example"
