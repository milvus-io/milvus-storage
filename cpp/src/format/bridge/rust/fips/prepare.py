"""Stage the Rust FIPS dependency patches without changing the ordinary Cargo build."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


FIPS_DIR = Path(__file__).resolve().parent
BRIDGE = FIPS_DIR.parent
VERSIONS = {
    "aws-lc-fips-sys": "0.14.2",
    "aws-lc-rs": "1.18.2",
    "rustls": "0.23.46",
}
DISALLOWED_BACKENDS = {"ring", "aws-lc-sys", "native-tls", "openssl", "openssl-sys"}


def output(command, *, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True)


def patch_source(patch, source):
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=source, check=True)


def copy_git_source(package, destination):
    source = Path(package["manifest_path"]).parent
    root = Path(output(["git", "rev-parse", "--show-toplevel"], cwd=source).strip())
    revision = package["source"].rsplit("#", 1)[1]
    destination.mkdir()
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "archive", revision], cwd=root, stdout=archive, check=True)
        archive.seek(0)
        subprocess.run(["tar", "-xf", "-"], cwd=destination, stdin=archive, check=True)
    return root


def stage_dependencies(cargo, target, stage, patches):
    metadata = json.loads(output([
        cargo, "metadata", "--locked", "--format-version", "1", "--features", "talon",
        "--filter-platform", target, "--manifest-path", BRIDGE / "Cargo.toml",
    ]))
    packages = metadata["packages"]
    manifest = (BRIDGE / "Cargo.toml").read_text()
    registry_overrides = []
    dependencies = stage / "dependencies"
    dependencies.mkdir()

    for patch in patches:
        package_name, version = patch.stem.rsplit("-", 1)
        candidates = [p for p in packages if p["name"] == package_name and p["version"] == version]
        destination = dependencies / patch.stem
        if len(candidates) == 1 and candidates[0]["source"].startswith("registry+"):
            shutil.copytree(Path(candidates[0]["manifest_path"]).parent, destination)
            patch_source(patch, destination)
            # Multiple versions of a crate can coexist under separate patch keys.
            relative = destination.relative_to(stage).as_posix()
            patch_key = patch.stem.replace(".", "_")
            registry_overrides.append(
                f'{patch_key} = {{ package = "{package_name}", path = "{relative}" }}'
            )
            continue

        git_packages = [p for p in packages if (p["source"] or "").startswith("git+")
                        and p["source"].split("#")[-1].startswith(version)]
        if not git_packages:
            raise ValueError(f"patch {patch.name} does not match Cargo.lock")
        root = copy_git_source(git_packages[0], destination)
        patch_source(patch, destination)
        url = git_packages[0]["source"][4:].split("?", 1)[0]
        revision = git_packages[0]["source"].rsplit("#", 1)[1]
        git_paths = {p["name"]: (destination / Path(p["manifest_path"]).parent.relative_to(root))
                     for p in git_packages}

        def replace(match):
            dependency, settings = match.groups()
            alias = re.search(r'package = "([^"]+)"', settings)
            name = alias[1] if alias else dependency
            relative = git_paths[name].relative_to(stage).as_posix()
            remaining = settings.replace(f'git = "{url}", rev = "{revision}"', f'path = "{relative}"')
            if name == "talon-rust-client":
                remaining += ', features = ["fips"]'
            return f"{dependency} = {{ {remaining} }}"

        manifest, count = re.subn(
            rf'(?m)^([\w-]+) = \{{ ([^\n]*git = "{re.escape(url)}", rev = "{revision}"[^\n]*?) \}}$',
            replace, manifest,
        )
        if not count:
            raise ValueError(f"no bridge dependencies replaced by {patch.name}")

    patch_section = "[patch.crates-io]\n"
    manifest = manifest.replace(patch_section, patch_section + "\n".join(registry_overrides) + "\n", 1)
    (stage / "Cargo.toml").write_text(manifest)
    (stage / "src").symlink_to(BRIDGE / "src", target_is_directory=True)


def check_graph(cargo, target, stage, update_lock):
    command = [cargo, "tree", "--target", target, "--features", "talon,fips", "-e", "normal,build",
               "--prefix", "none", "--format", "{p}", "--manifest-path", stage / "Cargo.toml"]
    if not update_lock:
        command.append("--locked")
    graph = output(command)
    modules = {tuple(line.split()[:2]) for line in graph.splitlines() if line}
    disallowed = sorted({name for name, _ in modules if name in DISALLOWED_BACKENDS})
    if disallowed:
        parents = [output([*command, "--invert", name, "--depth", "3"]) for name in disallowed]
        raise ValueError("unexpected Rust TLS crypto dependencies:\n" + "\n".join(parents))
    for name, version in VERSIONS.items():
        if {actual for module, actual in modules if module == name} != {f"v{version}"}:
            raise ValueError(f"FIPS Cargo graph requires {name} {version}")
    (stage / "cargo-tree.txt").write_text(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--target", required=True, choices=("aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--update-lock", action="store_true")
    args = parser.parse_args()

    stage = args.output.absolute()
    lock = FIPS_DIR / "Cargo.lock"
    patches = sorted((FIPS_DIR / "patches").glob("*.patch"))
    lock_input = lock if lock.exists() or not args.update_lock else BRIDGE / "Cargo.lock"
    inputs = [Path(__file__), BRIDGE / "Cargo.toml", BRIDGE / "Cargo.lock", lock_input, *patches]
    digest = hashlib.sha256(args.target.encode())
    for path in inputs:
        digest.update(path.read_bytes())
    stamp = stage / ".inputs"
    if stage.is_symlink() or (stage.exists() and (stamp.is_symlink() or not stamp.is_file())):
        parser.error(f"refusing to replace an unmanaged build directory: {stage}")
    if not args.update_lock and stamp.exists() and stamp.read_text() == digest.hexdigest():
        return

    stage.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fips-cargo-", dir=stage.parent) as temporary:
        prepared = Path(temporary) / "workspace"
        prepared.mkdir()
        stage_dependencies(args.cargo, args.target, prepared, patches)
        shutil.copyfile(lock_input, prepared / "Cargo.lock")
        check_graph(args.cargo, args.target, prepared, args.update_lock)
        if args.update_lock:
            shutil.copyfile(prepared / "Cargo.lock", lock)
        (prepared / ".inputs").write_text(digest.hexdigest())
        if stage.exists():
            shutil.rmtree(stage)
        prepared.rename(stage)


if __name__ == "__main__":
    main()
