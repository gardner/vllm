import importlib
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

ALLOWED_CUDA_IMAGES = {"sm_121a", "compute_121a"}
REQUIRED_FLASHINFER_DISTRIBUTIONS = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)


def _is_gb10_cuda13_version(version: object) -> bool:
    return isinstance(version, str) and "+cu13" in version and "gb10" in version


def collect_distribution_versions(
    distribution_names: tuple[str, ...] = REQUIRED_FLASHINFER_DISTRIBUTIONS,
) -> dict[str, str | None]:
    versions = {}
    for distribution_name in distribution_names:
        try:
            versions[distribution_name] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution_name] = None
    return versions


def validate_distribution_versions(
    distribution_versions: dict[str, str | None] | None = None,
) -> list[str]:
    if distribution_versions is None:
        distribution_versions = collect_distribution_versions()

    errors = []
    for distribution_name in REQUIRED_FLASHINFER_DISTRIBUTIONS:
        version = distribution_versions.get(distribution_name)
        if not _is_gb10_cuda13_version(version):
            errors.append(
                f"{distribution_name} must be a GB10 CUDA 13 build "
                f"with '+cu13' and 'gb10' in the version, got {version!r}."
            )
    return errors


def main() -> int:
    distribution_versions = collect_distribution_versions()
    distribution_errors = validate_distribution_versions(distribution_versions)
    if distribution_errors:
        print("GB10 FlashInfer runtime package version check failed:")
        for error in distribution_errors:
            print(error)
        return 1
    print(f"GB10 FlashInfer runtime package versions: {distribution_versions}")

    flashinfer_jit_cache = importlib.import_module("flashinfer_jit_cache")
    jit_cache_dir = Path(flashinfer_jit_cache.__file__).parent / "jit_cache"
    if not jit_cache_dir.is_dir():
        print(f"FlashInfer JIT cache directory not found: {jit_cache_dir}")
        return 1

    found_cuda = set()
    bad_files = []
    so_files = sorted(jit_cache_dir.rglob("*.so"))
    for so_file in so_files:
        proc = subprocess.run(
            ["cuobjdump", "--list-elf", str(so_file)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        tokens = set(re.findall(r"(?:sm|compute)_[0-9]+[a-z]?", proc.stdout))
        if not tokens:
            continue
        found_cuda.update(tokens)
        unexpected = sorted(tokens - ALLOWED_CUDA_IMAGES)
        if unexpected:
            bad_files.append((so_file, unexpected))

    if bad_files:
        print("GB10 FlashInfer JIT cache contains unexpected CUDA images:")
        for so_file, unexpected in bad_files:
            print(f"{so_file}: {unexpected}")
        return 1

    if not found_cuda:
        print(f"GB10 FlashInfer JIT cache has no CUDA images: {jit_cache_dir}")
        return 1

    print(
        "GB10 FlashInfer JIT cache CUDA images: "
        f"{sorted(found_cuda)} across {len(so_files)} shared objects"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
