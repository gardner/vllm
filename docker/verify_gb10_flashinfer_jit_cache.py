import re
import subprocess
import sys
from pathlib import Path

import flashinfer_jit_cache


ALLOWED_CUDA_IMAGES = {"sm_121a", "compute_121a"}


def main() -> int:
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
