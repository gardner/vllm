#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bundle GB10 release smoke reports into a durable artifact.

This script is report-only. It does not start containers, query a vLLM server,
import torch, or touch CUDA. It packages the JSON reports produced by the GB10
offline/image/OpenAI smoke helpers, writes checksums and provenance metadata,
and creates a tarball that can be uploaded to GitHub Actions, GitHub Releases,
or another artifact store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_REPORTS = (
    "gb10-nvfp4-smoke.json",
    "gb10-openai-server-smoke-image.json",
    "gb10-release-evidence-image.json",
)

EXPECTED_EVIDENCE_FILES = (
    *EXPECTED_REPORTS,
    "gb10-smoked-image-digest.txt",
)

PROVENANCE_FILES = {
    "release_manifest": "gb10-release-manifest.json",
    "runtime_image_metadata": "buildx-runtime-image-metadata.json",
}

PROVENANCE_RELATIVE_PATHS = {
    "release_manifest": "provenance/gb10-release-manifest.json",
    "runtime_image_metadata": "provenance/buildx-runtime-image-metadata.json",
}
SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _default_report_dir() -> Path:
    return Path(
        os.environ.get("GB10_RELEASE_EVIDENCE_REPORT_DIR", "gb10-smoke-reports")
    )


def _default_output_dir() -> Path:
    return Path(
        os.environ.get(
            "GB10_RELEASE_EVIDENCE_OUTPUT_DIR",
            "dist/gb10-release-evidence",
        )
    )


def _default_release_manifest_json() -> Path | None:
    explicit = os.environ.get("GB10_RELEASE_EVIDENCE_MANIFEST_JSON") or os.environ.get(
        "GB10_RELEASE_MANIFEST_JSON"
    )
    if explicit:
        return Path(explicit)
    manifest_dir = os.environ.get("GB10_RELEASE_MANIFEST_DIR")
    if manifest_dir:
        return Path(manifest_dir) / PROVENANCE_FILES["release_manifest"]
    return None


def _default_runtime_image_metadata_json() -> Path | None:
    explicit = os.environ.get(
        "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON"
    ) or os.environ.get("GB10_RUNTIME_IMAGE_METADATA_JSON")
    if explicit:
        return Path(explicit)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bundle GB10 release smoke JSON reports with checksums and "
            "provenance metadata."
        )
    )
    parser.add_argument(
        "--gb10-report-dir",
        type=Path,
        default=_default_report_dir(),
        help=(
            "Directory containing GB10 smoke reports "
            "(default: GB10_RELEASE_EVIDENCE_REPORT_DIR or ./gb10-smoke-reports)."
        ),
    )
    parser.add_argument(
        "--gb10-output-dir",
        type=Path,
        default=_default_output_dir(),
        help=(
            "Directory where the bundle, metadata, and checksums are written "
            "(default: GB10_RELEASE_EVIDENCE_OUTPUT_DIR or "
            "./dist/gb10-release-evidence)."
        ),
    )
    parser.add_argument(
        "--gb10-bundle-name",
        default=os.environ.get(
            "GB10_RELEASE_EVIDENCE_BUNDLE_NAME",
            "gb10-release-evidence",
        ),
        help="Base name for the generated .tar.gz bundle.",
    )
    parser.add_argument(
        "--gb10-include-glob",
        action="append",
        default=None,
        help=(
            "Report-dir glob to include. May be repeated. "
            "Defaults to gb10-*.json and gb10-*.txt."
        ),
    )
    parser.add_argument(
        "--gb10-image-ref",
        default=os.environ.get("GB10_RELEASE_EVIDENCE_IMAGE_REF"),
        help="Candidate runtime image ref recorded in metadata.",
    )
    parser.add_argument(
        "--gb10-release-tag",
        default=os.environ.get("GB10_RELEASE_EVIDENCE_RELEASE_TAG"),
        help="Release tag recorded in metadata.",
    )
    parser.add_argument(
        "--gb10-commit",
        default=os.environ.get("GB10_RELEASE_EVIDENCE_COMMIT"),
        help="Source commit recorded in metadata. Defaults to git HEAD if available.",
    )
    parser.add_argument(
        "--gb10-release-manifest-json",
        type=Path,
        default=_default_release_manifest_json(),
        help=(
            "Optional gb10-release-manifest.json from the release workflow. "
            "Defaults to GB10_RELEASE_EVIDENCE_MANIFEST_JSON, "
            "GB10_RELEASE_MANIFEST_JSON, or "
            "GB10_RELEASE_MANIFEST_DIR/gb10-release-manifest.json when set."
        ),
    )
    parser.add_argument(
        "--gb10-runtime-image-metadata-json",
        type=Path,
        default=_default_runtime_image_metadata_json(),
        help=(
            "Optional BuildKit runtime-image metadata JSON. Defaults to "
            "GB10_RELEASE_EVIDENCE_RUNTIME_IMAGE_METADATA_JSON or "
            "GB10_RUNTIME_IMAGE_METADATA_JSON when set."
        ),
    )
    parser.add_argument(
        "--gb10-allow-partial",
        action="store_true",
        help=(
            "Bundle available GB10 reports even if one of the final-image "
            "release reports is missing. The metadata status will be partial."
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    value = proc.stdout.strip()
    if proc.returncode != 0 or not value:
        return None
    return value


def _copy_reports(
    *,
    report_dir: Path,
    output_dir: Path,
    include_globs: list[str],
) -> list[dict[str, Any]]:
    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    reports_dir.mkdir(parents=True)

    selected: dict[str, Path] = {}
    for pattern in include_globs:
        for path in sorted(report_dir.glob(pattern)):
            if path.is_file():
                selected[path.name] = path

    included = []
    for name, source in sorted(selected.items()):
        destination = reports_dir / name
        shutil.copy2(source, destination)
        included.append(
            {
                "relative_path": destination.relative_to(output_dir).as_posix(),
                "source_path": str(source),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return included


def _copy_optional_provenance(
    *,
    output_dir: Path,
    release_manifest_json: Path | None,
    runtime_image_metadata_json: Path | None,
) -> list[dict[str, Any]]:
    provenance_dir = output_dir / "provenance"
    if provenance_dir.exists():
        shutil.rmtree(provenance_dir)

    sources = [
        ("release_manifest", release_manifest_json),
        ("runtime_image_metadata", runtime_image_metadata_json),
    ]
    included = []
    for kind, source in sources:
        if source is None:
            continue

        source = source.resolve()
        if not source.exists():
            raise RuntimeError(f"GB10 provenance file does not exist: {source}")
        if not source.is_file():
            raise RuntimeError(f"GB10 provenance path is not a file: {source}")

        provenance_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / PROVENANCE_RELATIVE_PATHS[kind]
        shutil.copy2(source, destination)
        included.append(
            {
                "kind": kind,
                "relative_path": destination.relative_to(output_dir).as_posix(),
                "source_path": str(source),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return included


def _write_metadata(
    *,
    output_dir: Path,
    report_dir: Path,
    included_files: list[dict[str, Any]],
    included_provenance: list[dict[str, Any]],
    commit: str | None,
    release_tag: str | None,
    image_ref: str | None,
) -> Path:
    included_by_name = {
        Path(item["relative_path"]).name: item for item in included_files
    }
    expected_evidence_files = []
    for evidence_file in EXPECTED_EVIDENCE_FILES:
        included = included_by_name.get(evidence_file)
        expected_evidence_files.append(
            {
                "name": evidence_file,
                "present": included is not None,
                "relative_path": included["relative_path"] if included else None,
                "sha256": included["sha256"] if included else None,
                "size_bytes": included["size_bytes"] if included else None,
            }
        )

    expected_reports = [
        evidence_file
        for evidence_file in expected_evidence_files
        if evidence_file["name"] in EXPECTED_REPORTS
    ]

    missing_evidence_files = [
        evidence_file["name"]
        for evidence_file in expected_evidence_files
        if not evidence_file["present"]
    ]
    missing_reports = [
        evidence_file
        for evidence_file in missing_evidence_files
        if evidence_file in EXPECTED_REPORTS
    ]

    report_summaries = _summarize_reports(
        output_dir=output_dir,
        included_files=included_files,
    )
    release_gate_summary = report_summaries["gb10-release-evidence-image.json"]
    smoked_image_digest = _summarize_smoked_image_digest(
        output_dir=output_dir,
        included_files=included_files,
    )
    support_matrix_summary = _summarize_support_matrix(
        output_dir=output_dir,
        included_provenance=included_provenance,
    )
    metadata_status = _metadata_status(
        missing_evidence_files=missing_evidence_files,
        release_gate_summary=release_gate_summary,
    )

    metadata = {
        "schema_version": 1,
        "status": metadata_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "report_dir": str(report_dir),
            "commit": commit,
            "release_tag": release_tag,
            "image_ref": image_ref,
            "image_digest": smoked_image_digest["digest"],
        },
        "expected_reports": expected_reports,
        "expected_evidence_files": expected_evidence_files,
        "report_summaries": report_summaries,
        "release_gate_summary": release_gate_summary,
        "release_gate_passed": release_gate_summary.get("release_gate_passed"),
        "smoked_image_digest": smoked_image_digest,
        "support_matrix_summary": support_matrix_summary,
        "missing_reports": missing_reports,
        "missing_evidence_files": missing_evidence_files,
        "included_files": included_files,
        "included_provenance": included_provenance,
    }
    metadata_path = output_dir / "release-evidence-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


def _summarize_reports(
    *,
    output_dir: Path,
    included_files: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    included_by_name = {
        Path(item["relative_path"]).name: output_dir / item["relative_path"]
        for item in included_files
    }
    summaries = {}
    for report_name in EXPECTED_REPORTS:
        report_path = included_by_name.get(report_name)
        report_summary: dict[str, Any] = {"present": report_path is not None}
        if report_path is None:
            summaries[report_name] = report_summary
            continue

        try:
            payload = json.loads(report_path.read_text())
        except json.JSONDecodeError as exc:
            report_summary["parse_error"] = str(exc)
        else:
            if isinstance(payload, dict):
                report_summary["status"] = payload.get("status")
                if report_name == "gb10-release-evidence-image.json":
                    report_summary["release_gate_passed"] = payload.get(
                        "release_gate_passed"
                    )
                    report_summary["failure_count"] = payload.get("failure_count")
            else:
                report_summary["parse_error"] = (
                    f"expected JSON object, got {type(payload).__name__}"
                )
        summaries[report_name] = report_summary
    return summaries


def _empty_support_matrix_summary(
    *,
    release_manifest_present: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    summary = {
        "present": False,
        "release_manifest_present": release_manifest_present,
        "architecture": None,
        "first_release_scope": None,
        "status_definitions": {},
        "entry_count": 0,
        "status_counts": {},
        "entries": {},
        "invalid_entries": [],
    }
    if reason is not None:
        summary["reason"] = reason
    return summary


def _release_manifest_path(
    *,
    output_dir: Path,
    included_provenance: list[dict[str, Any]],
) -> Path | None:
    for item in included_provenance:
        if item.get("kind") == "release_manifest":
            return output_dir / item["relative_path"]
    return None


def _summarize_support_matrix(
    *,
    output_dir: Path,
    included_provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = _release_manifest_path(
        output_dir=output_dir,
        included_provenance=included_provenance,
    )
    if manifest_path is None:
        return _empty_support_matrix_summary(
            release_manifest_present=False,
            reason="release manifest was not included",
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        summary = _empty_support_matrix_summary(
            release_manifest_present=True,
            reason="release manifest is not valid JSON",
        )
        summary["parse_error"] = str(exc)
        return summary

    if not isinstance(manifest, dict):
        return _empty_support_matrix_summary(
            release_manifest_present=True,
            reason=f"expected release manifest object, got {type(manifest).__name__}",
        )

    matrix = manifest.get("gb10_support_matrix")
    if not isinstance(matrix, dict):
        return _empty_support_matrix_summary(
            release_manifest_present=True,
            reason="gb10_support_matrix is missing or is not an object",
        )

    raw_status_definitions = matrix.get("status_definitions")
    status_definitions = (
        dict(sorted(raw_status_definitions.items()))
        if isinstance(raw_status_definitions, dict)
        else {}
    )
    raw_entries = matrix.get("entries")
    if not isinstance(raw_entries, dict):
        summary = _empty_support_matrix_summary(release_manifest_present=True)
        summary.update(
            {
                "present": True,
                "architecture": matrix.get("architecture"),
                "first_release_scope": matrix.get("first_release_scope"),
                "status_definitions": status_definitions,
                "reason": "gb10_support_matrix.entries is missing or is not an object",
            }
        )
        return summary

    entries: dict[str, str] = {}
    invalid_entries: list[str] = []
    status_counts: Counter[str] = Counter()
    for entry_name, entry_payload in sorted(raw_entries.items()):
        if not isinstance(entry_payload, dict):
            invalid_entries.append(str(entry_name))
            continue
        status = entry_payload.get("status")
        if not isinstance(status, str) or not status:
            invalid_entries.append(str(entry_name))
            continue
        entries[str(entry_name)] = status
        status_counts[status] += 1

    return {
        "present": True,
        "release_manifest_present": True,
        "architecture": matrix.get("architecture"),
        "first_release_scope": matrix.get("first_release_scope"),
        "status_definitions": status_definitions,
        "entry_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "entries": entries,
        "invalid_entries": invalid_entries,
    }


def _normalize_image_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if "@" in value:
        value = value.rsplit("@", 1)[1].strip()
    marker = "sha256:"
    marker_index = value.find(marker)
    if marker_index >= 0:
        value = value[marker_index:]
    value = value.split()[0].strip().strip('",')
    if SHA256_DIGEST_RE.fullmatch(value) is None:
        return None
    return value


def _summarize_smoked_image_digest(
    *,
    output_dir: Path,
    included_files: list[dict[str, Any]],
) -> dict[str, Any]:
    digest_path = None
    for item in included_files:
        if Path(item["relative_path"]).name == "gb10-smoked-image-digest.txt":
            digest_path = output_dir / item["relative_path"]
            break

    if digest_path is None:
        return {"present": False, "raw": None, "digest": None}

    raw_digest = digest_path.read_text().strip()
    return {
        "present": True,
        "raw": raw_digest,
        "digest": _normalize_image_digest(raw_digest),
    }


def _metadata_status(
    *,
    missing_evidence_files: list[str],
    release_gate_summary: dict[str, Any],
) -> str:
    if missing_evidence_files:
        return "partial"
    if release_gate_summary.get("release_gate_passed") is True:
        return "complete"
    return "failed"


def _missing_evidence(
    included_files: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    included_names = {Path(item["relative_path"]).name for item in included_files}
    missing_evidence_files = [
        evidence_file
        for evidence_file in EXPECTED_EVIDENCE_FILES
        if evidence_file not in included_names
    ]
    missing_reports = [
        report for report in EXPECTED_REPORTS if report not in included_names
    ]
    return missing_reports, missing_evidence_files


def _write_checksums(output_dir: Path, files: list[Path]) -> Path:
    checksum_path = output_dir / "SHA256SUMS"
    lines = []
    for path in sorted(files, key=lambda item: item.relative_to(output_dir).as_posix()):
        relative_path = path.relative_to(output_dir).as_posix()
        lines.append(f"{_sha256(path)}  {relative_path}")
    checksum_path.write_text("\n".join(lines) + "\n")
    return checksum_path


def _add_tar_entry(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(source), arcname=arcname)
    info.mtime = 0
    with source.open("rb") as stream:
        tar.addfile(info, stream)


def _write_tarball(output_dir: Path, bundle_name: str, files: list[Path]) -> Path:
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(
            files,
            key=lambda item: item.relative_to(output_dir).as_posix(),
        ):
            _add_tar_entry(
                tar,
                path,
                f"{bundle_name}/{path.relative_to(output_dir).as_posix()}",
            )
    (output_dir / f"{archive_path.name}.sha256").write_text(
        f"{_sha256(archive_path)}  {archive_path.name}\n"
    )
    return archive_path


def _bundle(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = args.gb10_report_dir.resolve()
    output_dir = args.gb10_output_dir.resolve()
    include_globs = args.gb10_include_glob or ["gb10-*.json", "gb10-*.txt"]

    if not report_dir.exists():
        raise RuntimeError(f"GB10 report directory does not exist: {report_dir}")
    if not report_dir.is_dir():
        raise RuntimeError(f"GB10 report path is not a directory: {report_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for known_file in (
        "release-evidence-metadata.json",
        "SHA256SUMS",
        f"{args.gb10_bundle_name}.tar.gz",
        f"{args.gb10_bundle_name}.tar.gz.sha256",
    ):
        path = output_dir / known_file
        if path.exists():
            path.unlink()

    included_files = _copy_reports(
        report_dir=report_dir,
        output_dir=output_dir,
        include_globs=include_globs,
    )
    missing_reports, missing_evidence_files = _missing_evidence(included_files)
    if missing_evidence_files and not args.gb10_allow_partial:
        missing = ", ".join(missing_evidence_files)
        raise RuntimeError(
            "Missing required GB10 final-image release evidence files: "
            f"{missing}. Pass --gb10-allow-partial to bundle available evidence."
        )
    if not included_files:
        raise RuntimeError(f"No GB10 evidence files found in {report_dir}")
    included_provenance = _copy_optional_provenance(
        output_dir=output_dir,
        release_manifest_json=args.gb10_release_manifest_json,
        runtime_image_metadata_json=args.gb10_runtime_image_metadata_json,
    )

    repo_root = Path(__file__).resolve().parents[1]
    commit = args.gb10_commit or _git_head(repo_root)
    metadata_path = _write_metadata(
        output_dir=output_dir,
        report_dir=report_dir,
        included_files=included_files,
        included_provenance=included_provenance,
        commit=commit,
        release_tag=args.gb10_release_tag,
        image_ref=args.gb10_image_ref,
    )

    bundle_files = [
        output_dir / item["relative_path"]
        for item in [*included_files, *included_provenance]
    ] + [metadata_path]
    checksum_path = _write_checksums(output_dir, bundle_files)
    archive_path = _write_tarball(
        output_dir,
        args.gb10_bundle_name,
        bundle_files + [checksum_path],
    )
    metadata = json.loads(metadata_path.read_text())

    return {
        "status": metadata["status"],
        "release_gate_passed": metadata.get("release_gate_passed"),
        "smoked_image_digest": metadata.get("smoked_image_digest"),
        "output_dir": str(output_dir),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "metadata_path": str(metadata_path),
        "checksum_path": str(checksum_path),
        "missing_reports": missing_reports,
        "missing_evidence_files": missing_evidence_files,
        "included_count": len(included_files),
        "included_provenance_count": len(included_provenance),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = _bundle(args)
    except RuntimeError as exc:
        print(f"GB10 release evidence bundle failed: {exc}")
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
