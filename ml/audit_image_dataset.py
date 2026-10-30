"""Audit an image-classification ZIP or folder and safely extract ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from ml.image_labels import IMAGE_CLASS_LABELS

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_IMAGE_INDEX = re.compile(r"^.+_(\d+)$")
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024**3


def _safe_archive_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(name)
    if (
        not normalized
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return path


def _archive_layout(archive: zipfile.ZipFile) -> tuple[str, dict[str, Any]]:
    entries = archive.infolist()
    total_uncompressed_bytes = sum(entry.file_size for entry in entries)
    if total_uncompressed_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError("ZIP archive exceeds the 8 GiB uncompressed-size limit")

    top_level: set[str] = set()
    class_counts: Counter[str] = Counter()
    ignored_checkpoint_images = 0
    image_entries = 0
    metadata_entries = 0
    for entry in entries:
        member = _safe_archive_member(entry.filename)
        if entry.is_dir():
            continue
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"ZIP archive contains a symbolic link: {entry.filename!r}")
        top_level.add(member.parts[0])
        suffix = Path(member.name).suffix.lower()
        if suffix in SUPPORTED_SUFFIXES:
            if any(part.startswith(".") for part in member.parts):
                ignored_checkpoint_images += 1
                continue
            image_entries += 1
            if len(member.parts) == 3:
                class_counts[member.parts[1]] += 1
        else:
            metadata_entries += 1

    if len(top_level) != 1:
        raise ValueError("ZIP archive must contain exactly one top-level dataset folder")
    root_name = next(iter(top_level))
    return root_name, {
        "entry_count": len(entries),
        "uncompressed_bytes": total_uncompressed_bytes,
        "image_entry_count_excluding_hidden": image_entries,
        "ignored_hidden_image_count": ignored_checkpoint_images,
        "non_image_entry_count": metadata_entries,
        "class_counts": dict(sorted(class_counts.items())),
    }


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extract one dataset root without traversal or overwriting existing files."""
    zip_path = zip_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not zip_path.is_file():
        raise ValueError(f"ZIP archive does not exist: {zip_path}")
    if destination.exists():
        raise FileExistsError(
            f"extraction destination already exists; choose a new path: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        root_name, _ = _archive_layout(archive)
        with tempfile.TemporaryDirectory(
            prefix=".signlearn-extract-", dir=destination.parent
        ) as temporary_directory:
            staging_root = Path(temporary_directory) / "dataset"
            staging_root.mkdir()
            for entry in archive.infolist():
                member = _safe_archive_member(entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(
                        f"ZIP archive contains a symbolic link: {entry.filename!r}"
                    )
                if member.parts[0] != root_name:
                    raise ValueError("ZIP archive has an inconsistent dataset root")
                relative_parts = member.parts[1:]
                if not relative_parts or any(part.startswith(".") for part in relative_parts):
                    continue
                target = staging_root.joinpath(*relative_parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            try:
                os.replace(staging_root, destination)
            except OSError as exc:
                raise FileExistsError(
                    f"could not create extraction destination: {destination}"
                ) from exc
    return destination


def _difference_hash(image_path: Path) -> int:
    with Image.open(image_path) as image:
        pixels = list(
            image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata()
        )
    result = 0
    for y in range(8):
        row = pixels[y * 9 : (y + 1) * 9]
        for x in range(8):
            result = (result << 1) | int(row[x] > row[x + 1])
    return result


def audit_image_dataset(dataset_root: Path) -> dict[str, Any]:
    """Report label coverage, decode failures, exact duplicates, and near matches.

    Near-duplicate candidates are conservatively compared only within the same
    class and within five consecutive filename indices. They require review and
    are not silently treated as duplicate labels.
    """
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {dataset_root}")

    class_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    classes = [path.name for path in class_dirs]
    unknown = sorted(set(classes).difference(IMAGE_CLASS_LABELS))
    missing = [label for label in IMAGE_CLASS_LABELS if label not in classes]
    class_counts: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    candidates: list[tuple[int, str, int, Path]] = []
    corrupt: list[dict[str, str]] = []
    unsupported_files: list[str] = []
    total_bytes = 0

    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        for path in images:
            relative = path.relative_to(root).as_posix()
            total_bytes += path.stat().st_size
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image_format = image.format or "unknown"
                    size_key = f"{image.width}x{image.height}"
                    image.convert("RGB")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                exact_hashes[digest].append(relative)
                class_counts[class_dir.name] += 1
                image_sizes[size_key] += 1
                formats[image_format] += 1
                match = _IMAGE_INDEX.fullmatch(path.stem)
                if match is not None:
                    candidates.append(
                        (int(match.group(1)), class_dir.name, 0, path)
                    )
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                corrupt.append({"path": relative, "reason": type(exc).__name__})

        for path in class_dir.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() not in SUPPORTED_SUFFIXES
            ):
                unsupported_files.append(path.relative_to(root).as_posix())

    near_duplicate_pairs: list[dict[str, str | int]] = []
    grouped_candidates: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for index, class_name, _, path in candidates:
        grouped_candidates[class_name].append((index, path))
    for class_name, records in grouped_candidates.items():
        records.sort(key=lambda item: (item[0], item[1].name))
        hashes = [(index, path, _difference_hash(path)) for index, path in records]
        for position, (left_index, left_path, left_hash) in enumerate(hashes):
            for right_index, right_path, right_hash in hashes[position + 1 :]:
                if right_index - left_index > 5:
                    break
                distance = (left_hash ^ right_hash).bit_count()
                if distance <= 2:
                    near_duplicate_pairs.append(
                        {
                            "first": left_path.relative_to(root).as_posix(),
                            "second": right_path.relative_to(root).as_posix(),
                            "hash_distance": distance,
                        }
                    )

    duplicate_groups = [
        sorted(paths) for paths in exact_hashes.values() if len(paths) > 1
    ]
    return {
        "dataset_root": str(root),
        "expected_labels": list(IMAGE_CLASS_LABELS),
        "classes_found": classes,
        "missing_labels": missing,
        "unexpected_labels": unknown,
        "valid_image_count": sum(class_counts.values()),
        "class_counts": {
            label: class_counts.get(label, 0) for label in IMAGE_CLASS_LABELS
        },
        "corrupt_image_count": len(corrupt),
        "corrupt_images": corrupt,
        "unsupported_file_count": len(unsupported_files),
        "unsupported_files": unsupported_files,
        "file_bytes": total_bytes,
        "image_dimensions": dict(sorted(image_sizes.items())),
        "image_formats": dict(sorted(formats.items())),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_groups": duplicate_groups,
        "near_duplicate_candidate_pair_count": len(near_duplicate_pairs),
        "near_duplicate_candidates": near_duplicate_pairs[:500],
        "near_duplicate_candidate_note": (
            "dHash candidates within the same class and filename-index distance "
            "of at most five; inspect manually, and do not treat as proof of "
            "duplicate content or independent source sessions."
        ),
        "source_or_license_metadata_found": False,
        "dataset_version": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = audit_image_dataset(args.dataset)
    if args.zip:
        with zipfile.ZipFile(args.zip) as archive:
            _, report["archive"] = _archive_layout(archive)
    if args.extract_to:
        if not args.zip:
            parser.error("--extract-to requires --zip")
        report["extracted_to"] = str(safe_extract_zip(args.zip, args.extract_to))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
