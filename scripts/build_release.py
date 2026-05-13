from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RELEASE_NAME_PREFIX = "interactive_decision_tree_business"
INCLUDE_FILES = [
    "interactive_decision_tree_app.py",
    "requirements.txt",
    "requirements-notebook.txt",
    "pyproject.toml",
    "README.md",
    "PROJECT_CONTEXT.md",
    "Start Interactive Tree.bat",
    "Open Notebook.bat",
    "start_interactive_tree.sh",
    "open_notebook.sh",
]
INCLUDE_DIRS = [
    "interactive_decision_tree",
    "examples",
    "scripts",
]
EXCLUDED_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".streamlit",
    ".tree_checkpoints",
    ".tree_sessions",
    "__pycache__",
    ".ipynb_checkpoints",
    "ora_config",
    "oracle_config",
    "dist",
    "wheelhouse",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def current_platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported build platform: {sys.platform}")


def requirements_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("requirements.txt", "requirements-notebook.txt"):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return False


def copy_tree(src: Path, dst: Path) -> None:
    if should_skip(src):
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            copy_tree(child, dst / child.name)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_staging(root: Path, target_platform: str) -> Path:
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)
    staging = dist_dir / f"{RELEASE_NAME_PREFIX}_{target_platform}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for file_name in INCLUDE_FILES:
        source = root / file_name
        if source.exists():
            shutil.copy2(source, staging / file_name)

    for dir_name in INCLUDE_DIRS:
        copy_tree(root / dir_name, staging / dir_name)

    (staging / "wheelhouse" / "windows").mkdir(parents=True, exist_ok=True)
    (staging / "wheelhouse" / "linux").mkdir(parents=True, exist_ok=True)
    return staging


def build_wheelhouse(root: Path, staging: Path, target_platform: str) -> None:
    if target_platform != current_platform():
        raise RuntimeError(
            f"This script builds {current_platform()} wheelhouses only. "
            f"Run the matching build script on {target_platform}."
        )

    wheelhouse = staging / "wheelhouse" / target_platform
    if wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    wheelhouse.mkdir(parents=True)

    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "-r",
        str(root / "requirements.txt"),
        "-r",
        str(root / "requirements-notebook.txt"),
        "-d",
        str(wheelhouse),
    ]
    subprocess.check_call(command, cwd=str(root))

    metadata = {
        "platform": target_platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "machine": platform.machine(),
        "requirements_hash": requirements_hash(root),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (wheelhouse / "WHEELHOUSE_METADATA.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def write_release_manifest(root: Path, staging: Path, target_platform: str) -> None:
    manifest = {
        "name": staging.name,
        "platform": target_platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "excluded": sorted(EXCLUDED_NAMES),
        "notes": [
            "oracle_config and .streamlit secrets are intentionally excluded.",
            "Offline install requires a wheelhouse built for the same OS and Python minor version.",
        ],
    }
    (staging / "BUSINESS_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def make_archives(root: Path, staging: Path, archive: str) -> list[Path]:
    dist_dir = root / "dist"
    created: list[Path] = []
    if archive in ("zip", "both"):
        archive_path = shutil.make_archive(
            str(dist_dir / staging.name),
            "zip",
            root_dir=str(dist_dir),
            base_dir=staging.name,
        )
        created.append(Path(archive_path))
    if archive in ("gztar", "both"):
        archive_path = shutil.make_archive(
            str(dist_dir / staging.name),
            "gztar",
            root_dir=str(dist_dir),
            base_dir=staging.name,
        )
        created.append(Path(archive_path))
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a business release package.")
    parser.add_argument("--platform", choices=["windows", "linux"], required=True)
    parser.add_argument("--archive", choices=["zip", "gztar", "both"], default="zip")
    parser.add_argument(
        "--skip-wheelhouse",
        action="store_true",
        help="Create the release layout without downloading offline wheels.",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")
    args = parse_args()
    root = project_root()
    staging = prepare_staging(root, args.platform)
    if not args.skip_wheelhouse:
        build_wheelhouse(root, staging, args.platform)
    write_release_manifest(root, staging, args.platform)
    archives = make_archives(root, staging, args.archive)
    print(f"Release staging folder: {staging}")
    for archive in archives:
        print(f"Release archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
