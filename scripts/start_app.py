from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser
from pathlib import Path


APP_IMPORTS = [
    "streamlit",
    "pandas",
    "numpy",
    "sqlalchemy",
    "openpyxl",
    "xlrd",
    "streamlit_agraph",
]
NOTEBOOK_IMPORTS = ["ipykernel", "notebook"]
KERNEL_NAME = "interactive_decision_tree_env"
KERNEL_DISPLAY_NAME = "interactive_decision_tree_env (.venv)"
SETUP_STATE_FILE = ".setup_state.json"
RELEASE_MANIFEST_FILE = "BUSINESS_RELEASE_MANIFEST.json"
SESSION_DIR_ENV = "INTERACTIVE_TREE_SESSION_DIR"
SESSION_DIR_NAME = ".tree_sessions"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_session_dir(root: Path, override: str | None = None) -> Path:
    configured = override or os.environ.get(SESSION_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (root / SESSION_DIR_NAME).resolve()


def configure_session_dir(root: Path, override: str | None = None) -> Path:
    session_dir = resolve_session_dir(root, override)
    os.environ[SESSION_DIR_ENV] = str(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def platform_key() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported platform for business launcher: {sys.platform}")


def venv_dir(root: Path) -> Path:
    return root / ".venv"


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return venv_dir(root) / "Scripts" / "python.exe"
    return venv_dir(root) / "bin" / "python"


def running_inside_project_venv(root: Path) -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python(root).resolve()
    except OSError:
        return False


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")


def requirement_files(root: Path, profile: str) -> list[Path]:
    files = [root / "requirements.txt"]
    if profile == "notebook":
        files.append(root / "requirements-notebook.txt")
    return files


def requirements_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_setup_state(root: Path) -> dict[str, object]:
    path = root / SETUP_STATE_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def is_business_release(root: Path) -> bool:
    return (root / RELEASE_MANIFEST_FILE).exists()


def save_setup_state(root: Path, state: dict[str, object]) -> None:
    (root / SETUP_STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_venv(root: Path) -> None:
    python_path = venv_python(root)
    if python_path.exists():
        return
    print(f"Creating local virtual environment: {venv_dir(root)}")
    venv.EnvBuilder(with_pip=True).create(venv_dir(root))


def wheelhouse_dir(root: Path) -> Path:
    return root / "wheelhouse" / platform_key()


def validate_wheelhouse(root: Path) -> Path:
    wheelhouse = wheelhouse_dir(root)
    if not wheelhouse.exists() or not any(wheelhouse.iterdir()):
        raise RuntimeError(
            f"Offline wheelhouse not found: {wheelhouse}\n"
            "Use the matching business release package or rebuild it on this platform."
        )

    metadata_path = wheelhouse / "WHEELHOUSE_METADATA.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        built_python = str(metadata.get("python_version", ""))
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if built_python and built_python != current_python:
            raise RuntimeError(
                f"Wheelhouse was built for Python {built_python}, "
                f"but current Python is {current_python}. Use a matching Python version "
                "or rebuild the release package."
            )
        built_platform = str(metadata.get("platform", ""))
        if built_platform and built_platform != platform_key():
            raise RuntimeError(
                f"Wheelhouse was built for {built_platform}, but current platform is {platform_key()}."
            )
    return wheelhouse


def imports_available(python_path: Path, imports: list[str]) -> bool:
    code = "import " + ",".join(imports)
    result = subprocess.run(
        [str(python_path), "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(project_root()),
    )
    return result.returncode == 0


def project_package_installed(python_path: Path) -> bool:
    result = subprocess.run(
        [str(python_path), "-m", "pip", "show", "interactive-decision-tree"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(project_root()),
    )
    return result.returncode == 0


def install_project_editable(root: Path, python_path: Path) -> None:
    subprocess.check_call(
        [str(python_path), "-m", "pip", "install", "--no-deps", "-e", str(root)],
        cwd=str(root),
    )


def install_dependencies(root: Path, profile: str, allow_online: bool = False) -> None:
    python_path = venv_python(root)
    req_files = requirement_files(root, profile)
    req_hash = requirements_hash(req_files)
    state = load_setup_state(root)
    profile_state = state.get(profile)
    expected_imports = APP_IMPORTS + (NOTEBOOK_IMPORTS if profile == "notebook" else [])

    if (
        isinstance(profile_state, dict)
        and profile_state.get("requirements_hash") == req_hash
        and imports_available(python_path, expected_imports)
    ):
        if not project_package_installed(python_path):
            install_project_editable(root, python_path)
        print(f"Dependencies already installed for profile: {profile}")
        return

    command = [str(python_path), "-m", "pip", "install"]
    if allow_online:
        print("Installing dependencies from online package index.")
    else:
        try:
            wheelhouse = validate_wheelhouse(root)
        except RuntimeError:
            if is_business_release(root):
                raise
            print(
                "Offline wheelhouse was not found in this source checkout; "
                "installing dependencies from the online package index."
            )
            print("Business release packages still require a platform wheelhouse.")
        else:
            print(f"Installing dependencies from offline wheelhouse: {wheelhouse}")
            command.extend(["--no-index", "--find-links", str(wheelhouse)])
    for req_file in req_files:
        command.extend(["-r", str(req_file)])

    subprocess.check_call(command, cwd=str(root))
    install_project_editable(root, python_path)
    state[profile] = {
        "requirements_hash": req_hash,
        "platform": platform_key(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "installed_at": int(time.time()),
    }
    save_setup_state(root, state)


def server_ready(host: str, port: int) -> bool:
    url = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def local_ip_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addresses.add(item[4][0])
    except OSError:
        pass
    return sorted(addr for addr in addresses if not addr.startswith("127."))


def run_streamlit(root: Path, args: argparse.Namespace) -> int:
    session_dir = configure_session_dir(root, args.session_dir)
    host = "127.0.0.1" if args.mode == "local" else "0.0.0.0"
    browser_host = "localhost" if args.mode == "local" else "127.0.0.1"
    url = f"http://{browser_host}:{args.port}"

    if server_ready(browser_host, args.port):
        print(f"Interactive Decision Tree is already running: {url}")
        if args.mode == "local" and args.open_browser:
            webbrowser.open(url)
        return 0

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "interactive_decision_tree_app.py"),
        "--server.address",
        host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]

    env = os.environ.copy()
    env[SESSION_DIR_ENV] = str(session_dir)
    print("Starting Interactive Decision Tree...")
    print(f"Session directory: {session_dir}")
    process = subprocess.Popen(command, cwd=str(root), env=env)

    deadline = time.time() + 30
    while time.time() < deadline:
        if server_ready(browser_host, args.port):
            break
        if process.poll() is not None:
            return process.returncode or 1
        time.sleep(0.5)

    if args.mode == "local":
        print(f"Open this URL: {url}")
        if args.open_browser:
            webbrowser.open(url)
    else:
        print(f"Server is listening on: http://0.0.0.0:{args.port}")
        for address in local_ip_addresses():
            print(f"Network URL: http://{address}:{args.port}")
        print("Use server mode only on a trusted internal network.")

    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def register_notebook_kernel(root: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            KERNEL_DISPLAY_NAME,
        ],
        cwd=str(root),
    )
    print(f"Notebook kernel registered: {KERNEL_NAME}")


def vscode_command_candidates() -> list[str]:
    command_names = ["code.cmd", "code.exe", "code"] if os.name == "nt" else ["code"]
    candidates: list[str] = []
    for command_name in command_names:
        command_path = shutil.which(command_name)
        if command_path and command_path not in candidates:
            candidates.append(command_path)
    return candidates


def try_open_vscode(root: Path, notebook_path: Path) -> bool:
    for code_exe in vscode_command_candidates():
        try:
            if os.name == "nt" and Path(code_exe).suffix.lower() in {".cmd", ".bat"}:
                subprocess.Popen(["cmd", "/c", code_exe, str(root), str(notebook_path)], cwd=str(root))
            else:
                subprocess.Popen([code_exe, str(root), str(notebook_path)], cwd=str(root))
        except OSError as exc:
            print(f"VS Code command failed ({code_exe}): {exc}")
            continue
        print(f"Opened notebook in VS Code: {notebook_path}")
        return True
    return False


def open_notebook(root: Path, args: argparse.Namespace) -> int:
    session_dir = configure_session_dir(root, args.session_dir)
    print(f"Session directory: {session_dir}")
    register_notebook_kernel(root)
    notebook_path = root / "examples" / "notebook_dataframe_sql_demo.ipynb"
    if try_open_vscode(root, notebook_path):
        return 0

    print("VS Code command `code` was not found. Starting Jupyter Notebook instead.")
    command = [
        sys.executable,
        "-m",
        "notebook",
        str(notebook_path),
        "--notebook-dir",
        str(root),
    ]
    return subprocess.call(command, cwd=str(root))


def bootstrap_and_rerun(root: Path, profile: str, argv: list[str], allow_online: bool) -> int:
    ensure_python_version()
    ensure_venv(root)
    install_dependencies(root, profile, allow_online=allow_online)
    python_path = venv_python(root)
    return subprocess.call([str(python_path), str(Path(__file__).resolve()), *argv], cwd=str(root))


def build_parser(argv: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Decision Tree business launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    app_parser = subparsers.add_parser("app", help="Start the Streamlit app")
    app_parser.add_argument("--mode", choices=["local", "server"], default="local")
    app_parser.add_argument("--port", type=int, default=8501)
    app_parser.add_argument("--open-browser", dest="open_browser", action="store_true", default=True)
    app_parser.add_argument("--no-open-browser", dest="open_browser", action="store_false")
    app_parser.add_argument("--session-dir", help="Shared folder for notebook/UI DataFrame sessions")
    app_parser.add_argument("--allow-online", action="store_true", help="Allow online pip install if no wheelhouse is present")

    notebook_parser = subparsers.add_parser("notebook", help="Open the notebook example")
    notebook_parser.add_argument("--session-dir", help="Shared folder for notebook/UI DataFrame sessions")
    notebook_parser.add_argument("--allow-online", action="store_true", help="Allow online pip install if no wheelhouse is present")

    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in {"app", "notebook"}:
        return ["app", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    raw_argv = normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser(raw_argv)
    args = parser.parse_args(raw_argv)
    root = project_root()
    profile = "notebook" if args.command == "notebook" else "app"
    configure_session_dir(root, args.session_dir)

    if not running_inside_project_venv(root):
        return bootstrap_and_rerun(root, profile, raw_argv, allow_online=bool(args.allow_online))

    ensure_python_version()
    if args.command == "notebook":
        return open_notebook(root, args)
    return run_streamlit(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
