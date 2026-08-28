"""Run mypy and expose captured diagnostics as a GitHub Actions annotation."""

from __future__ import annotations

import subprocess
import sys


def _github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--no-incremental",
        "--explicit-package-bases",
        "src/edge_opt",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        diagnostics = (result.stdout + result.stderr).strip() or "mypy produced no output"
        print(f"::error title=mypy failed::{_github_escape(diagnostics)}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
