#!/usr/bin/env bash
# One-stop setup: installs uv (if needed) and syncs this repo's Python
# environment. Safe to re-run any time.
#
# Usage (from anywhere):
#   bash 00-Setup-Start-Here/bootstrap.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "== Designing Quantum Applications: environment bootstrap =="

if ! command -v uv >/dev/null 2>&1; then
    echo "-> uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo
        echo "uv was installed but isn't on PATH in this shell session."
        echo "Open a new terminal (or re-source your shell profile) and re-run:"
        echo "  bash 00-Setup-Start-Here/bootstrap.sh"
        exit 1
    fi
else
    echo "-> uv already installed ($(uv --version))"
fi

echo "-> Setting up the environment (uv sync)..."
cd "$REPO_ROOT"
uv sync

echo "-> Verifying the install..."
uv run 00-Setup-Start-Here/01_check_install.py

cat <<EOF

== Bootstrap complete ==

Next steps:
  1. Save your IBM Quantum API token:
       uv run 00-Setup-Start-Here/02_save_token.py
  2. Verify the token and check your instance:
       uv run 00-Setup-Start-Here/03_check_token.py
  3. Run a real circuit:
       uv run 00-Setup-Start-Here/04_test_quantum.py

See README.md for full details.
EOF
