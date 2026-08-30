#!/usr/bin/env python3
"""Verify that a Python environment has everything needed for this course.

Checks Python version, Qiskit (with visualization extras), Qiskit IBM
Runtime, Qiskit Aer, and the upstream numerical dependencies Qiskit relies
on. Run with no arguments; pass -v/--verbose for extra detail.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata

MIN_PYTHON = (3, 12)

# package name in pip -> minimum required version
MIN_VERSIONS = {
    "qiskit": "2.5.2",
    "qiskit-ibm-runtime": "0.49",
    "numpy": "2.0",
    "scipy": "1.14",
    "rustworkx": "0.15.0",
}

# distributions that make up `pip install qiskit[visualization]`
VISUALIZATION_EXTRAS = [
    "matplotlib",
    "pylatexenc",
    "Pillow",
    "pydot",
    "seaborn",
    "sympy",
]


def parse_version(version: str) -> tuple:
    """Extract version numbers from a version string as a tuple of integers.

    Parameters
    ----------
    version : str
        Version string (e.g., "1.2.3" or "2.0.0rc1").

    Returns
    -------
    tuple
        Tuple of integers extracted from the version string.
    """
    return tuple(int(x) for x in re.findall(r"\d+", version)) or (0,)


def version_at_least(installed: str, minimum: str) -> bool:
    """Check if installed version meets the minimum requirement.

    Parameters
    ----------
    installed : str
        The installed version string.
    minimum : str
        The minimum required version string.

    Returns
    -------
    bool
        True if installed version >= minimum version, False otherwise.
    """
    return parse_version(installed) >= parse_version(minimum)


def installed_version(dist_name: str) -> str | None:
    """Return the installed version of a package, or None if not installed.

    Parameters
    ----------
    dist_name : str
        The distribution/package name.

    Returns
    -------
    str or None
        The installed version string, or None if the package is not found.
    """
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def check_python() -> tuple[bool, str]:
    """Verify Python version meets minimum requirement.

    Returns
    -------
    tuple[bool, str]
        A tuple of (passed: bool, message: str).
    """
    version = ".".join(str(v) for v in sys.version_info[:3])
    ok = sys.version_info[:2] >= MIN_PYTHON
    required = ".".join(str(v) for v in MIN_PYTHON)
    return ok, f"Python {version} (>= {required} required)"


def check_min_version(dist_name: str, minimum: str) -> tuple[bool, str]:
    """Check if a distribution is installed with the required minimum version.

    Parameters
    ----------
    dist_name : str
        The distribution/package name.
    minimum : str
        The minimum required version string.

    Returns
    -------
    tuple[bool, str]
        A tuple of (passed: bool, message: str).
    """
    version = installed_version(dist_name)
    if version is None:
        return False, f"{dist_name} is not installed (>= {minimum} required)"
    ok = version_at_least(version, minimum)
    return ok, f"{dist_name} {version} (>= {minimum} required)"


def check_visualization_extras(verbose: bool) -> tuple[bool, str]:
    """Verify all qiskit visualization extra packages are installed.

    Parameters
    ----------
    verbose : bool
        If True, include detailed version information in the message.

    Returns
    -------
    tuple[bool, str]
        A tuple of (passed: bool, message: str).
    """
    missing = [
        name for name in VISUALIZATION_EXTRAS if installed_version(name) is None
    ]
    if missing:
        missing_str = ", ".join(missing)
        return False, f"qiskit[visualization] extras missing: {missing_str}"
    detail = ""
    if verbose:
        found = ", ".join(
            f"{n} {installed_version(n)}" for n in VISUALIZATION_EXTRAS
        )
        detail = f" ({found})"
    return True, f"qiskit[visualization] extras installed{detail}"


def check_aer(verbose: bool) -> tuple[bool, str]:
    """Verify qiskit-aer is installed and a test circuit can be simulated.

    Parameters
    ----------
    verbose : bool
        If True, include test circuit results in the message.

    Returns
    -------
    tuple[bool, str]
        A tuple of (passed: bool, message: str).
    """
    version = installed_version("qiskit-aer")
    if version is None:
        return False, "qiskit-aer is not installed"
    try:

        # A small inline program that import Aer and runs a test circuit
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        backend = AerSimulator()
        transpiled = transpile(circuit, backend)
        result = backend.run(transpiled, shots=100).result()
        counts = result.get_counts()
    except Exception as exc:  # noqa: BLE001 - report failure to user
        msg = (
            f"qiskit-aer {version} is installed but a test "
            f"simulation failed: {exc}"
        )
        return False, msg
    if verbose:
        detail = f" (test circuit ran, counts={counts})"
    else:
        detail = " (test simulation passed)"
    return True, f"qiskit-aer {version}{detail}"


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check that this environment has everything needed for the course."
        )
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print extra detail for each check.",
    )
    return parser


def main() -> int:
    """Run all environment checks and report results.

    Returns
    -------
    int
        Exit code: 0 if all checks passed, 1 if any checks failed.
    """
    parser = build_parser()
    args = parser.parse_args()

    print("Checking Python environment ...\n")

    # Collect all the checks to be performed
    checks = [
        check_python(),
        check_min_version("qiskit", MIN_VERSIONS["qiskit"]),
        check_visualization_extras(args.verbose),
        check_min_version(
            "qiskit-ibm-runtime", MIN_VERSIONS["qiskit-ibm-runtime"]
        ),
        check_aer(args.verbose),
        check_min_version("numpy", MIN_VERSIONS["numpy"]),
        check_min_version("scipy", MIN_VERSIONS["scipy"]),
        check_min_version("rustworkx", MIN_VERSIONS["rustworkx"]),
    ]

    # Run the checks and report success/failure
    failures = []
    for ok, message in checks:
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {message}")
        if not ok:
            failures.append(message)

    print() # Newline

    # Report the failures and suggested fixes
    if failures:
        print(f"{len(failures)} check(s) failed.")
        print(
            "If you're using uv (the recommended setup): re-run `uv sync` "
            "from the repo root, then try this script again."
        )
        qiskit_ver = MIN_VERSIONS["qiskit"]
        runtime_ver = MIN_VERSIONS["qiskit-ibm-runtime"]
        print(
            "If you installed with plain pip/conda instead: try:\n"
            f'  pip install -U "qiskit[visualization]>={qiskit_ver}" '
            f'"qiskit-ibm-runtime>={runtime_ver}" qiskit-aer'
        )
        return 1

    print("All checks passed. Your environment is ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
