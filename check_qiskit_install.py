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

MIN_PYTHON = (3, 10)

# package name in pip -> minimum required version
MIN_VERSIONS = {
    "qiskit": "2.5.2",
    "qiskit-ibm-runtime": "0.49",
    "numpy": "2.0",
    "scipy": "1.14",
    "rustworkx": "0.15.0",
}

# distributions that make up `pip install qiskit[visualization]`
VISUALIZATION_EXTRAS = ["matplotlib", "pylatexenc", "Pillow", "pydot", "seaborn", "sympy"]


def parse_version(version: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", version)) or (0,)


def version_at_least(installed: str, minimum: str) -> bool:
    return parse_version(installed) >= parse_version(minimum)


def installed_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def check_python() -> tuple[bool, str]:
    version = ".".join(str(v) for v in sys.version_info[:3])
    ok = sys.version_info[:2] >= MIN_PYTHON
    required = ".".join(str(v) for v in MIN_PYTHON)
    return ok, f"Python {version} (>= {required} required)"


def check_min_version(dist_name: str, minimum: str) -> tuple[bool, str]:
    version = installed_version(dist_name)
    if version is None:
        return False, f"{dist_name} is not installed (>= {minimum} required)"
    ok = version_at_least(version, minimum)
    return ok, f"{dist_name} {version} (>= {minimum} required)"


def check_visualization_extras(verbose: bool) -> tuple[bool, str]:
    missing = [name for name in VISUALIZATION_EXTRAS if installed_version(name) is None]
    if missing:
        return False, f"qiskit[visualization] extras missing: {', '.join(missing)}"
    detail = ""
    if verbose:
        found = ", ".join(f"{n} {installed_version(n)}" for n in VISUALIZATION_EXTRAS)
        detail = f" ({found})"
    return True, f"qiskit[visualization] extras installed{detail}"


def check_aer(verbose: bool) -> tuple[bool, str]:
    version = installed_version("qiskit-aer")
    if version is None:
        return False, "qiskit-aer is not installed"
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        backend = AerSimulator()
        result = backend.run(transpile(circuit, backend), shots=100).result()
        counts = result.get_counts()
    except Exception as exc:  # noqa: BLE001 - surface any simulator failure to the user
        return False, f"qiskit-aer {version} is installed but a test simulation failed: {exc}"
    detail = f" (test circuit ran, counts={counts})" if verbose else " (test simulation passed)"
    return True, f"qiskit-aer {version}{detail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that this environment has everything needed for the course."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print extra detail for each check."
    )
    args = parser.parse_args()

    print("Checking Python environment for Qiskit coursework...\n")

    checks = [
        check_python(),
        check_min_version("qiskit", MIN_VERSIONS["qiskit"]),
        check_visualization_extras(args.verbose),
        check_min_version("qiskit-ibm-runtime", MIN_VERSIONS["qiskit-ibm-runtime"]),
        check_aer(args.verbose),
        check_min_version("numpy", MIN_VERSIONS["numpy"]),
        check_min_version("scipy", MIN_VERSIONS["scipy"]),
        check_min_version("rustworkx", MIN_VERSIONS["rustworkx"]),
    ]

    failures = []
    for ok, message in checks:
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {message}")
        if not ok:
            failures.append(message)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        print(
            'Try: pip install -U "qiskit[visualization]>={}" '
            '"qiskit-ibm-runtime>={}" qiskit-aer'.format(
                MIN_VERSIONS["qiskit"], MIN_VERSIONS["qiskit-ibm-runtime"]
            )
        )
        return 1

    print("All checks passed. Your environment is ready for class.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
