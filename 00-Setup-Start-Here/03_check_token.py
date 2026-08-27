#!/usr/bin/env python3
"""Check a saved IBM Quantum account and list accessible instances.

Run this after 02_save_token.py to confirm the saved token actually
works, and to see every instance you have access to (useful when you have
more than one, e.g. a personal instance plus one dedicated to a class).
"""

import argparse
import sys

from qiskit_ibm_runtime import QiskitRuntimeService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a saved IBM Quantum account and list accessible instances.",
    )
    parser.add_argument(
        "--name",
        default="default",
        help="Name of the saved account to check (default: 'default').",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    saved = QiskitRuntimeService.saved_accounts(name=args.name)
    if not saved:
        print(f"No saved account named '{args.name}' was found.")
        print("Run 02_save_token.py first.")
        return 1

    default_instance = saved[args.name].get("instance")

    print(f"Connecting with saved account '{args.name}'...")
    try:
        service = QiskitRuntimeService(name=args.name)
        instances = service.instances()
    except Exception as exc:  # noqa: BLE001 - report any connection failure to the user
        print(f"Connection failed: {exc}")
        print(
            "Your saved token may be invalid or expired. Get a new one and "
            "re-run 02_save_token.py."
        )
        return 1

    print(f"Connected. Found {len(instances)} accessible instance(s):")
    for inst in instances:
        is_selected = default_instance in (inst.get("name"), inst.get("crn"))
        marker = "  <-- default for this account" if default_instance and is_selected else ""
        print(f"  - {inst.get('name')}  [plan: {inst.get('plan')}]{marker}")

    if not default_instance:
        print(
            "\nNo default instance is set for this account, so IBM picks "
            "one automatically when you submit a job. To set one explicitly "
            "(e.g. the instance dedicated to this class), run:\n"
            f"  uv run 02_save_token.py --name {args.name} --instance "
            '"<name from the list above>" --overwrite'
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
