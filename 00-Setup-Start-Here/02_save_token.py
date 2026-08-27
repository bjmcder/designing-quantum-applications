#!/usr/bin/env python3
"""Save an IBM Quantum API token to local Qiskit Runtime credentials.

Run with no arguments to see usage and examples. After saving, run
03_check_token.py to confirm the token works and see which instance
is selected.
"""

import argparse
import getpass
import json
import sys
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

DEFAULT_CHANNEL = "ibm_quantum_platform"

# Keys IBM Cloud has used for the API key string in a downloaded credentials
# JSON file (e.g. apikey.json), checked in order.
TOKEN_KEYS_IN_FILE = ["apikey", "api_key", "token", "key"]


def load_token_from_file(path: str) -> str:
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text())
    except FileNotFoundError:
        sys.exit(f"error: no such file: {file_path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {file_path} is not valid JSON: {exc}")

    for key in TOKEN_KEYS_IN_FILE:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    sys.exit(
        f"error: couldn't find an API key in {file_path} "
        f"(looked for keys: {', '.join(TOKEN_KEYS_IN_FILE)})"
    )


def resolve_instance_crn(token: str, channel: str, instance: str) -> str:
    """Resolve an instance name to its CRN.

    QiskitRuntimeService.save_account()'s docstring says --instance accepts
    either a CRN or a display name, but the service only ever validates a
    *CRN* against your accessible instances when it's next instantiated --
    a name saved as-is silently fails to take effect (or later raises a
    confusing "account does not have access" error). Resolve names here so
    --instance works with either, the way it's documented to.
    """
    if instance.startswith("crn:"):
        return instance

    try:
        probe = QiskitRuntimeService(channel=channel, token=token)
        available = probe.instances()
    except Exception as exc:  # noqa: BLE001 - report any connection failure to the user
        sys.exit(f"error: couldn't verify instances with this token: {exc}")

    matches = [inst for inst in available if inst.get("name") == instance]
    if not matches:
        names = ", ".join(inst.get("name") for inst in available) or "(none found)"
        sys.exit(
            f"error: no instance named '{instance}' is accessible with this "
            f"token. Available instances: {names}"
        )

    crn = matches[0]["crn"]
    print(f"Resolved instance name '{instance}' to CRN: {crn}")
    return crn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="02_save_token.py",
        description=(
            "Save your IBM Quantum API token locally so qiskit-ibm-runtime "
            "can find it automatically (QiskitRuntimeService())."
        ),
        epilog=(
            "Examples:\n"
            "  uv run 02_save_token.py\n"
            "      (prompts for your token interactively)\n\n"
            "  uv run 02_save_token.py --token YOUR_API_KEY\n\n"
            "  uv run 02_save_token.py --from-file ~/Downloads/apikey.json\n\n"
            "  uv run 02_save_token.py --token YOUR_API_KEY "
            "--instance CRN_OR_INSTANCE_NAME --name my-instance\n\n"
            "Get an API key from https://quantum.cloud.ibm.com/\n"
            "After saving, run 03_check_token.py to verify it and see "
            "which instance is selected."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    token_source = parser.add_mutually_exclusive_group()
    token_source.add_argument(
        "--token",
        help="IBM Quantum API token. If omitted (and --from-file is not "
        "used), you will be prompted (input is hidden).",
    )
    token_source.add_argument(
        "--from-file",
        metavar="PATH",
        help="Path to a credentials JSON file downloaded from the IBM "
        "Quantum/Cloud dashboard (e.g. apikey.json) to read the token from.",
    )
    parser.add_argument(
        "--channel",
        choices=["ibm_quantum_platform", "ibm_cloud"],
        default=DEFAULT_CHANNEL,
        help=f"Account channel (default: {DEFAULT_CHANNEL}).",
    )
    parser.add_argument(
        "--instance",
        help="CRN or display name of the instance to use as the default "
        "(a name is looked up and resolved to its CRN automatically).",
    )
    parser.add_argument(
        "--name",
        default="default",
        help="Name to save this account under (default: 'default'). Use "
        "this if you need to store more than one account.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing saved account with the same name.",
    )
    parser.add_argument(
        "--set-as-default",
        action="store_true",
        help="If you have multiple named accounts (--name), make this one "
        "the default used by QiskitRuntimeService() when no name is given. "
        "Has nothing to do with which instance is used -- see --instance "
        "for that.",
    )
    return parser


def main() -> int:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.from_file:
        token = load_token_from_file(args.from_file)
    else:
        token = args.token or getpass.getpass("Enter your IBM Quantum API token: ")
    if not token:
        parser.error("a token is required (pass --token or enter it at the prompt)")

    instance = args.instance
    if instance:
        instance = resolve_instance_crn(token, args.channel, instance)

    QiskitRuntimeService.save_account(
        token=token,
        channel=args.channel,
        instance=instance,
        name=args.name,
        overwrite=args.overwrite,
        set_as_default=args.set_as_default or None,
    )
    print(f"Account saved as '{args.name}'.")
    print(f"Run 03_check_token.py --name {args.name} to verify it and see "
          "which instance is selected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
