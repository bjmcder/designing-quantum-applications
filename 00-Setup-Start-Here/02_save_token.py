#!/usr/bin/env python3
"""Save an IBM Quantum API token to local Qiskit Runtime credentials.

Run with no arguments to see usage and examples. After saving, run
03_check_token.py to confirm the token works and see which instance
is selected.
"""

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

logging.basicConfig(
    level=logging.WARN,
    format="[%(levelname)-8s] %(name)s: %(message)s",
)
# Suppress harmless Qiskit warnings about instance discovery
logging.getLogger("qiskit_runtime_service").setLevel(logging.ERROR)

DEFAULT_CHANNEL = "ibm_quantum_platform"

# Keys IBM Cloud has used for the API key string in a downloaded credentials
# JSON file (e.g. apikey.json), checked in order.
TOKEN_KEYS_IN_FILE = ["apikey", "api_key", "token", "key"]


def load_token_from_file(path: str) -> str:
    """Load an IBM Quantum API token from a credentials JSON file.

    Parameters
    ----------
    path : str
        Path to the credentials JSON file (e.g., apikey.json).

    Returns
    -------
    str
        The API token string extracted from the file.

    Raises
    ------
    SystemExit
        If the file is not found, is invalid JSON, or does not contain an API key.
    """
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


def list_available_instances(token: str, channel: str) -> list[dict]:
    """Fetch the instances accessible with the given token.

    Parameters
    ----------
    token : str
        The IBM Quantum API token.
    channel : str
        The account channel (e.g., "ibm_quantum_platform" or "ibm_cloud").

    Returns
    -------
    list[dict]
        Instance dictionaries as returned by QiskitRuntimeService.instances(),
        each with (at least) "name", "crn", and "plan" keys.

    Raises
    ------
    SystemExit
        If the instances can't be listed with the given token.
    """
    try:
        probe = QiskitRuntimeService(channel=channel, token=token, instance="auto")
        return probe.instances()
    except Exception as exc:  # noqa: BLE001 - report any connection failure to the user
        sys.exit(f"error: couldn't verify instances with this token: {exc}")


def resolve_instance_crn(token: str, channel: str, instance: str) -> str:
    """Resolve an instance name to its CRN.

    QiskitRuntimeService.save_account()'s docstring says "instance accepts
    either a CRN or a display name, but the service only ever validates a
    *CRN* against your accessible instances when it's next instantiated."

    This means that a name saved as-is silently fails to take effect (or later
    raises a confusing "account does not have access" error). Resolve names
    here so the `--instance` argument works with either.

    Parameters
    ----------
    token : str
        The IBM Quantum API token.
    channel : str
        The account channel (e.g., "ibm_quantum_platform" or "ibm_cloud").
    instance : str
        The instance identifier; either a CRN (starting with "crn:") or a display name.

    Returns
    -------
    str
        The Cloud Resource Name (CRN) of the instance.

    Raises
    ------
    SystemExit
        If the instance name cannot be resolved or is not accessible with the given token.
    """
    if instance.startswith("crn:"):
        return instance

    available = list_available_instances(token, channel)

    matches = [inst for inst in available if inst.get("name") == instance]
    if not matches:
        names = ", ".join(inst.get("name") for inst in available) or "(none found)"
        sys.exit(
            f"error: no instance named '{instance}' is accessible with this "
            f"token. Available instances: {names}"
        )

    match = matches[0]
    if match.get("plan") != "on-prem":
        print(
            f"warning: instance '{instance}' has plan '{match.get('plan')}', "
            "not 'on-prem' -- this course uses an on-prem instance."
        )
    print(f"Resolved instance name '{instance}' to CRN: {match['crn']}")
    return match["crn"]


def auto_select_onprem_instance(token: str, channel: str) -> str | None:
    """Automatically pick the on-prem instance used for this course.

    Called when the user doesn't pass --instance explicitly. Without an
    instance set, IBM prioritizes free/trial plan instances over the
    on-prem instance dedicated to this course, so look it up instead of
    leaving the default to chance.

    Parameters
    ----------
    token : str
        The IBM Quantum API token.
    channel : str
        The account channel (e.g., "ibm_quantum_platform" or "ibm_cloud").

    Returns
    -------
    str or None
        The CRN of the on-prem instance, or None if none (or more than one)
        is accessible with this token -- a warning is printed in that case
        and the caller should fall back to leaving no default instance set.
    """
    available = list_available_instances(token, channel)
    onprem = [inst for inst in available if inst.get("plan") == "on-prem"]

    if not onprem:
        print(
            "warning: no on-prem instance is accessible with this token; "
            "leaving no default instance set. Once you know its name, set "
            "one explicitly with --instance."
        )
        return None

    if len(onprem) > 1:
        names = ", ".join(inst.get("name") for inst in onprem)
        print(
            f"warning: multiple on-prem instances are accessible ({names}); "
            "leaving no default instance set. Pick one explicitly with --instance."
        )
        return None

    inst = onprem[0]
    print(f"Auto-selected on-prem instance '{inst.get('name')}' for this course.")
    return inst["crn"]


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser for token saving.
    """
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
        "(a name is looked up and resolved to its CRN automatically). If "
        "omitted, the on-prem instance for this course is auto-selected.",
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
    """Save an IBM Quantum API token to local Qiskit Runtime credentials.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
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
    else:
        instance = auto_select_onprem_instance(token, args.channel)

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
