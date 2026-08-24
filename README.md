# Designing Quantum Applications

MANE 4960, Fall 2026

## Setup

You need a working Python environment with Qiskit before the first class. Pick
whichever path fits you — they all end up with the same package versions.

### Option A: Local Installation With `uv`

This repo uses [`uv`](https://docs.astral.sh/uv/) to pin an identical set of
package versions that give a consistent experience on Windows, macOS, Linux,
and Windows Subsystem for Linux (WSL).

1. **Install `uv`** (one-time, skip if you already have it):
   - macOS / Linux / WSL:
     ```
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - Windows (PowerShell):
     ```
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - Or, if you already have Python: `pip install uv`

2. **Clone this repo and set up the environment**, from the repo root:
   ```
   uv sync
   ```
   This downloads the exact Python version and package versions pinned in
   `pyproject.toml`/`uv.lock` (Qiskit, Qiskit IBM Runtime, Qiskit Aer,
   JupyterLab, etc.) into a local `.venv/` — no manual `pip install` needed,
   and nothing installed system-wide.

3. **Verify the install:**
   ```
   uv run check_qiskit_install.py
   ```
   All checks should print `[OK]`. If something fails, re-run `uv sync` and
   try again before asking for help.

4. **Save your IBM Quantum API token:**
   - Follow the instructions at [RPI Quantum Computing](https://foci.rpi.edu/computing-resources/rpi-quantum-computing) to set up your access to the IBM quantum platform.
   - Log in to your IBM Quantum dashboard and create an API key ([instructions here](https://quantum.cloud.ibm.com/docs/en/guides/save-credentials)).
   - Either **copy** the key, or use the dashboard's **Download** option, which
     saves a credentials file named `apikey.json`.

   Then save it locally with whichever matches what you did above:
   ```
   # If you copied the key, paste it when prompted
   uv run save_qiskit_token.py

   # If you downloaded apikey.json, point the script at it directly
   uv run save_qiskit_token.py --from-file ~/Downloads/apikey.json
   ```
   The script saves your credentials to `~/.qiskit/qiskit-ibm.json` and verifies the connection.
   `apikey.json` itself is never copied into this repo, and is git-ignored if you download it here by mistake. Never share your API key with anyone or allow it to be committed to a version control repository (e.g. Git).

5. **Run notebooks:**
   ```
   uv run jupyter lab
   ```

Any time you see `uv run <script>.py`, it will run the script inside this project's
pinned environment (you don't need to activate anything manually). If you'd rather
activate the environment the normal way, `uv sync` also creates a standard `.venv/`
that you can activate by running `source .venv/bin/activate` in bash (macOS/Linux/WSL)
or `.venv\Scripts\activate` in Powershell (Windows).

### Option B: Preconfigured Cloud Environment

 - If you'd rather not install anything locally, we've set up a preconfigured cloud environment with all the required packages pre-installed. Details on accessing this platform will be provided in class.

### Option C: bring your own environment

If you already have a Python setup you like (conda, plain `venv`, etc.) and
would rather not use `uv`, please install from the exported, pinned dependency list
instead:
```
pip install -r requirements.txt
```
`requirements.txt` is generated from the same lockfile as the `uv` path, so
you'll get the same tested versions. Whatever environment you use, run
`python check_qiskit_install.py` (or `python3`, depending on your setup) to
confirm everything is installed correctly before class.

## Scripts

- [`check_qiskit_install.py`](check_qiskit_install.py) — verifies your
  environment has compatible versions of Qiskit, Qiskit IBM Runtime, and
  Qiskit Aer, and runs a real test circuit to confirm the simulator works.
- [`save_qiskit_token.py`](save_qiskit_token.py) — saves your IBM Quantum
  API token locally so `QiskitRuntimeService()` can find it automatically.
