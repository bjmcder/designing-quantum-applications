# Designing Quantum Applications

Course Code: MANE-4960
Semester: Fall 2026
Instructor: Brian McDermott

## Getting This Repo (New to Git/GitHub?)

The setup steps below assume you have a local copy of this repo. If you've
never used Git or GitHub before, here's what that means and how to do it.

**Option 1: Download a ZIP (simplest, no Git required)**

1. On the GitHub page for this repo, click the green **Code** button, then
   **Download ZIP**.
2. Unzip it somewhere on your computer (e.g. your Documents folder).
3. Open a terminal (macOS/Linux) or PowerShell (Windows) and `cd` into the
   unzipped folder before continuing to the [Setup](#setup) steps below.

**Note:** You won't be able to easily pull updates as this repo changes
during the semester, you will need to re-download the latest version.

**Option 2: Clone with Git (recommended, lets you pull updates)**

1. **Install Git**, if you don't already have it:
   - macOS: install [Xcode Command Line Tools](https://developer.apple.com/xcode/resources/)
     (`xcode-select --install`) or install via [Homebrew](https://brew.sh/):
     `brew install git`.
   - Linux: `sudo apt install git` (Debian/Ubuntu) or your distro's
     equivalent.
   - Windows: install [Git for Windows](https://git-scm.com/download/win),
     which also gives you "Git Bash," a terminal you can use for the rest of
     these instructions.
   - WSL: same as Linux, inside your WSL distro.
2. **Clone the repo.** Open a terminal, navigate to wherever you want the
   folder to live (e.g. `cd ~/Documents`), then run:
   ```
   git clone <repo-url>
   ```
   (Use the URL from the green **Code** button on the GitHub page — copy
   the HTTPS link.) This creates a folder with a full copy of the repo,
   including its history.
3. **Move into the folder** before continuing to the [Setup](#setup) steps
   below:
   ```
   cd F2026
   ```
4. **Pulling updates later:** if changes are pushed to the repo during the
   semester, get them by running this from inside the folder:
   ```
   git pull
   ```

You won't be expected to know anything else about Git for this class. However,
it's an extremely powerful tool for tracking and working with complex projects. If you're
curious to learn more, GitHub's own
[Git and GitHub basics guide](https://docs.github.com/en/get-started/quickstart)
is a good starting point.

## Setup

You need a working Python environment with Qiskit. Pick whichever of the following
options you feel most comfortable with:

### Option A: Local Installation With `uv`

This repo uses [`uv`](https://docs.astral.sh/uv/) to pin an identical set of
package versions that give a consistent experience on Windows, macOS, Linux,
and Windows Subsystem for Linux (WSL).

**Shortcut:** steps 1–3 below (install `uv`, sync the environment, verify
the install) are automated by a bootstrap script. From the repo root:
```
# macOS / Linux / WSL
bash 00-Setup-Start-Here/bootstrap.sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -File 00-Setup-Start-Here\bootstrap.ps1
```
It's safe to re-run any time. If you'd rather run each step yourself (or
want to understand what it's doing), follow steps 1–3 manually below.

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
   uv run 00-Setup-Start-Here/01_check_install.py
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
   uv run 00-Setup-Start-Here/02_save_token.py

   # If you downloaded apikey.json, point the script at it directly
   uv run 00-Setup-Start-Here/02_save_token.py --from-file ~/Downloads/apikey.json
   ```
   The script saves your credentials to `~/.qiskit/qiskit-ibm.json`.
   `apikey.json` itself is never copied into this repo, and is git-ignored if you download it here by mistake. Never share your API key with anyone or allow it to be committed to a version control repository (e.g. Git).

5. **Verify the token and check your instance:**
   ```
   uv run 00-Setup-Start-Here/03_check_token.py
   ```
   This connects with the credentials you just saved and lists every
   instance you have access to.

   **If you have access to more than one instance** (e.g. a personal
   instance plus the one dedicated to this class), this list is how you
   find its name. Note the name of the class instance, then re-run
   `02_save_token.py` pointing at it so it becomes your default:
   ```
   uv run 00-Setup-Start-Here/02_save_token.py --instance "<class instance name>" --overwrite
   ```
   (`--instance` accepts either a CRN or a display name — a name is looked
   up and resolved automatically. This is unrelated to `--set-as-default`,
   which instead picks a default *account* when you've saved more than one
   under different `--name`s.)
   Run `uv run 00-Setup-Start-Here/03_check_token.py` again afterward to
   confirm it's now marked as the default.

6. **Run a real circuit:**
   ```
   uv run 00-Setup-Start-Here/04_test_quantum.py
   ```
   Builds a Bell state and runs it first on a local, noiseless Aer
   simulator, then on the class's IBM hardware backend, 10,000 shots each.
   The hardware queue can be long — if the script loses its connection
   (or you close your laptop) while waiting, the job keeps running on
   IBM's servers. Fetch the result later instead of resubmitting:
   ```
   uv run 00-Setup-Start-Here/05_retrieve_job.py
   ```

7. **Run notebooks:**
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
`python 00-Setup-Start-Here/01_check_install.py` (or `python3`, depending
on your setup) to confirm everything is installed correctly before class.

## Scripts

All in [`00-Setup-Start-Here/`](00-Setup-Start-Here/), meant to be run in order the first time:

- [`bootstrap.sh`](00-Setup-Start-Here/bootstrap.sh) /
  [`bootstrap.ps1`](00-Setup-Start-Here/bootstrap.ps1) — one-stop setup
  script for Option A: installs `uv` if it's missing, runs `uv sync`, and
  runs `01_check_install.py` to confirm everything worked.
- [`01_check_install.py`](00-Setup-Start-Here/01_check_install.py) —
  verifies your environment has compatible versions of Qiskit, Qiskit IBM
  Runtime, and Qiskit Aer, and runs a real test circuit to confirm the
  simulator works.
- [`02_save_token.py`](00-Setup-Start-Here/02_save_token.py) — saves your
  IBM Quantum API token locally so `QiskitRuntimeService()` can find it
  automatically.
- [`03_check_token.py`](00-Setup-Start-Here/03_check_token.py) — connects
  with a saved token to confirm it works, and lists every instance you
  have access to (and which one is your default).
- [`04_test_quantum.py`](00-Setup-Start-Here/04_test_quantum.py) — builds
  a Bell state and runs it on both the Aer simulator and real IBM
  hardware, showing named registers, named circuits, PUBs, samplers, job
  submission/retrieval, and post-processing. Saves the job ID locally as
  soon as it's submitted so a lost connection doesn't lose the job.
- [`05_retrieve_job.py`](00-Setup-Start-Here/05_retrieve_job.py) — fetches
  the result of a previously submitted hardware job (by ID, or the most
  recent one by default) without resubmitting it.

## License

Source code in this repository is licensed under the [MIT License](LICENSE).
All other course materials (slides, notes, problem sets, etc.) are licensed
under [CC BY 4.0](LICENSE-CONTENT).
