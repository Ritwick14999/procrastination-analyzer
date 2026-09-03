# Running this project locally in VS Code

Everything below assumes you have [VS Code](https://code.visualstudio.com/) and
Python 3.10+ installed. Total setup time is about five minutes.

---

## 1. Get the code onto your machine

The work lives on the `improvement` branch.

```bash
git clone https://github.com/ritwick14999/procrastination-analyzer.git
cd procrastination-analyzer
git checkout improvement
code .
```

If you already have the repo cloned locally, just pull the branch:

```bash
git fetch origin improvement
git checkout improvement
git pull origin improvement
code .
```

> `code .` opens the current folder in VS Code. If that command isn't found on
> macOS, open VS Code and run **Shell Command: Install 'code' command in PATH**
> from the Command Palette (<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>).

When the folder opens, VS Code will offer to install the recommended extensions
(Python, Pylance, Ruff, Mypy) — accept. They are listed in `.vscode/extensions.json`.

---

## 2. Create a virtual environment

In the VS Code terminal (<kbd>Ctrl</kbd>+<kbd>`</kbd>):

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

Then install the project with its development and app dependencies:

```bash
pip install -e ".[dev,app]"
```

The `-e` flag installs in editable mode, so your edits take effect immediately
without reinstalling.

**Point VS Code at the environment:** press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>
(<kbd>Cmd</kbd> on macOS), run **Python: Select Interpreter**, and choose the one
inside `.venv`. This is the step people most often skip, and skipping it is why
imports show as unresolved.

---

## 3. Check it works

```bash
pytest                 # 135 tests, should all pass
make check             # lint + typecheck + tests, the full CI gate
```

On Windows without `make`, run the steps directly:

```powershell
ruff check src tests
mypy
pytest
```

---

## 4. Run the app

### From the terminal

```bash
streamlit run src/procrastination_analyzer/ui/app.py
```

Opens at <http://localhost:8501>. Pick **Sample data** or **Simulated persona** in
the sidebar to see it working immediately — no data of your own required.

### From the debugger (better for development)

Open the **Run and Debug** panel (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd>).
Four configurations ship in `.vscode/launch.json`:

| Configuration | What it does |
|---|---|
| **Streamlit: dashboard** | Launches the app with the debugger attached, so breakpoints work inside the analysis code |
| **CLI: analyze sample data** | Steps through a full analysis run |
| **CLI: evaluate models** | Steps through cross-validation |
| **Pytest: current file** | Debugs the test file you have open |

Set a breakpoint by clicking left of a line number, then press <kbd>F5</kbd>.

---

## 5. Run the analysis on your own data

Any CSV with a timestamp column works — the column is auto-detected from common
names (`timestamp`, `ts`, `date`, `created_at`, ...).

```bash
procrastination-analyzer analyze mydata.csv
procrastination-analyzer analyze mydata.csv --format markdown -o report.md
```

To analyse your own git history:

```bash
git log --pretty=format:"%ad" --date=iso > commits.csv
# add a header line so the column is named
sed -i '1i timestamp' commits.csv          # macOS: sed -i '' '1i\'$'\n''timestamp' commits.csv
procrastination-analyzer analyze commits.csv
```

Scoring a *live* record rather than a closed historical one? Pass the current time,
so the recency features are measured against now rather than against your last event:

```bash
procrastination-analyzer analyze commits.csv --now "$(date -Iseconds)"
```

---

## 6. Testing inside VS Code

Test discovery is preconfigured in `.vscode/settings.json`. Open the **Testing**
panel (flask icon) and tests appear automatically — run individually, debug them,
or run the whole suite. If nothing appears, re-check the interpreter selection in
step 2, then run **Test: Refresh Tests** from the Command Palette.

---

## 7. Optional: use the devcontainer

If you have Docker and the **Dev Containers** extension, skip the environment setup
entirely: open the Command Palette and run **Dev Containers: Reopen in Container**.
The container builds Python 3.12, installs everything, and forwards port 8501 for
Streamlit.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: procrastination_analyzer` | The venv isn't active or the package isn't installed. Re-run `pip install -e ".[dev,app]"`. |
| Imports underlined in the editor but code runs fine | Wrong interpreter selected. **Python: Select Interpreter** → the one in `.venv`. |
| `procrastination-analyzer: command not found` | The venv isn't active. Or use `python -m procrastination_analyzer.cli` instead. |
| Tests don't appear in the Testing panel | **Test: Refresh Tests** from the Command Palette; check the Python output panel for discovery errors. |
| Streamlit port already in use | `streamlit run ... --server.port 8502` |
| `Activate.ps1 cannot be loaded` (Windows) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell session. |
