# DocMind — Windows Setup Guide

## Step 1 · Fix PowerShell's execution policy (one-time)

By default, PowerShell blocks `.ps1` scripts — including the venv activator.
Run this **once** in PowerShell (as your normal user, not Administrator):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 2 · Create the virtual environment

Open a terminal in the project root (`C:\Users\Keshav Kapur\Downloads\Projects\rag-pdf-chatbot`):

```powershell
python -m venv venv
```

> If `python` isn't found, try `py -m venv venv` or install Python from python.org
> and tick "Add Python to PATH" during installation.

## Step 3 · Activate the venv

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt (cmd.exe):**
```cmd
venv\Scripts\activate.bat
```

You'll see `(venv)` appear at the start of your prompt when it's active.

> ⚠️ Common mistake: `source venv/bin/activate` is the Linux/macOS command.
> On Windows, always use the paths above.

## Step 4 · Install dependencies

```powershell
pip install -r requirements.txt
```

## Step 5 · Set up your .env file

```powershell
copy .env.example .env
```

Then open `.env` and fill in your actual API keys:
- `ANTHROPIC_API_KEY` — your Claude API key
- `OPENAI_API_KEY` — your OpenAI key (used only for embeddings)

## Step 6 · Run the tests

```powershell
pytest tests/ -v
```

All 8 tests should pass. ✓

## Step 7 · Launch the app

```powershell
streamlit run app/ui.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Use `py` instead, or reinstall Python with "Add to PATH" ticked |
| `Activate.ps1 cannot be loaded` | Run Step 1 (execution policy) |
| `ModuleNotFoundError` | Make sure venv is active `(venv)` before running pip install |
| `chromadb` import errors | `pip install chromadb --upgrade` inside the active venv |
| Tests fail with `ANTHROPIC_API_KEY` error | Tests use mock keys — this shouldn't happen after the fix; ensure you're running from the project root |
