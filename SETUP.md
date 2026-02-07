# Setup Guide

Complete end-to-end guide to get the Android AI Agent running on your machine. Follow every step in order — no prior Android development experience required.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Android Studio](#2-install-android-studio)
3. [Create an Android Emulator (AVD)](#3-create-an-android-emulator-avd)
4. [Start the Emulator](#4-start-the-emulator)
5. [Clone the Repository](#5-clone-the-repository)
6. [Create a Virtual Environment & Install Dependencies](#6-create-a-virtual-environment--install-dependencies)
7. [Get a Groq API Key (FREE)](#7-get-a-groq-api-key-free)
8. [Configure the Environment](#8-configure-the-environment)
9. [Run the Agent](#9-run-the-agent)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Install these before proceeding:

| Requirement | Version | Download |
|---|---|---|
| **Python** | 3.11 or newer | https://www.python.org/downloads/ |
| **Git** | any | https://git-scm.com/downloads |
| **Android Studio** | latest | https://developer.android.com/studio |

> **Windows users**: During Python installation, check **"Add python.exe to PATH"**.

Verify installations in a terminal:

```bash
python --version   # should print 3.11+
git --version      # should print any version
```

---

## 2. Install Android Studio

1. Download Android Studio from https://developer.android.com/studio.
2. Run the installer. Accept all defaults.
3. On first launch, the **Setup Wizard** will download the Android SDK, build tools, and platform tools. Let it finish.
4. After the wizard completes, verify ADB is installed:

**Windows** (PowerShell):
```powershell
# ADB is typically located here:
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" version
```

**macOS / Linux**:
```bash
~/Android/Sdk/platform-tools/adb version
# or (macOS): ~/Library/Android/sdk/platform-tools/adb version
```

You should see output like `Android Debug Bridge version 1.0.41`.

### Add ADB to PATH

ADB must be accessible from any terminal window.

**Windows** (PowerShell — run as Administrator):
```powershell
# Add platform-tools to your user PATH permanently
$sdkPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools"
[Environment]::SetEnvironmentVariable("Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";$sdkPath",
    "User")
```
Close and reopen your terminal, then verify:
```powershell
adb version
```

**macOS / Linux** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="$HOME/Android/Sdk/platform-tools:$PATH"
```
Then reload: `source ~/.bashrc` (or `~/.zshrc`). Verify with `adb version`.

---

## 3. Create an Android Emulator (AVD)

1. Open **Android Studio**.
2. Go to **Tools → Device Manager** (or click the phone icon in the toolbar).
3. Click **Create Virtual Device**.
4. Select a device definition:
   - Recommended: **Pixel 8** (any phone with Play Store is fine).
   - Click **Next**.
5. Select a system image:
   - Choose the **latest stable release** (e.g., API 35 / Android 15, or API 34 / Android 14).
   - If it says "Download" next to the image, click the download link and wait.
   - Click **Next**.
6. Name the AVD (e.g., `Pixel_8_API_35`) and click **Finish**.

---

## 4. Start the Emulator

### Option A: From Android Studio

1. Open **Device Manager** (Tools → Device Manager).
2. Click the **▶ Play** button next to your AVD.
3. Wait for the emulator to boot to the home screen (first boot takes 1–3 minutes).

### Option B: From the Command Line

```bash
# List available AVDs
emulator -list-avds

# Start the emulator (replace with your AVD name)
emulator -avd Pixel_8_API_35
```

> The `emulator` command lives in `<SDK>/emulator/`. If not found, add that directory to PATH the same way you added `platform-tools`.

### Verify the emulator is visible to ADB

Open a **new** terminal (separate from the emulator) and run:

```bash
adb devices
```

You should see output like:

```
List of devices attached
emulator-5554   device
```

If the status says `offline` or `unauthorized`, unlock the emulator screen and accept any USB debugging prompts.

---

## 5. Clone the Repository

```bash
git clone https://github.com/varunaditya27/android-ai-agent.git
cd android-ai-agent
```

---

## 6. Create a Virtual Environment & Install Dependencies

**Windows** (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You should see output ending with `Successfully installed ...`. If any package fails, check that you have Python 3.11+ and pip is up to date (`pip install --upgrade pip`).

---

## 7. Get a Groq API Key (FREE)

The agent uses **Groq** (Llama 4 Scout vision model) as its default LLM. The free tier gives you **1,000 requests/day** — more than enough for development.

1. Go to https://console.groq.com/keys.
2. Sign up or log in (Google/GitHub SSO works).
3. Click **Create API Key**.
4. Copy the key (it starts with `gsk_`). You will paste it in the next step.

> **Alternative**: If you prefer Google Gemini, get a key at https://aistudio.google.com/apikey and set `LLM_PROVIDER=gemini` instead. See `.env.example` for Gemini-specific variables.

---

## 8. Configure the Environment

```bash
# Copy the example config
cp .env.example .env
```

Open `.env` in any text editor and set your Groq key:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_paste_your_actual_key_here
```

That's the only required change. All other defaults work out of the box with a local ADB emulator.

> **Do not** commit your `.env` file. It is already in `.gitignore`.

---

## 9. Run the Agent

Make sure:
- Your virtual environment is active (you see `(venv)` in the prompt).
- The Android emulator is running and `adb devices` shows it as `device`.

```bash
python scripts/run_demo.py
```

You should see:

```
╔════════════════════════════════════════════════════════╗
║                Android AI Agent — Demo                 ║
║    AI-powered mobile automation for accessibility      ║
║                                                        ║
║      Groq / Gemini  •  ADB / AWS Device Farm           ║
╚════════════════════════════════════════════════════════╝

📱 Device provider: adb
   ✓ Connected to sdk_gphone64_x86_64
   ✓ Android 16
   ✓ Screen: 1080x2400
🤖 Using Groq model: meta-llama/llama-4-scout-17b-16e-instruct
   Free tier: 1,000 requests/day, 30 req/min
✓ Agent ready!

Interactive Mode
Type your tasks or commands:
  • Type a task description to execute it
  • Type 'exit' or 'quit' to stop
  • Type 'help' for example tasks

🎯 Enter task:
```


### Run with a Specific Task (Non-Interactive)

```bash
python scripts/run_demo.py --task "Open YouTube and search for music" --no-interactive
```

### Run the REST API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

---

## 10. Troubleshooting

### `adb: command not found`

ADB is not in your PATH. See [Add ADB to PATH](#add-adb-to-path) above.

### `adb devices` shows no devices

- Make sure the emulator is running (check the emulator window).
- Try `adb kill-server && adb start-server`.
- If using a physical device: enable **USB Debugging** in Developer Options and accept the prompt on the phone.

### `❌ No Groq API key configured`

You didn't set `GROQ_API_KEY` in your `.env`, or the key is still the placeholder `your-groq-api-key`. Paste your actual key from https://console.groq.com/keys.

### `UnicodeDecodeError` on Windows

This was fixed in the codebase. Make sure you have the latest version:
```bash
git pull origin main
```

### `429 Rate Limit` errors from Groq

Groq's free tier allows 30 requests/minute and 1,000 requests/day. If you hit the limit:
- Wait a minute (RPM limit) or until tomorrow (daily limit).
- Switch to Gemini: set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`.

### Emulator boots but agent says "Failed to connect"

- Run `adb devices` — does the device show as `device` (not `offline`)?
- Try `adb reconnect`.
- Restart the emulator.

### `pip install` fails

- Ensure Python 3.11+ is installed and `pip` is up to date:
  ```bash
  python --version
  pip install --upgrade pip
  ```
- On some systems, use `python3` and `pip3` instead of `python` and `pip`.

### Tests

Run the full test suite to verify your setup:

```bash
pytest tests/ -v
```

All 314 tests should pass.

---

_If you encounter an issue not listed here, open a GitHub Issue with the full error output and your OS/Python version._
