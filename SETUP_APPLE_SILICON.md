# Setting up `eyetools` on an Apple Silicon Mac (M1/M2/M3/M4)

## Step 1 — Make sure Terminal itself is NOT running under Rosetta

This is the #1 gotcha. If your Terminal app is set to open with Rosetta, then *everything*
it launches — including a fresh native install — will still be x86_64.

1. Open **Finder → Applications** (or Applications → Utilities for Terminal).
2. Click your terminal app once (**Terminal**, or **iTerm** if you use that).
3. Press **Cmd+I** (Get Info).
4. Make sure **"Open using Rosetta" is UNCHECKED.**
5. If you had to change it, fully **quit and reopen** the terminal.

Verify your shell is native:

```bash
arch
```

This should print **`arm64`**. If it prints `i386` or `x86_64`, the terminal is still in
Rosetta — go back and uncheck the box.

---

## Step 2 — Install miniforge (native Apple Silicon)

1. Go to: https://github.com/conda-forge/miniforge
2. Download the installer for **macOS Apple Silicon (arm64)**.
   - The file is named like `Miniforge3-MacOSX-arm64.sh`.
   - **Do not** download the x86_64 / Intel one.
3. In the terminal, run the installer (adjust the filename/path to where it downloaded):

   ```bash
   bash ~/Downloads/Miniforge3-MacOSX-arm64.sh
   ```

4. Accept the license, accept the default install location (`~/miniforge3`), and when it
   asks whether to run `conda init`, answer **yes**.
5. **Close the terminal completely and open a new one** so the changes take effect.

---

## Step 3 — Confirm the new `conda` is the one being used

In the new terminal:

```bash
which conda
conda info --base
```

- `which conda` should point into **`miniforge3`** (e.g. `~/miniforge3/bin/conda`).
- `conda info --base` should be **`~/miniforge3`**.

If either still shows **`/opt/anaconda3`**, the old Anaconda is taking priority. Tell Ben
before continuing — we'll need to fix the order in your `~/.zshrc`. Do not proceed until
`conda` points at miniforge, or the new environment will come out x86_64 again.

Also confirm the base Python is native:

```bash
python -c "import platform; print(platform.machine())"
```

Should print **`arm64`**.

---

## Step 4 — Create the `eyetools` environment from the repo's YAML file

1. Go into the eyetools repo folder (wherever you cloned/downloaded it). For example:

   ```bash
   cd ~/Documents/eyetools-main
   ```

   (Use your actual folder name. It must contain the file `env.yml`.)

2. If you already have an old `eyetools` environment (the slow x86 one), remove it first
   so the name is free:

   ```bash
   conda env remove -n eyetools
   ```

   (If you'd rather keep the old one, skip this and instead create under a different name
   in the next step by adding `-n eyetools-arm`.)

3. Create the environment from the file:

   ```bash
   conda env create -f env.yml
   ```

   This will take a few minutes while it downloads native packages.

4. Activate it:

   ```bash
   conda activate eyetools
   ```

---

## Step 5 — VERIFY it is native (the only check that really matters)

```bash
python -c "import platform; print(platform.machine())"
```

This **must** print **`arm64`**.

- ✅ `arm64` → you're done. Loading will now be fast.
- ❌ `x86_64` → something upstream is still emulated (most likely Step 1 Terminal/Rosetta,
  or Step 3 `conda` still pointing at Anaconda). Recheck those steps or send Ben the output
  of `which conda`, `arch`, and the line above.