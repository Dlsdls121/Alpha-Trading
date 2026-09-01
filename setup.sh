#!/usr/bin/env bash
# One-shot setup for Linux, macOS and Termux.
#   curl -sSL <raw-url>/setup.sh | bash     -- or just run it from a clone.
set -euo pipefail

REPO_URL="https://github.com/Dlsdls121/Alpha-Trading.git"
BRANCH="claude/trading-platform-tablet-df45uw"
DIR="${ALPHA_DIR:-$HOME/alpha-trading}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m!  %s\033[0m\n' "$1"; }

# Termux needs its packages from pkg, not pip: numpy and pandas would otherwise
# try to compile from source on-device, which takes a very long time and often
# fails on a tablet.
if [ -n "${PREFIX:-}" ] && [ -d "/data/data/com.termux" ]; then
  say "Termux detected - installing prebuilt packages"
  pkg update -y && pkg upgrade -y
  pkg install -y python git binutils
  pkg install -y python-numpy python-pandas || \
    warn "python-numpy/python-pandas not available via pkg; pip will try to build them (slow)."
  TERMUX=1
else
  TERMUX=0
  command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
  command -v git >/dev/null || { echo "git is required"; exit 1; }
fi

say "Fetching the code into $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch origin "$BRANCH" && git -C "$DIR" checkout "$BRANCH" && git -C "$DIR" pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$DIR"
fi
cd "$DIR"

say "Installing Python dependencies"
if [ "$TERMUX" = "1" ]; then
  # Reuse Termux's system numpy/pandas rather than rebuilding them in a venv.
  pip install --upgrade pip
  pip install fastapi "uvicorn[standard]" httpx pydantic
  pip install --no-deps -e .
else
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -e ".[dev]" -q
fi

say "Checking the install"
if [ "$TERMUX" = "1" ]; then PY=python; else PY=./.venv/bin/python; fi
$PY -m alpha.cli expiries --no-color | head -8

cat <<MSG

==> Done.

  Start the dashboard:
      cd $DIR
      $( [ "$TERMUX" = "1" ] && echo "python -m alpha.cli serve" || echo "./.venv/bin/python -m alpha.cli serve" )

  Then open  http://localhost:8000  on this device,
  or  http://<this-device-ip>:8000  from another device on the same Wi-Fi.

  It starts in FIXTURE mode (simulated data, clearly labelled).
  To use real NSE data:
      export ALPHA_DATA_MODE=live
      $PY -m alpha.data.nse selftest

MSG
