#!/usr/bin/env bash
set -euo pipefail

# Idempotent uv bootstrap for fresh pods/containers.
# - Installs uv (https://astral.sh/uv) into $HOME/.local/bin if missing.
# - Ensures uv is on PATH (symlink into /usr/local/bin when possible).
# - Optionally initializes a uv project if pyproject.toml doesn't exist.

log() { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

require_cmd() {
  if ! have "$1"; then
    log "error: missing required command: $1"
    exit 1
  fi
}

install_uv() {
  if have uv; then
    return 0
  fi

if have curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif have wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    log "error: neither curl nor wget is available to download uv"
    exit 1
  fi

  # If the installer put uv in ~/.local/bin, ensure current script can find it.
  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.local/bin/env"
  fi
}

ensure_on_path() {
  # Prefer /usr/local/bin so new shells/pods can find uv without sourcing env.
  if [[ -w /usr/local/bin ]]; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
      ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv
    fi
    if [[ -x "${HOME}/.local/bin/uvx" ]]; then
      ln -sf "${HOME}/.local/bin/uvx" /usr/local/bin/uvx
    fi
  fi

  if ! have uv; then
    log "warning: uv installed but not on PATH for this shell."
    log "         Run: source \"${HOME}/.local/bin/env\""
    log "         Or add ${HOME}/.local/bin to PATH in your shell rc."
    exit 1
  fi
}

maybe_init_project() {
  if [[ ! -f pyproject.toml ]]; then
    log "No pyproject.toml found; running: uv init"
    uv init
  fi
}

main() {
  require_cmd bash
  install_uv
  ensure_on_path

  log "uv: $(uv --version)"
  if have uvx; then
    log "uvx: $(uvx --version)"
  fi

  maybe_init_project
  log "setup complete"
}

main "$@"
