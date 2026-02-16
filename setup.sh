#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo "==> Installing Python packages (transformers, accelerate, ipykernel)..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install transformers accelerate ipykernel

echo "==> Registering Jupyter kernel..."
"$VENV_DIR/bin/python" -m ipykernel install --user --name=reasoning --display-name="Python (reasoning)"

echo "==> Downloading model weights (Qwen3-0.6B) to cache..."
"$VENV_DIR/bin/python" -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
print('  Done.')
"

echo "==> Setting workspace default Python to .venv..."
mkdir -p .vscode
cat > .vscode/settings.json << 'SETTINGS'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
SETTINGS

echo "==> Configuring git..."
git config user.email "akhil.masters21@gmail.com"
git config user.name "shekkari1999"

echo "Done. Activate with: source $VENV_DIR/bin/activate"
echo "In Jupyter, select kernel: Python (reasoning)"
