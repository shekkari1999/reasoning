#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo "==> Installing Python packages from requirements.txt..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

# Keep everything in workspace (mounted volume)
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export JUPYTER_DATA_DIR="$REPO_ROOT/.jupyter"
mkdir -p "$REPO_ROOT/.cache/huggingface" "$JUPYTER_DATA_DIR"

echo "==> Registering Jupyter kernel (in workspace .jupyter)..."
JUPYTER_DATA_DIR="$JUPYTER_DATA_DIR" "$VENV_DIR/bin/python" -m ipykernel install --name=reasoning --display-name="Python (reasoning)"

echo "==> Downloading model weights (Qwen3-0.6B) to workspace .cache..."
HF_HOME="$HF_HOME" "$VENV_DIR/bin/python" -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
print('  Done.')
"

echo "==> Setting workspace default Python and env (all in workspace)..."
mkdir -p .vscode
# .env so Python/Jupyter use workspace cache and kernel
echo "HF_HOME=$REPO_ROOT/.cache/huggingface" > .env
echo "JUPYTER_DATA_DIR=$REPO_ROOT/.jupyter" >> .env
cat > .vscode/settings.json << SETTINGS
{
  "python.defaultInterpreterPath": "\${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.envFile": "\${workspaceFolder}/.env",
  "terminal.integrated.env.linux": {
    "HF_HOME": "\${workspaceFolder}/.cache/huggingface",
    "JUPYTER_DATA_DIR": "\${workspaceFolder}/.jupyter"
  }
}
SETTINGS

echo "==> Configuring git..."
git config user.email "akhil.masters21@gmail.com"
git config user.name "shekkari1999"

echo "Done. Activate with: source $VENV_DIR/bin/activate"
echo "In Jupyter, select kernel: Python (reasoning)"
