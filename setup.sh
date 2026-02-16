#!/usr/bin/env bash
set -e

echo "==> Installing Python packages (transformers, accelerate)..."
pip install transformers accelerate

echo "==> Configuring git..."
git config user.email "akhil.masters21@gmail.com"
git config user.name "shekkari1999"

echo "Done."
