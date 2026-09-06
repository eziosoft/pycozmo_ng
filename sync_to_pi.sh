#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PI_HOST="e@192.168.10.100"
PI_DEST="~/pycozmo/"

rsync -avz --delete \
  --exclude-from='.gitignore' \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='pycozmo.egg-info/' \
  --exclude='sync_to_pi.sh' \
  ./ "$PI_HOST:$PI_DEST"
