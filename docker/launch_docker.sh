#!/usr/bin/env bash
set -euo pipefail

# 1) SSH‑Agent starten und Key laden
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa   # oder passe an, falls dein Key anders heißt

# 2) Sicherstellen, dass SSH_AUTH_SOCK gesetzt ist
if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  echo "✗ SSH_AUTH_SOCK ist leer – bitte mit Agent‑Forwarding verbinden (ssh -A …)."
  exit 1
fi

echo "✓ Verwende SSH_AUTH_SOCK: ${SSH_AUTH_SOCK}"

GIT_NAME="$(git config --get user.name)"
GIT_EMAIL="$(git config --get user.email)"

# 3) Container starten
docker run -d \
  --gpus all \
  --name mrjo \
  -p 8888:8888 \
  -v "${SSH_AUTH_SOCK}:/ssh-agent.sock" \
  -e SSH_AUTH_SOCK=/ssh-agent.sock \
  -e GIT_AUTHOR_NAME="${GIT_NAME}" \
  -e GIT_AUTHOR_EMAIL="${GIT_EMAIL}" \
  -e GIT_COMMITTER_NAME="${GIT_NAME}" \
  -e GIT_COMMITTER_EMAIL="${GIT_EMAIL}" \
  -v "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish:/workspace" \
  dl

# 4) Git safe.directory setzen (einmalig)
docker exec -u hostuser mrjo \
  git config --global --add safe.directory /workspace/Deuterium_Reconstruction

echo "✅ Container 'mrjo' läuft und Git safe.directory ist gesetzt."
