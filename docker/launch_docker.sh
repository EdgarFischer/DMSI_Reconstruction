#!/usr/bin/env bash

eval "$(ssh-agent -s)"
ssh-add   # to load your key
echo $SSH_AUTH_SOCK   # confirm it’s non‑empty

#2  Pfad ausgeben, damit du siehst, was gemounted wird
echo "✓ Verwende SSH_AUTH_SOCK: $SSH_AUTH_SOCK"

docker run -d \
  --gpus all \
  --name mrjo \
  -p 8888:8888 \
  -v "$SSH_AUTH_SOCK":/ssh-agent.sock \
  -e SSH_AUTH_SOCK=/ssh-agent.sock \
  -v /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish:/workspace \
  dl

# Git vertraut jetzt Deinem gemounteten Repo, ganz ohne chown o.Ä.
docker exec mrjo \
  git config --global --add safe.directory /workspace/Deuterium_Reconstruction
