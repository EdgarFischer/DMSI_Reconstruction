#!/usr/bin/env bash
set -euo pipefail

# 3) Container starten
docker run -d \
  --gpus all \
  --name mrjo \
  -p 8888:8888 \
  -v "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish:/workspace" \
  dl
