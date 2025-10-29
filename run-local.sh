#!/usr/bin/env bash
set -e
PORT=${1:-8050}
docker build -t louick/painel-ods:local .
docker run --rm -p ${PORT}:8050 -e PORT=${PORT} louick/painel-ods:local
