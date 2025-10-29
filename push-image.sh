#!/usr/bin/env bash
set -e
USER="louick"
IMAGE="painel-ods"
TAG="latest"
FULL="$USER/$IMAGE:$TAG"
docker build -t "$FULL" .
docker login
docker push "$FULL"
echo "Imagem enviada: $FULL"
