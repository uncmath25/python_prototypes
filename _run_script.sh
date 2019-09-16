#!/bin/bash

DOCKERIMAGE="uncmath25/jupyter-notebook"
SOURCE_SCRIPTS_DIR="$(pwd)/scripts"
TARGET_SCRIPTS_DIR="/home/jovyan/scripts"
SOURCE_OUTPUT_DIR="$(pwd)/output"
TARGET_OUTPUT_DIR="/home/jovyan/output"

echo "*** Running script $1 ***"

ARGS=$@

docker run \
  -it \
  --rm \
  -v $SOURCE_SCRIPTS_DIR:$TARGET_SCRIPTS_DIR \
  -v $SOURCE_OUTPUT_DIR:$TARGET_OUTPUT_DIR \
  $DOCKERIMAGE \
  bash -c "python ${ARGS[@]}"
