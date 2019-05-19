#!/bin/bash

DOCKERIMAGE="pythonprototypes_scripts"
SOURCE_SCRIPTS_DIR="$(pwd)/scripts"
TARGET_SCRIPTS_DIR="/home/python_user/scripts"
SOURCE_OUTPUT_DIR="$(pwd)/output"
TARGET_OUTPUT_DIR="/home/python_user/output"

echo "*** Running script $1 ***"

ARGS=$@

docker run -it \
  -v $SOURCE_SCRIPTS_DIR:$TARGET_SCRIPTS_DIR \
  -v $SOURCE_OUTPUT_DIR:$TARGET_OUTPUT_DIR \
  --rm \
  $DOCKERIMAGE \
  bash -c "python ${ARGS[@]}"
