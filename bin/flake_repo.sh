#!/bin/bash

DOCKERIMAGE="pythonexamples_scripts"
SOURCE_SCRIPTS_DIR="$(pwd)/scripts"
TARGET_SCRIPTS_DIR="/home/python_user/scripts"

docker run -v $SOURCE_SCRIPTS_DIR:$TARGET_SCRIPTS_DIR --rm $DOCKERIMAGE bash -c "flake8 --ignore='E501' scripts"
