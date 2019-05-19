#!/bin/bash

echo "*** Setting up repo ***"

# Make script docker command executable
BINS=$(ls bin)
for f in ${BINS[@]}
do
   chmod +x bin/$f
done

# Ensure necessary directory structure
DIRS=( "output" "notebooks/data" "notebooks/templates" "notebooks/work" "scripts/work" )
for dir in ${DIRS[@]}
do
   mkdir -p $dir
done
