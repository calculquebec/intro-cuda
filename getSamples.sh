#!/bin/bash

if [ -z "$EBROOTCUDA" ]; then
    echo "Please load a cuda module. For example:"
    echo
    echo "    module load cuda"
    echo
    exit
fi

echo Getting samples from $EBROOTCUDA...

mkdir -p samples/common

rsync -av $EBROOTCUDA/samples/1_Utilities ./samples/
rsync -av $EBROOTCUDA/samples/common/inc  ./samples/common/
