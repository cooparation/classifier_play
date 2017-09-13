#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=VGG/solver19.prototxt
WEIGHTS=VGG/VGG19_layers.caffemodel
OUTPUT=VGG/models19

find $OUTPUT/* | xargs rm -rf
$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --log_dir=$OUTPUT $@
