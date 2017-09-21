#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=VGG/solver16.prototxt
WEIGHTS=VGG/VGG16_layers.caffemodel
OUTPUT=VGG/models16

$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --log_dir=$OUTPUT $@
