#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=googlenet/solver.prototxt
WEIGHTS=googlenet/googlenet.caffemodel
OUTPUT=googlenet/models

find $OUTPUT/* | xargs rm -rf
$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --log_dir=$OUTPUT $@ 
