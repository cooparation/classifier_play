#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=nin/solver.prototxt
#WEIGHTS=alexnet/bvlc_alexnet.caffemodel
WEIGHTS=models/alexnet/alexnet_50000.caffemodel

#$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --gpu 0 $@
$CAFFE train --solver=$SOLVER --gpu 0 $@
