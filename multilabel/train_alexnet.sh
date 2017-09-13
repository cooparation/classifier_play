#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=multilabel/solver.prototxt
#WEIGHTS=alexnet/bvlc_alexnet.caffemodel

#$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --gpu 0 $@
$CAFFE train --solver=$SOLVER --gpu 0 $@
