#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=multilabel/solver.prototxt
WEIGHTS=models/alexnet/alexnet_train_iter_5000.caffemodel

$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --gpu 0 $@
#$CAFFE train --solver=$SOLVER --gpu 0 $@
