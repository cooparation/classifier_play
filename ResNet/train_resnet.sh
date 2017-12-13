#!/usr/bin/env sh
set -e

CAFFE=./caffe

# resnet50
SOLVER=ResNet/ResNet_50_solver.prototxt

WEIGHTS=./models/resnet/resnet50_train_iter_40000.caffemodel
SNAPSHOT=./models/resnet/resnet50_train_iter_40000.solverstate

OUTPUT=models/resnet

$CAFFE train --solver=$SOLVER --gpu 0,1,2,3 --log_dir=$OUTPUT $@
#$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --gpu 0,1,2,3 $@
#$CAFFE train --solver=$SOLVER --snapshot=$SNAPSHOT --gpu 2,3 $@
