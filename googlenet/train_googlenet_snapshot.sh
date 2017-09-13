#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=googlenet/solver.prototxt

OUTPUT=googlenet/models
solverstate=`ls $OUTPUT/*.solverstate | sort -rn | head -n 1`
echo $solverstate
$CAFFE train --solver=$SOLVER --snapshot=$solverstate --log_dir=$OUTPUT $@ 
