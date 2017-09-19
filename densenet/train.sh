set -e

TOOLS=./caffe
SOLVER=./densenet/solver.prototxt

$TOOLS train \
  --solver=$SOLVER --gpu 0,1,2,3
