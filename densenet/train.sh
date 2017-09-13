set -e

TOOLS=./caffe

$TOOLS train \
  --solver=solver.prototxt --gpu 1
