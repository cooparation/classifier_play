#!/usr/bin/env sh
set -e

TOOLS=./classification
DEPLOY_PROTO=alexnet/deploy.prototxt
WEIGHTS=models/alexnet/alexnet_train_iter_5000.caffemodel
MEAN_BINARY=data/foodnet_mean.binaryproto
LABEL_NAMES=data/label_name.dat

$TOOLS $DEPLOY_PROTO $WEIGHTS \
    $MEAN_BINARY $LABEL_NAMES \
    ./testimages/201707141104355.jpg
#./testimages/20170905162336.jpg
#./testimages/20170828155158.jpg 
