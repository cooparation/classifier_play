#!/usr/bin/env sh
set -e

TOOLS=./classification
DEPLOY_PROTO=../nets/deploy.prototxt
WEIGHTS=../nets/alexnet_train_iter_5000.caffemodel
MEAN_BINARY=../nets/foodnet_mean.binaryproto
LABEL_NAMES=../nets/label_name.dat

$TOOLS $DEPLOY_PROTO $WEIGHTS \
    $MEAN_BINARY $LABEL_NAMES \
    ./../../testimages/20170905162336.jpg
    #./../../testimages/20170828155158.jpg 
