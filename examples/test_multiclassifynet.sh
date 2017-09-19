#!/usr/bin/env sh
set -e

TOOLS=./build/multi_classification
DEPLOY_PROTO=../multilabel/deploy.prototxt
WEIGHTS=../models/multilabel_alexnet/alexnet_train_iter_5000.caffemodel
MEAN_BINARY=../nets/foodnet_mean.binaryproto
LABEL_NAMES1=../data/multi_label_name2.dat
LABEL_NAMES2=../data/multi_label_name1.dat

$TOOLS $DEPLOY_PROTO $WEIGHTS \
    $MEAN_BINARY $LABEL_NAMES1 $LABEL_NAMES2 \
    ./../../testimages/20170905162336.jpg
    #./../../testimages/20170828155158.jpg 
