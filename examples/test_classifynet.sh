#!/usr/bin/env sh
set -e

#TOOLS=./classification
#DEPLOY_PROTO=../nets/deploy.prototxt
#WEIGHTS=../nets/alexnet_train_iter_5000.caffemodel
#MEAN_BINARY=../nets/foodnet_mean.binaryproto
#LABEL_NAMES=../nets/label_name.dat

TOOLS=./classification
DEPLOY_PROTO=../../nin/deploy.prototxt
WEIGHTS=../../models/nin/nin_train_iter_5000.caffemodel
#DEPLOY_PROTO=../../../foodClassifier/ResNet/ResNet_20_deploy.prototxt
#WEIGHTS=../../../foodClassifier/models/resnet/resnet_train_iter_60000.caffemodel

MEAN_BINARY=../../data/net_mean.binaryproto
LABEL_NAMES=../nets/label_name.dat

$TOOLS $DEPLOY_PROTO $WEIGHTS \
    $MEAN_BINARY $LABEL_NAMES \
    ./../../testimages/20170905162336.jpg
    #./../../testimages/201707141104355.jpg
    #./../../testimages/20170828155158.jpg 
