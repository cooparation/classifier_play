#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

LMDB_DATA_DIR=data/train
MEAN_DATA_DIR=data
TOOLS=./compute_image_mean

$TOOLS $LMDB_DATA_DIR $MEAN_DATA_DIR/net_mean.binaryproto

echo "Done."
