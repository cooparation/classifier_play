#!/usr/bin/env sh
TOOLS=./convert_imageset

if [ -d "data/train" ]; then
    rm -rf data/train
fi

if [ -d "data/test" ]; then
    rm -rf data/test
fi

image_root_dir=''

re_width=227
re_height=227

$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir data/image_list_train.txt data/train

$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir data/image_list_test.txt data/test
