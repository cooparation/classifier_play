#!/usr/bin/env sh

TOOLS=./convert_imageset

if [ -d "data/nintrain" ]; then
    rm -rf data/nintrain
fi

if [ -d "data/nintest" ]; then
    rm -rf data/nintest
fi

image_root_dir=''

re_width=224
re_height=224


$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir/ data/image_list_train.txt data/nintrain

$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir/ data/image_list_test.txt data/nintest
