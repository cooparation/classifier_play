#!/usr/bin/env sh
TOOLS=./convert_imageset

if [ -d "data/densenettrain" ]; then
    rm -rf data/densenettrain
fi

if [ -d "data/densenettest" ]; then
    rm -rf data/densenettest
fi

image_root_dir=''

re_width=224
re_height=224

$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir/ data/image_list_train.txt data/densenettrain

$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir/ data/image_list_test.txt data/densenettest
