#!/usr/bin/env sh
TOOLS=./multi_convert_imageset

if [ -d "data/multitrainimage" ]; then
    rm -rf data/multitrainimage
fi

if [ -d "data/multitestimage" ]; then
    rm -rf data/multitestimage
fi

if [ -d "data/multitrainlabel" ]; then
    rm -rf data/multitrainlabel
fi

if [ -d "data/multitestlabel" ]; then
    rm -rf data/multitestlabel
fi

image_root_dir=''

re_width=227
re_height=227

# $TOOLS image_root_path file_list_path image_db_path \
#        label_db_path label_counts -backend=DB_TYPE
$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir/ data/multi_image_list_train.txt \
    data/multitrainimage data/multitrainlabel 2

$TOOLS --resize_width $re_width --resize_height $re_height \
    $image_root_dir/ data/multi_image_list_test.txt \
    data/multitestimage data/multitestlabel 2
