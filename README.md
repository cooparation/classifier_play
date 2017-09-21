# classifier study based on caffe

## Deps
* caffe: https://github.com/BVLC/caffe.git
* commits: iea455eb29393ebe6de9f14e88bfce9eae74edf6d

## Usage
* prepare datasets
* create the related soft links to caffe
* 1_dataset.py: generate train and test dataset lists, and write class name label list
* 2_images_to_lmdb.sh: convert images to lmdb format
* 3_compute_image_mean.sh: get image mean values
* modify the prototxt files, and prepare to train and test

## Outputs
* data
* models

## Test [result](https://github.com/cooparation/classifier_play/blob/master/TEST.md)   
