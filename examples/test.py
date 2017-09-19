import os
import ctypes
from ctypes import *

lib = ctypes.cdll.LoadLibrary("./build/libclassify.so")

testImgs = "../testimages/20170828155158.jpg"
model_file = "./nets/deploy.prototxt"
trained_file = "./nets/alexnet_train_iter_5000.caffemodel"
mean_file = "./nets/foodnet_mean.binaryproto"
label_file = "./nets/label_name.dat"
FLOAT_INPUT = c_float * 4
prob = FLOAT_INPUT()

INT_INPUT = c_int * 4
layerID = INT_INPUT()

paths = ['/home/liusj/datasets/layerIDDataSets/testSets']
types = {'0-empty':0, '1-bottom':1, '2-mid':2, '3-top':3}

num_images = 0
right_nums = 0
lines_write = []
for path in paths:
    for root, _, files in os.walk(path):
        for filename in files:
            filename = os.path.join(root, filename)
            print 'filename', filename
            if os.path.isfile(filename) and filename.find('.jpg') >0:
                label = filename.split('/')[6]
                print 'label ', label
                num_images += 1
                lib.getResults(filename, \
                        model_file, \
                        trained_file, \
                        mean_file, \
                        label_file, \
                        prob, layerID);
                print "type--id", types[label]
                if layerID[0] == types[label]:
                    right_nums +=1
                else:
                    lines_write += filename + ' ' + str(layerID[0])+ '\n'

                print "layerID", layerID[0]
                print "prob ", prob[0]

print "accuracy:",right_nums, '/', num_images, right_nums/num_images
lines_write += str(right_nums/num_images) + '\n'
file = open('result.txt', 'w')
file.writelines(lines_write)
file.close()
