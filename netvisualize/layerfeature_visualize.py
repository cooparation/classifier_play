#coding=utf-8
# more details to see http://blog.csdn.net/qq_30401249/article/details/51469184


### 1. setup
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
# %matplotlib inline # put the image in ipython notebook

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# load caffe
# The caffe module needs to be on the Python path;
# we'll add it here explicitly.
import sys
caffe_root='/home/liusj/dl/caffe'  # this file should be run from {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os

### 2.Load net and set up input preprocessing
caffe.set_mode_cpu()
model_def = './examples/nets/deploy.prototxt'
model_weights = 'examples/nets/nin_train_iter_50000.caffemodel'
mean_data = './data/nin_net_mean.npy'
input_image = './testimages/1.jpg'

if os.path.isfile(model_weights):
    print 'Caffe model found.'
else:
    print 'error: the model', model_weights, ' is not found'
    exit(1)

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(mean_data)
#print 'mean data', mu
#mu = mu.mean(1)  # average over pixels to obtain the mean (BGR) pixel values
mu = np.array([83, 79, 80])
print 'mean data', mu
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
#
#print 'data blob shape', net.blobs['data'].data.shape

### 3. CPU classification
# set the size of the input (we can skip this if we're happy
# with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

# Load an image (that comes with Caffe) and perform the preprocessing we've set up.
image = caffe.io.load_image(input_image)
transformed_image = transformer.preprocess('data', image)
plt.imsave('origin', image)

# Adorable! Let's classify it!

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

# The net gives us a vector of probabilities; the most probable class was the 281st one. But is 
# that correct? Let's check the ImageNet labels...

# load ImageNet labels
labels_file = './examples/nets/label_name.dat'
if not os.path.exists(labels_file):
    # ../data/ilsvrc12/get_ilsvrc_aux.sh
    print 'error: label file', labels_file, 'is not exist'

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'predicted label is:', labels[output_prob.argmax()]

### 4. Switching to GPU models
# Let's see how long classification took, and compare it to GPU mode.

#%timeit net.forward()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net.forward()  # run once before timing to set up memory
#%timeit net.forward()

# That should be much faster!

### 5. Examining intermediate output
print 'for each layer, show the output shape'
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
print 'for each layer, show the net param'
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data, name_str):
    """Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imsave(name_str, data)
    #plt.imshow(data); plt.axis('off')

# the parameters are a list of [weights]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1), 'conv1_w')
# The first layer output, conv1 (rectified responses of the filters above, first 96 only)
# conv1 (96, 3, 11, 11) (96,)

# show the first three filters
vis_square(filters[:96].reshape(96*3, 11, 11), 'conv2_w_3th')

# the parameters are a list of biases
filters_b = net.params['conv1'][1].data
# The first layer output, conv1 (rectified responses of the filters above, first 96 only)
# the params in conv1 is (96, 3, 11, 11) (96,)
print 'filters_b:'
print filters_b

#now the output after conv1 layer
# conv1 (1, 96, 55, 55)
feat = net.blobs['conv1'].data[0]
vis_square(feat, 'conv1_o')

# show the output after pool1 layer
# pool1 (1, 96, 27, 27)
feat = net.blobs['pool0'].data[0]
vis_square(feat, 'pool0_o')

# show the output after norm1 layer
# norm1 (1, 96, 27, 27)
feat = net.blobs['cccp1'].data[0]
vis_square(feat, 'cccp1_o')

# the parameters are a list of weights in conv2 layer
filters = net.params['conv2'][0].data
vis_square(filters[:256].reshape(256*96, 5, 5), 'conv2')

# the parameters are a list of biases.
filters_b = net.params['conv2'][1].data
#vis_square(filters.transpose(0, 2, 3, 1))

# The first layer output, conv1 (rectified responses of the filters above, first 96 only)
print filters_b
#conv2   (256, 48, 5, 5) (256,)

# show the result after conv2
feat = net.blobs['conv2'].data[0]
vis_square(feat, 'conv2_o')
# conv2 (1, 256, 27, 27)

# show the result after pooling2
feat = net.blobs['pool2'].data[0]
vis_square(feat, 'pool2_o')
# pool2 (1, 256, 13, 13)

# show the result after conv3
feat = net.blobs['conv3'].data[0]
vis_square(feat, 'conv3_o')
# conv3 (1, 384, 13, 13)

# show the result after conv4
feat = net.blobs['conv4'].data[0]
vis_square(feat, 'conv4_o')
# conv4 (1, 384, 13, 13)

'''
# show the result after conv5
feat = net.blobs['conv5'].data[0]
vis_square(feat)
# conv5 (1, 256, 13, 13)

# show the result after pooling layer 5
feat = net.blobs['pool5'].data[0]
vis_square(feat)
# pool5 (1, 256, 6, 6)

# show the result after fc6 layer
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# fc6   (1, 4096)

# show the result after fc7
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# fc7   (1, 4096)

# show the result after fc8
feat = net.blobs['fc8'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# fc8   (1, 1000)

# show the result after prob layer
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
# prob  (1, 1000)
'''
