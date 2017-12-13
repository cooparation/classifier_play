#!/usr/bin/env python

import numpy
import cv2
import os
import sys
import random

# 9 regions patches
def genMultiPatches(srcImage, srcDim=(256, 256), patchesDim=(224, 224)):
    if srcImage.shape != srcDim:
        resizedImage = cv2.resize(srcImage, srcDim)
    genImages=[]
    genImages.append(resizedImage[0:patchesDim[1], 0:patchesDim[0]])
    genImages.append(resizedImage[srcDim[1]-patchesDim[1]:srcDim[1], 0:patchesDim[1]])
    genImages.append(resizedImage[0:patchesDim[1], srcDim[0]-patchesDim[0]:srcDim[0]])
    genImages.append(resizedImage[srcDim[1]-patchesDim[1]:srcDim[1], srcDim[0]-patchesDim[0]:srcDim[0]])
    genImages.append(resizedImage[(srcDim[1]-patchesDim[1])/2:(srcDim[1]-patchesDim[1])/2+patchesDim[1],
                                  (srcDim[0]-patchesDim[0])/2:(srcDim[0]-patchesDim[0])/2+patchesDim[0]])
    genImages.append(resizedImage[(srcDim[1]-patchesDim[1])/2:(srcDim[1]-patchesDim[1])/2+patchesDim[1],
                                  0:patchesDim[0]])
    genImages.append(resizedImage[(srcDim[1]-patchesDim[1])/2:(srcDim[1]-patchesDim[1])/2+patchesDim[1],
                                  srcDim[0]-patchesDim[0]:srcDim[0]])
    genImages.append(resizedImage[0:patchesDim[1],
                                  (srcDim[0]-patchesDim[0])/2:(srcDim[0]-patchesDim[0])/2+patchesDim[0]])
    genImages.append(resizedImage[srcDim[1]-patchesDim[1]:srcDim[1],
                                  (srcDim[0]-patchesDim[0])/2:(srcDim[0]-patchesDim[0])/2+patchesDim[0]])
    return genImages

def genData(argv):
    srcList = argv[1]
    dstDir = argv[2]
    srcFp = open(srcList)
    l = srcFp.read().split('\n')
    l = l[:-1]
    srcFp.close()

    file_label = []
    for i in range(len(l)):
        line = l[i].split(' ')
        srcFilePath = line[0].split('/')
        category = srcFilePath[-2]
        name = srcFilePath[-1].split('.')[0]

        if not os.path.isdir(dstDir + '/' + category):
            os.makedirs(dstDir + '/' + category)
        srcImage = cv2.imread(line[0])
        patches = []
        # directly resize
        patches.append(cv2.resize(srcImage, (224, 224)))
        # Center crop
        if srcImage.shape[0] > srcImage.shape[1]:
            cropW = srcImage.shape[1]
        else:
            cropW = srcImage.shape[0]
        cropX = (srcImage.shape[1] - cropW) / 2
        cropY = (srcImage.shape[0] - cropW) / 2
        centerCropImage = srcImage[cropY:cropY+cropW, cropX:cropX+cropW]
        patches.append(cv2.resize(centerCropImage, (224, 244)))
        # Side Left Top crop
        sideLTCropImage = srcImage[0:cropW, 0:cropW]
        patches.append(cv2.resize(sideLTCropImage, (224, 244)))
        # Side Right Bottom crop
        sideRBCropImage = srcImage[srcImage.shape[0]-cropW:srcImage.shape[0], srcImage.shape[1]-cropW:srcImage.shape[1]]
        patches.append(cv2.resize(sideRBCropImage, (224, 244)))

        patches += genMultiPatches(srcImage)
        patches += genMultiPatches(centerCropImage)
        patches += genMultiPatches(sideLTCropImage)
        patches += genMultiPatches(sideRBCropImage)

        for i in range(len(patches)):
            if i == 0:
                dstFileName = dstDir + '/' + category + '/' + name + '.jpg'
            else:
                dstFileName = dstDir + '/' + category + '/' + name + '_' + str(i) + '.jpg'
            if not os.path.isfile(dstFileName):
                cv2.imwrite(dstFileName, patches[i])
            file_label.append((dstFileName, line[1]))

    prefix = 'patches'
    totalNumFiles = len(file_label)
    fileData = open(prefix+srcList, 'w')
    for i in range(totalNumFiles):
        fileData.write(file_label[i][0] + ' ' + file_label[i][1] + '\n')
    fileData.close()

if __name__ == '__main__':
    if (sys.argv) != 3:
        raise Exception('Usage: {:s} srcList dstDir'.format(sys.argv[0]))
    genData(sys.argv)


