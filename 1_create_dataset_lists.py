import os
import cv2
from random import shuffle
from collections import OrderedDict

paths = ['/home/liusj/datasets/layerIDDataSets/trainSets']
file_type_list =['GIF', 'gif', 'jpeg',  'bmp', 'png', 'JPG',  'jpg', 'JPEG']
#file_type_list = ['jpg']

filenames = OrderedDict()
## write the class name lists with manual
# [label_name label_index image_num]
filenames['0-empty'] = [0, 0]
filenames['1-bottom'] = [1, 0]
filenames['2-mid'] = [2, 0]
filenames['3-top'] = [3, 0]

write_lines = []
types = set()
for path in paths:
    for root, _, files in os.walk(path):
        for fname in files:
            types.add(fname.split('.')[-1])
            if fname.split('.')[-1] in file_type_list:
                file_path = os.path.join(root,fname)
                label_name = file_path.split(path+'/')[-1]
                label_name = label_name.split('/')[0]
                if label_name in filenames.keys():
                    label = filenames[label_name][0]
                    filenames[label_name][1] += 1
                else:
                    label = len(filenames)
                    filenames[label_name] =[label,1]
                # print(cv2.imread(file_path).shape)
                write_lines.append(file_path+' '+str(label)+'\n')
shuffle(write_lines)
print(types)
L  = int(len(write_lines)*0.1)
f = open('data/image_list_test.txt','w')
f.writelines(write_lines[:L])
f.close()

f = open('data/image_list_train.txt','w')
f.writelines(write_lines[L:])
f.close()

f = open('data/label_name.dat','w')
for key in filenames.keys():
    f.write('{:20}{}\n'.format(key, filenames[key][0]))
    print('{:20}{:10}{:10}'.format(key,filenames[key][0],filenames[key][1]))
f.close()
