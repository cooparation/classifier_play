import os
import cv2
from random import shuffle
from collections import OrderedDict

paths = ['/home/liusj/datasets/layerIDDataSets/trainSets']
file_type_list =['GIF', 'gif', 'jpeg',  'bmp', 'png', 'JPG',  'jpg', 'JPEG']
#file_type_list = ['jpg']

## write the class name lists with manual
# [label_name label_index image_num]
label_dict1 = OrderedDict()
label_dict1['0-empty'] = [0, 0]
label_dict1['1-bottom'] = [1, 0]
label_dict1['2-mid'] = [2, 0]
label_dict1['3-top'] = [3, 0]

label_dict2 = OrderedDict()

write_lines = []
types = set()
for path in paths:
    for root, _, files in os.walk(path):
        for fname in files:
            types.add(fname.split('.')[-1])
            if fname.split('.')[-1] in file_type_list:
                file_path = os.path.join(root,fname)
                label_name = file_path.split(path+'/')[-1]

                label_name1 = label_name.split('/')[0]
                label_name2 = label_name.split('/')[1]
                if label_name1 in label_dict1.keys():
                    label1 = label_dict1[label_name1][0]
                    label_dict1[label_name1][1] += 1
                else:
                    label1 = len(label_dict1)
                    label_dict1[label_name1] =[label1, 1]

                if label_name2 in label_dict2.keys():
                    label2 = label_dict2[label_name2][0]
                    label_dict2[label_name2][1] += 1
                else:
                    label2 = len(label_dict2)
                    label_dict2[label_name2] =[label2, 1]
                # print(cv2.imread(file_path).shape)
                write_lines.append(file_path + ' ' \
                        + str(label1) + ' ' \
                        + str(label2) + '\n')
shuffle(write_lines)
print(types)
L  = int(len(write_lines)*0.1)
f = open('data/multi_image_list_test.txt','w')
f.writelines(write_lines[:L])
f.close()

f = open('data/multi_image_list_train.txt','w')
f.writelines(write_lines[L:])
f.close()

f = open('data/multi_label_name1.dat','w')
for key in label_dict1.keys():
    f.write('{:20}{:10}\n'.format(key, label_dict1[key][0]))
    print('{:20}{:10}{:10}'.format(key,label_dict1[key][0],label_dict1[key][1]))
f.close()

f = open('data/multi_label_name2.dat','w')
for key in label_dict2.keys():
    f.write('{:20}{:10}\n'.format(key, label_dict2[key][0]))
    print('{:20}{:10}{:10}'.format(key,label_dict2[key][0],label_dict2[key][1]))
f.close()
