import sys
import numpy as np
import matplotlib.pyplot as plt

path = 'models19/'
plot_name = 'VGG19'
if len(sys.argv) > 1:
    if sys.argv[1] == 'VGG16':
        path = 'models16/'
        plot_name = 'VGG16'
    elif sys.argv[1] == 'VGG19':
        path = 'models19/'
        plot_name = 'VGG19'
    else:
        raise 'Please choose VGG16 or VGG19.'

lines = open(path+'caffe.INFO','r').readlines()
if len(lines) < 1:
    raise 'File error: ' + path + 'caffe.INFO'

train_loss = []
test = []
iter,iter_test = [], []

i = 0
while i < len(lines):
    line = lines[i]


    if 'solver.cpp:218] Iteration' in line:
        line = line.split()
        iter.append(int(line[5]))
        train_loss.append(float(line[-1]))
        i += 1
    elif 'Testing net (#0)' in line:
        line = line.split()
        # print(line)
        iter_test.append(int(line[5][:-1]))
        i += 1
        while 'Test net output' not in lines[i]:
            i += 1

        test.append(float(lines[i].split()[-1]))
        i += 1
    else:
        i += 1

plt.figure(plot_name)
plt.plot(iter, train_loss,label='train')
plt.ylim([0, 1])
plt.title(plot_name)
# plt.figure('test')
plt.plot(iter_test, test,label='test')
plt.legend()


t = np.asarray(test, dtype=np.float32)
print(iter_test[t.argmax()], t.max())

plt.show()
# print(iter)
# print(train_loss)
# print(iter_test)
# print(test)