import matplotlib.pyplot as plt
import numpy as np

lines = open('models/caffe.INFO','r').readlines()

train_loss = []
test = [[] for _ in range(6)]
iter,iter_test = [],[]
i = 0
idx = [1,2,4,5,7,8]
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
        while 'Restarting data prefetching from start.' in lines[i]:
            i += 1
        for n,j in enumerate(idx):
            test[n].append(float(lines[i+j].split()[-1]))
        i += 9
    else:
        i += 1
plt.figure('train')
plt.plot(iter,train_loss)
plt.ylim([0,1])
plt.figure('test')
labels = ['loss1/top-1','loss1/top-5','loss2/top-1','loss2/top-5','loss3/top-1','loss3/top-2']
for i, t in enumerate(test):
    plt.plot(iter_test,t,label=labels[i])
plt.legend()

for i, t in enumerate(test):
    t = np.asarray(t,dtype=np.float32)
    print(labels[i], iter_test[t.argmax()], t.max())

plt.show()
# print(iter)
# print(train_loss)
# print(iter_test)
# print(test)
