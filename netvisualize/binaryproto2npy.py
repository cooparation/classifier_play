import numpy as np
c=np.fromfile("mean.binaryproto", dtype=np.float)
print c
np.save("a.npy", c)
b=np.load('a.npy')
print b
