import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

hess = np.load('test.out.npy')
hess = hess[0]
w,v = LA.eig(hess)

h = plt.hist(np.abs(w))
stuff = np.histogram(np.abs(w))

