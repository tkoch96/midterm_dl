import numpy as np
from numpy import linalg as LA 
import math
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)
h = np.load('test.out.npy')
h = h[0]
h = h.round(2)
w,v = LA.eig(h) #w = evals, v = evects
plt.hist(np.real(w))
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues of Hessian Matrix of the Loss Function')
plt.show()
w = w.round(3)
print(np.real(w))
print((h.transpose() == h).all()) #symmetric?


plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')