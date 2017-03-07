import numpy as np
from numpy import linalg as LA 
import math

np.set_printoptions(threshold=np.nan)
h = np.load('test.out.npy')
h = h[0]
h = h.round(3)
w,v = LA.eig(h) #w = evals, v = evects
count = 0
for i in range(len(h)):
	for j in range(len(h)):
		if i > j:
			continue
		if h[i,j] != h[j,i]:
			print(h[i,j])
			print(h[j,i])
			print(i)
			print(j)
			count += 1
print(count)
# for i in range(len(w)):
# 	if np.abs(w[i]) != 0:
# 		print(w)

print((h.transpose() == h).all()) #symmetric?


