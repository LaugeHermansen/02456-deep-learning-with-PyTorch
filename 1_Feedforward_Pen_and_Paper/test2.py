import numpy as np

f = np.vectorize(lambda x: max(x,0))

a = np.array([2,6,5,-2,8-7,45,-1])

print(f(a))