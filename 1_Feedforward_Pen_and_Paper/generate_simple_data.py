import numpy as np
from matplotlib import pyplot as plt

N = 10000
D = 5
n_classes = 5
X_noise_size = 3
X_range = (0,21)

X_archetypes = np.random.randint(*X_range, size = (n_classes,D))
y_archetypes = np.eye(n_classes).astype(np)

X_noise = np.random.rand(N,D)*X_noise_size

idx = np.random.randint(0,n_classes,N)

X = X_archetypes[idx] + X_noise
y = y_archetypes[idx]


print(np.std(X,axis = 1))
print(np.std(X[y[:,0].astype(bool)],axis = 0))
print(np.std(X[y[:,1].astype(bool)],axis = 0))
print(np.std(X[y[:,2].astype(bool)],axis = 0))
print(np.mean(X,axis = 1))
plt.plot(np.mean(X[y[:,0].astype(bool)],axis = 0))
plt.plot(np.mean(X[y[:,1].astype(bool)],axis = 0))
plt.plot(np.mean(X[y[:,2].astype(bool)],axis = 0))
plt.show()


np.save("1_Feedforward_Pen_and_Paper/data/X.npy", X)
np.save("1_Feedforward_Pen_and_Paper/data/y.npy", y)
np.save("1_Feedforward_Pen_and_Paper/data/X_archetypes.npy",X_archetypes)
np.save("1_Feedforward_Pen_and_Paper/data/y_archetypes.npy",y_archetypes)