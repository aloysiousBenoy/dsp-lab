import numpy as np
import matplotlib.pyplot as plt

N = 4
D = np.empty((N, N), dtype=np.cdouble)
W = np.exp(-1j*2*np.pi/N)
for k in np.arange(N):
    for n in np.arange(N):
        D[k, n] = W**(k*n)
np.round(D)
x = np.array([1, 2, 3, 4]).T
X = D@x

np.round(x)

plt.stem(X)
plt.xlabel("k")
plt.ylabel("X(k)")
plt.show()
