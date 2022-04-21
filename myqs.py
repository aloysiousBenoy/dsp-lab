import numpy as np

# for rotating once
# pop the last element of the array, then shift all elements one position to right
# then insert the popped element to the start of the array
 
def rotate_once(x):
    y = np.zeros_like(x)
    last = x[-1]
    for i in np.arange(len(x)-1):
        y[i+1]=x[i]
    y[0]=last
    return y

1 2 3 4
1 4 3 2 
# takes sequence in ccw and rotates n times
# This function repeatedly rotates the given sequence
def rotate(x,n):
    x = x[::-1]
    for i in np.arange(n+1):
        x = rotate_once(x)
    return x
 
# finds the circ. conv of x and h by rotaing and mult
def circ_conv(x,h):
    y = np.zeros_like(x)
    for i in np.arange(len(x)):
        y[i] = np.dot(x,rotate(h,i))
    return y
 
 
 
x = np.array([1,1,0,3])
h = np.array([2,3,1,1])
circ_conv(x,h)    


def dft(x):
    N = len(x)
    w = np.exp(-1j*2*np.pi/N)
    tw = np.zeros((N,N),dtype=np.cdouble)
    for i in np.arange(N):
        for j in np.arange(N):
            tw[i,j] = w**(i*j)
    return (tw@(x.T))
 
def idft(x):
    N = len(x)
    w = np.exp(1j*2*np.pi/N)
    tw = np.zeros((N,N),dtype=np.cdouble)
    for i in np.arange(N):
        for j in np.arange(N):
            tw[i,j] = w**(i*j)
    return np.round((tw@(x.T))/N)
 
X = dft(x)

H = dft(h)
# circ conv using dft and idft
conv = idft(X*H)
 
print("x = ",x)
print("DFT(x) = ",X)
print("h = ",h)
print("DFT(h) = ",H)
print("result of circular convolution = ",conv)

# linear conv using circular conv



z = len(x)+len(h)-1
xp = np.append(x,np.zeros(z))
hp = np.append(h,np.zeros(z))
 
print(xp,hp)
 
linconv = idft(dft(xp)*dft(hp))[0:z]
print(linconv)
 