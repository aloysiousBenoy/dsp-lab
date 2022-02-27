
import numpy as np
import matplotlib.pyplot as plt
 

def tf(k,N):
    return np.exp((-1j*2*np.pi*k)/N)

tf(0,4)
def fft(x):
    f = np.zeros_like(x,dtype=complex)
    for i in np.arange(len(f)):
        f[i]=fft_k(x,i) 
    return f

def fft_k(x,k):
    if(len(x)==1):
         return x[0]
    f= 0+0j
    x_even = x[::2]
    x_odd = x[1::2]
    
    fe= fft_k(x_even,k)
    fo = fft_k(x_odd,k)
    xfo= np.multiply(tf(k,len(x)) ,fo,dtype=complex)
    f=np.add(fe,xfo,dtype=complex)
    return f
    


x = np.array([1,2,3,4])

res = fft(x)


plt.stem(x,abs(res))
plt.title("Fast-FT")
plt.ylabel("FFT(x)")
plt.xlabel("samples")
plt.grid(True)
plt.show()
plt.savefig("e4_fft.pdf")


