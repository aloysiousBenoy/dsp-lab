#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2A. Energy of Signal
import numpy as np
import matplotlib.pyplot as plt

x=np.random.randint(1,4,4)
e=np.dot(x,x)

print("x(n)=",x)
print("Energy of x(n)=",e)


# In[2]:


#2B.Linear Convolution

import numpy as np
import matplotlib.pyplot as plt

def dirconv(x,h):
    k=len(x)-1
    m=len(h)-1
    if (k<m):
        print("Please pass longer sequence as first argument")
        # 12345 * 876
#          1 2 3 4 5 
#      1 2 3 

    y=np.zeros(k+m+1) #[0,0,0,0....]
    for i in np.arange(k+m+1):
        if (i < m+1):
            y[i]=np.dot(x[0:i+1],h[i::-1])
        elif (i < k+1):
            y[i]=np.dot(x[i-m:i+1],h[::-1])
        else:
            y[i]=np.dot(x[i-m:],h[m:i-k-1:-1])
    
    return y
    
x=np.array([1,2,3,4,5])
h=np.array([1,1,1])
y=dirconv(x,h)

nx=np.arange(0,len(x))
nh=np.arange(0,len(h))
ny=np.arange(0,len(y))

print("x(n)=",x)
print("h(n)=",h)
print("x(n)*h(n)=",y)
      
plt.xlabel("n")
plt.ylabel("x(n)")    
plt.stem(nx,x,use_line_collection=True)
plt.show()

plt.xlabel("n")
plt.ylabel("h(n)")    
plt.stem(nh,h,use_line_collection=True)
plt.show()

plt.xlabel("n")
plt.ylabel("x(n)*h(n)")    
plt.stem(ny,y,use_line_collection=True)
plt.show()


# In[10]:


#2C.Linear Correlation

import numpy as np
import matplotlib.pyplot as plt

def corr(y,x):
    k=len(y)-1
    m=len(x)-1
    if (k<m):
        print("Please pass longer sequence as first argument")
    w=np.zeros(k+m+1)
#          1 2 3 4 5 
#      1 2 3 
    for i in np.arange(-k,m+1):
        print("#{}#".format(i))
        if (i < -k+m):
            print(y[-i:],x[0:i+k+1],sep="\n")
            w[i]=np.dot(y[-i:],x[0:i+k+1])
        elif (i < 1):
            print(y[-i:-i+m+1],x[:],sep="\n")
            w[i]=np.dot(y[-i:-i+m+1],x[:])
        else:
            print(y[0:-i+m+1],x[i:m+1],sep="\n")
            w[i]=np.dot(y[0:-i+m+1],x[i:m+1])
        print("\n\n")
    return w


def p_corr(y,x):
    k=len(y)-1
    m=len(x)-1
    if (k<m):
        print("Please pass longer sequence as first argument")
    w=np.zeros(k+m+1)
#          1 2 3 4 5 
#            1 2 3 
    for i in np.arange(k+m+1):
        print("#{}#".format(i))
        if i<m+1:
            print(y[:i+1],x[m-i-1:],sep="\n")
            w[i]=np.dot(y[:i+1],x[m-i-1:])
        elif i<k+1:
            print(y[i-m:i],x[:],sep="\n")
            w[i]=np.dot(y[i-m:i],x[:])
        else:
            print(y[0:-i+m+1],x[i:m+1],sep="\n")
            w[i]=np.dot(y[i-m:],x[:i-k-1])
        print("\n\n")
    return w[::-1]

y=np.array([1,2,3,4,5])
x=np.array([1,2,3])
w=p_corr(y,x)

ny=np.arange(0,len(y))
nx=np.arange(0,len(x))
nw=np.arange(0,len(w))

print("y(n)=",y)
print("x(n)=",x)
print("Output w(n)=",w)
       
plt.xlabel("n")
plt.ylabel("y(n)")    
plt.stem(ny,y,use_line_collection=True)
plt.show()

plt.xlabel("n")
plt.ylabel("x(n)")    
plt.stem(nx,x,use_line_collection=True)
plt.show()

plt.xlabel("n")
plt.ylabel("w(n)")    
plt.stem(nw,w,use_line_collection=True)
plt.show()


# In[4]:


#4A. DFT

import numpy as np
import matplotlib.pyplot as plt

N=4
D=np.empty((N,N),dtype=np.cdouble)
W=np.exp(-1j*2*np.pi/N) #Twiddle factor

# 1 2 3 4    1 
# 1 2 3 4    1
# 1 2 3 4    1
# 1 2 3 4    1
for k in np.arange(N):
    for n in np.arange(N):
        D[k,n]=W**(k*n) #Twiddle factor matrix
np.round(D)

x=np.array([[1,2,3,4]]).T #Column vector
X=D @ x  #Matrix product

n=np.arange(0,N)
mag=np.zeros(N)
ph=np.zeros(N)

for k in np.arange(N):
    mag[k]=np.absolute(X[k])
    ph[k]=np.angle(X[k])

print("x(n)=",x)
print("X(k)=",np.round(X))

plt.title("Magnitude Spectrum") 
plt.xlabel("k")
plt.ylabel("|X(k)|")  
plt.stem(n,mag,use_line_collection=True)
plt.show()

plt.title("Phase Spectrum") 
plt.xlabel("k")
plt.ylabel("<X(k)")    
plt.stem(n,ph,use_line_collection=True)
plt.show()


# In[5]:


#4B. IDFT

import numpy as np

N=4
D=np.empty((N,N),dtype=complex)
W=np.exp(1j*2*np.pi/N) #Twiddle factor

for k in np.arange(N):
   for n in np.arange(N):
       D[k,n]=W**(k*n) #Twiddle factor matrix
np.round(D)

X=np.array([[10,-2+2j,-2,-2-2j]]).T #Column vector
x=(1/N)* (D @ X)  #matrix product divided by N

print("X(k)=",X)
print("x(n)=",np.round(x))


# In[19]:


#5A. FFT & IFFT

import numpy as np
import matplotlib.pyplot as plt

def FFT(x):
    N = len(x)
    if N == 1:
        return x
    else:
      X_even = FFT(x[::2])
      X_odd = FFT(x[1::2])
      factor = np.exp(-2j*np.pi*np.arange(N)/ N)
      X = np.concatenate([X_even+factor[:int(N/2)]*X_odd, X_even+factor[int(N/2):]*X_odd])

    return X

def IFFT(k):
  kx=np.asarray(k,dtype=complex)
  kconj=np.conjugate(kx)
  X=FFT(kconj)
  x=np.conjugate(X)
  x=x/k.shape[0]

  return np.round(x)

x=np.random.randint(0,10,32)
X=np.round(FFT(x)) 
print("x(n)=",x)
print("X(k)=",X) 



print("x(n) by IFFT=",IFFT(x))    

#Magnitude & Phase spectrum
N=len(x)
mag=np.zeros(N)
ph=np.zeros(N)
n=np.arange(N)
for k in np.arange(N):
    mag[k]=np.absolute(X[k])
    ph[k]=np.angle(X[k])

plt.title("Magnitude Spectrum") 
plt.xlabel("k")
plt.ylabel("|X(k)|")  
plt.stem(n,mag, markerfmt=" ", basefmt="-b",use_line_collection=True)
plt.show()

plt.title("Phase Spectrum") 
plt.xlabel("k")
plt.ylabel("<X(k)")    
plt.stem(n,ph,markerfmt=" ", basefmt="-b",use_line_collection=True)
plt.show()


# In[30]:


# 5B. FFT with sampling Rate.
# Plot amplitude spectrum for both 2-sided and 1-sided frequencies

import numpy as np
import matplotlib.pyplot as plt

def FFT(x):
    N = len(x)
    if N == 1:
        return x
    else:
      X_even = FFT(x[::2])
      X_odd = FFT(x[1::2])
      factor = np.exp(-2j*np.pi*np.arange(N)/ N)
      X = np.concatenate([X_even+factor[:int(N/2)]*X_odd, X_even+factor[int(N/2):]*X_odd])

    return X    

#With sampling
sr = 128  # sampling rate
ts = 1.0/sr #sampling interval
t = np.arange(0,1,ts)

freq = 5
x = 3*np.sin(2*np.pi*freq*t) 

freq = 10
x += np.sin(2*np.pi*freq*t)

freq =15  
x += 0.5*np.sin(2*np.pi*freq*t)

plt.plot(t,x)
plt.ylabel('Amplitude')
plt.show()

X=np.round(FFT(x))

#One-side and two-side frequency plots
#Calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 


plt.stem(freq, abs(X), markerfmt=" ", basefmt="-b",use_line_collection=True)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.show()

# Get the one-sided specturm and frequency
n_oneside = N//2
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_oneside =X[:n_oneside]/n_oneside

plt.stem(f_oneside, abs(X_oneside), markerfmt=" ", basefmt="-b",use_line_collection=True)
plt.xlabel('Freq (Hz)')
plt.ylabel('Normalized FFT Amplitude |X(freq)|')
plt.show()


# In[8]:


#6. Design of FIR filter by Hamming window

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def sinc_filter(M, fc):
    """Return an M + 1 point symmetric point sinc kernel with normalised cut-off 
            frequency fc 0->0.5."""
    if M%2:
        raise Exception('M must be odd')
    return np.sinc(2*fc*(np.arange(M + 1) - M/2))

def hamming(M):
    """Return an M + 1 point symmetric hamming window."""
    if M%2:
        raise Exception('M must be odd')
    return 0.54 - 0.46*np.cos(2*np.pi*np.arange(M + 1)/M)

def build_filter(M, fc, window=None):
    
    """Construct filter using the windowing method for filter parameters M
    number of taps, cut-off frequency fc and window. Window defaults to None 
    i.e. a rectangular window."""
    if window is None:
        h = sinc_filter(M, fc)
    else:
        h = sinc_filter(M, fc)*window(M)
    return h/h.sum()

f0 = 20 #20Hz
ts = 0.01 # i.e. sampling frequency is 1/ts = 100Hz
sr = 1/ts
x = np.arange(-10, 10, ts)
signal = (np.cos(2*np.pi*f0*x) + np.sin(2*np.pi*2*f0*x) + 
                np.cos(2*np.pi*0.5*f0*x) + np.cos(2*np.pi*1.5*f0*x))

#build filters
#Low pass
M = 100 #number of taps in filter
fc = 0.25 #i.e. normalised cutoff frequency 1/4 of sampling rate i.e. 25Hz
ham_lp = build_filter(M, fc, window=hamming)
y_ham = np.convolve(signal, ham_lp)

X = fft(signal)
Nx = len(X)
n = np.arange(Nx)
T = Nx/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude of Input |X(freq)|')
plt.show()

Y=fft(y_ham)
Ny = len(Y)
n = np.arange(Ny)
T = Ny/sr
freq = n/T 

plt.stem(freq, abs(Y), 'b', markerfmt=" ", basefmt="-b",use_line_collection=True)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude of Output |Y(freq)|')
plt.show()


# In[9]:


#7. Linear Convolution by Overlap Save Method

import numpy as np
from scipy.fft import fft, ifft

x=[3,-1,0,1,3,2,0,1,2,1]
TL=len(x)
h=[1,1,1]
M=len(h)      
y=[]

h=np.append(h,np.zeros(M-1))
x=np.append(x,np.zeros(M-TL%M))

b=int(np.ceil(TL/M)) #Total No of blocks

for i in np.arange(b):
  if i==0:
    xi=np.append(np.zeros(M-1),x[0:M])

  else:
    xi=x[i*M-(M-1):(i+1)*M]
  
  #Discarding first M-1 terms
  yn=ifft(fft(xi)*fft(h))
  yi=yn[M-1:2*M-1]
  y=np.append(y,yi)

print("x(n):",x)
print("h(n):",h)
print("By overlap save, x(n)*h(n)=",np.real(y))


# In[10]:


#Linear Convolution by Overlap Add Method

import numpy as np
from scipy.fft import fft, ifft
x = np.array([1,2,-1,2,3,-2,-3,-1,1,1,2,-1])
h = np.array([1,2])

M = len(h) #lengh of h
XL = len(x) #length of input sequence
FL = M+XL-1 #required length of the final convoluted sequence

N = 2*M-1 #size of each block
B = int(np.ceil(XL/M)) #number of blocks

Yi = np.zeros(FL+M-1) #to store the final sequence
 
for i in np.arange(B):
  Xi = x[i*M : (i+1)*M] #extracting each group
  y = np.real(np.round(ifft(np.multiply(fft(h,N), fft(Xi,N))))) #output of each block
  Yi[i*M:i*M+N] += y  #adding and storing the required points

#if len(Yi)>FL:
  #Yi = Yi[:FL] #discarding extra points if any
print ('The output sequence is:',Yi)

