import numpy as np
import pylab as plt

def vector_random_complex_number(sigma,mu,N):
    x = (sigma * np.random.randn(N) + mu) + (sigma * np.random.randn(N) + mu)*1j
    return x 

def specific_sum(x):
    num = np.cumsum(x.conj())
    den = np.cumsum(x*x.conj())
    rat = num/den
    return rat



if __name__ == "__main__":
   
   '''
   x = vector_random_complex_number(10,5,10)
   one = np.ones(x.shape,dtype=int)
   one = np.cumsum(one)

   rat = specific_sum(x)
   
   plt.plot(one,np.absolute(x))
   plt.show()

   plt.plot(one,1/np.absolute(x))
   plt.plot(one,np.absolute(rat))
   plt.show()   
   
   
   '''
   x = np.random.uniform(0.9,1.1,1000)
   y = np.random.uniform(1,10,1000)
   one = np.ones(x.shape,dtype=int)
   one = np.cumsum(one)
   num = np.cumsum(x*y**2 + x**2*y)
   den = np.cumsum(x**2*y**2)
   rat = 1 + num/den

   plt.plot(one,rat)
   plt.show()	    


 
