import numpy as np
import pylab as plt

def spar_ratio(N):
    num = 5*N**2-7*N+2
    den = 1.0*8*N**2-8*N+2
    
    gamma_N = (1.0)*num*den**(-1)  
    return gamma_N

def P_com(N):
    return 4*N-2

def k_com(N):
    P = P_com(N)
    k = (np.log(1-spar_ratio(N))+2*np.log(P))/np.log(P)
    return k

def c_com(N):
    P = P_com(N)
    c = np.log(0.25*P**2-1)/np.log(P)
    return c

if __name__ == "__main__":
   
   N_v = np.ones((200,),dtype=int)
   N_v = np.cumsum(N_v)
   N_v = N_v[N_v > 4]
   
   ones = np.ones(N_v.shape,dtype=int)
   gamma = ones*5./8
   gamma_v = spar_ratio(N_v)

   k_v = k_com(N_v)
   c_v = c_com(N_v)
   P_v = P_com(N_v)
   
   #print "gamma_v = ",gamma_v
   plt.plot(N_v,gamma_v)
   plt.plot(N_v,gamma,"r--")

   plt.xlabel("$N$")
   plt.ylabel("$\gamma_N$")
   plt.xlim([5,200])
   plt.show()   
   
   plt.plot(P_v,k_v,label="$k_N$")
   plt.plot(P_v,c_v,label="$c_N$")
   plt.plot(P_v,2*ones,"r--",label="Limit")
   plt.ylim([1.5,2.1])
   plt.xlim([4*5-2,4*200-2])
   plt.xlabel("$P$")
   plt.legend()
   plt.ylabel("Computational cost order")
   plt.show()

   plt.plot(N_v,k_v,label="$k_N$")
   plt.plot(N_v,c_v,label="$c_N$")
   plt.plot(N_v,2*ones,"r--",label="Limit")
   plt.ylim([1.5,2.1])
   plt.xlim([5,200])
   plt.xlabel("$N$")
   plt.legend()
   plt.ylabel("Computational cost order")
   plt.show()

   """
   plt.plot(P_v,c_v)
   plt.plot(P_v,2*ones,"r--")
   #plt.ylim([1.7,2.1])
   #plt.xlim([4*5-2,4*200-2])
   plt.xlabel("$P$")
   plt.ylabel("$c_N$")
   plt.show()

   plt.plot(N_v,c_v)
   plt.plot(N_v,2*ones,"r--")
   #plt.ylim([1.7,2.1])
   #plt.xlim([5,200])
   plt.xlabel("$N$")
   plt.ylabel("$c_N$")
   plt.show()
   """	    
	    


 
