import numpy as np
import scipy as sp
import pylab as plt
import simulator

def LINCAL_J(z,N,L,phi):
    g = z[:N]
    y = z[N:]

    rows = (N**2 - N)/2
    columns = 2*N + 2*L

    J = np.zeros((rows,columns),dtype=complex)

    p_v = np.zeros((rows,),dtype=int)
    q_v = np.zeros((rows,),dtype=int)
          
    counter = 0
    for k in xrange(N):
        for j in xrange(k+1,N):
              p_v[counter] = k
              q_v[counter] = j
              counter = counter + 1

    for r in xrange(rows):
          p = p_v[r]
          q = q_v[r]
          phi_v = phi[p,q]
          for c in xrange(columns):
              if c == p:
                 J[r,c] = g[p]*np.conj(g[q])*y[phi_v-1]  
              if c == q:
                 J[r,c] = g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (N + p)):
                 J[r,c] = 1j*g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (N + q)):
                 J[r,c] = -1j*g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (2*(N) - 1 + phi_v)):
                 J[r,c] = g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (2*(N) + L - 1 + phi_v)):
                 J[r,c] = 1j*g[p]*np.conj(g[q])*y[phi_v-1]  

    J = np.vstack([J,np.conj(J)])     

    return J


if __name__ == "__main__":
   s = simulator.sim(layout="REG",order=5)
   s.generate_antenna_layout()
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   s.plot_ant(title="REG")
   s.plot_zeta(data=zeta,step1=1,step2=1,label_size=14,cs="cubehelix")
   
   g = np.random.randn(s.N)+1j*np.random.randn(s.N)
   y = np.random.randn(s.L)+1j*np.random.randn(s.L)
   z = np.hstack([g,y])

   J = LINCAL_J(z,s.N,s.L,phi)

   H = np.dot(J.conj().transpose(),J)

   plt.imshow(np.absolute(J),interpolation='nearest')

   plt.imshow(np.absolute(H),interpolation='nearest')

   plt.show()


 
    

   
