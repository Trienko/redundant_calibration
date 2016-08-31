import numpy as np
import scipy as sp
import pylab as plt
import simulator
import analytic
from scipy.sparse import csr_matrix

class SPARC():

      def __init__(self,N,L,phi,zeta,PQ):

          self.N = N
          self.L = L
          self.B = (N**2-N)/2
          self.phi = phi
          self.zeta = zeta
          self.PQ = PQ
 
          self.v = np.array([])
          self.r = np.array([])

      def compute_v(self,z):
          g = z[:self.N] 
          y = z[self.N:]
           
         
          v = np.zeros((self.B,),dtype=complex)

          counter = 0
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  v[counter] = g[p-1]*np.conjugate(g[q-1])*y[self.phi[p-1,q-1]-1]
                  counter = counter + 1
          v = np.hstack([v,v.conjugate()])
        
          return v

      def vectorize_D(self,D):
          d = np.zeros((self.B,),dtype=complex)  
          
          counter = 0
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  d[counter] = D[p-1,q-1]
                  counter = counter + 1
          d = np.hstack([d,d.conjugate()])
        
          return d

      def compute_r(self,d,v):
          r = d-v
          return r

      def construct_R(self,r):
          r_new = r[:self.B]
          counter = 0
          R = np.zeros((self.N,self.N),dtype=complex)
          for k in xrange(self.N):
              for j in xrange(k+1,self.N):
                  R[k,j] = r_new[counter]
                  R[j,k] = np.conjugate(r_new[counter])
                  counter = counter + 1
          return R 

      def compute_J(self,z):
          g = z[:self.N]
          y = z[self.N:]

          rows = (self.N**2 - self.N)/2
          columns = self.N + self.L

          J1 = np.zeros((rows,columns),dtype=complex)
          J2 = np.zeros((rows,columns),dtype=complex)

          J = np.zeros((2*rows,2*columns),dtype=complex)
          
          p_v = np.zeros((rows,),dtype=int)
          q_v = np.zeros((rows,),dtype=int)
          
          counter = 0
          for k in xrange(self.N):
              for j in xrange(k+1,self.N):
                  p_v[counter] = k
                  q_v[counter] = j
                  counter = counter + 1
          
          for r in xrange(rows):
              p = p_v[r]
              q = q_v[r]
              phi_v = phi[p,q]
              for c in xrange(columns):
                  if c == p:
                     J1[r,c] = np.conjugate(g[q])*y[phi_v-1]            
                  if c == (self.N + phi_v - 1):
                     J1[r,c] = g[p]*np.conjugate(g[q])

          for r in xrange(rows):
              p = p_v[r]
              q = q_v[r]
              phi_v = phi[p,q]
              for c in xrange(columns):
                  if c == q:
                     J2[r,c] = g[p]*y[phi_v-1] 

          J[:rows,:columns] = J1
          J[:rows,columns:] = J2
          J[rows:,:columns] = J2.conjugate()
          J[rows:,columns:] = J1.conjugate()
          return J

      def compute_JHJ(self,J):
          return np.dot(J.transpose().conjugate(),J)

      def compute_JHr(self,J,r):
          return np.dot(J.transpose().conjugate(),r)

      def compute_JHr_formula(self,z,R):
          
          g = z[:self.N]
          y = z[self.N:]
          
          vec1 = np.zeros((self.N,),dtype=complex)
          for i in xrange(len(vec1)):
              sum_v = 0j
              for k in xrange(len(vec1)):
                  if k <> i:
                     if k > i:
                        x_ki = np.conjugate(y[self.zeta[i,k]-1])  
                     else:
                        x_ki = y[self.zeta[i,k]-1]
                     sum_v = sum_v + g[k]*x_ki*R[i,k]
              vec1[i] = sum_v

          vec2 = np.zeros((self.L,),dtype=complex)
          
          for i in xrange(len(vec2)):
              sum_v = 0j
              pq = PQ[str(i)]
              for k in xrange(len(pq)):
                  p = pq[k][0]
                  q = pq[k][1]
                  
                  sum_v = sum_v + np.conjugate(g[p])*g[q]*R[p,q]
              vec2[i] = sum_v

          return np.hstack([vec1,vec2,np.conjugate(vec1),np.conjugate(vec2)]) 

             

      '''
      INPUTS:
      z - Input vector.
      '''
      def generate_H_formula(self,z,psi_func_eval,xi_func_eval):
          g = z[:self.N]
          y = z[self.N:]
                    
          #CONSTRUCTING C
          #**************
          C = np.zeros((self.N,self.N),dtype=complex)
          for i in xrange(self.N):
              sum_v = 0j
              for k in xrange(self.N):
                  if k != i:
                     sum_v = np.absolute(g[k])**2*np.absolute(y[zeta[k,i]-1])**2 + sum_v
              C[i,i] = sum_v

          #CONSTRUCTING E
          #**************
          E = np.zeros((self.L,self.L),dtype=complex)
          for i in xrange(self.L):
              #pq = zip(*np.where(phi == (i+1))) #[(0, 0), (0, 1), (1, 2), (1, 3)]
              pq = self.PQ[str(i)]
              sum_v = 0j
              for k in xrange(len(pq)):
                  sum_v = np.absolute(g[pq[k][0]])**2*np.absolute(g[pq[k][1]])**2 + sum_v
              E[i,i] = sum_v

          #CONSTRUCTING D
          #**************
          D = np.zeros((self.N,self.L),dtype=complex)
          for i in xrange(self.N):
              for j in xrange(self.L):
                  psi,found = psi_func_eval(i,j+1,self.phi)
                  if found:
                     D[i,j] = g[i]*y[j].conj()*np.absolute(g[psi])**2 

          DH = D.transpose().conj()  

          #CONSTRUCTING F
          #**************
          F = np.zeros((self.N,self.N),dtype=complex)
          for i in xrange(self.N):
              for j in xrange(i,self.N):
                  if i <> j:
                     F[i,j] = g[i]*g[j]*np.absolute(y[self.phi[i,j]-1])**2
                     F[j,i] = F[i,j] 

          #CONSTRUCTING G
          #**************
          G = np.zeros((self.N,self.L),dtype=complex)
          for i in xrange(self.N):
              for j in xrange(self.L):
                  xi,found = xi_func_eval(i,j+1,self.phi)
                  
                  if found:
                     #print "i","j","xi",i+1,j+1,xi+1
                     #print "found",found
                     G[i,j] = g[i]*y[j]*np.absolute(g[xi])**2
                  #else:
                  #   print "i","j","xi",i+1,j+1,xi
                  #   print "found",found  

          GT = G.transpose() 

          #CONSTRUCTING Z
          #**************
          Z = np.zeros((self.L,self.L),dtype=complex)

          #CONSTRUCTING A
          #**************
          A = np.zeros((self.N+self.L,self.N+self.L),dtype=complex)
          A[:self.N,:self.N] = C
          A[:self.N,self.N:] = D
          A[self.N:,self.N:] = E
          A[self.N:,:self.N] = DH

          #CONSTRUCTING B
          #**************
          B = np.zeros((self.N+self.L,self.N+self.L),dtype=complex)
          B[:self.N,:self.N] = F
          B[:self.N,self.N:] = G
          B[self.N:,self.N:] = Z
          B[self.N:,:self.N] = GT

          #CONSTRUCTING H
          #**************
          H = np.zeros((2*(self.N+self.L),2*(self.N+self.L)),dtype=complex)
          H[:self.N+self.L,:self.N+self.L] = A
          H[:self.N+self.L,self.N+self.L:] = B
          H[self.N+self.L:,:self.N+self.L] = B.conj()
          H[self.N+self.L:,self.N+self.L:] = A.conj()

          H_sparse = csr_matrix(H,dtype=complex) #THIS MIGHT NOT BE THE FASTEST WHY - MAY NEED TO TWEAK THIS 

          return H,H_sparse



           
if __name__ == "__main__":
   s = simulator.sim(layout="REG",order=5) #INSTANTIATE OBJECT
   #s.read_antenna_layout()
   s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
   s.plot_ant(title="HEX") #PLOT THE LAYOUT
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   PQ = s.create_PQ(phi,s.L)
   s.uv_tracks() #GENERATE UV TRACKS
   #s.plot_uv_coverage(title="HEX") #PLOT THE UV TRACKS
   point_sources = s.create_point_sources(100,fov=3,a=2) #GENERATE RANDOM SKYMODEL
   g=s.create_antenna_gains(s.N,0.9,0.1,50,1,5,s.nsteps,plot = True) #GENERATE GAINS
   D,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=10,w_m=None) #CREATE VIS MATRIX
   #M,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=None,SNR=None,w_m=None) #PREDICTED VIS
   #s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   #s.plot_visibilities([0,1],M,"r",s=True)    
   
   g = np.random.randn(s.N)+1j*np.random.randn(s.N)
   y = np.random.randn(s.L)+1j*np.random.randn(s.L)

   z = np.hstack([g,y])

   print "z = ",z

   sparc_object = SPARC(s.N,s.L,phi,zeta,PQ)
   v = sparc_object.compute_v(z)
   J = sparc_object.compute_J(z)
   H_comp = sparc_object.compute_JHJ(J)
   H,H_sparse = sparc_object.generate_H_formula(z,s.psi_func_eval,s.xi_func_eval)
   d = sparc_object.vectorize_D(D[:,:,0])
   r = sparc_object.compute_r(d,v)
   JHr_com = sparc_object.compute_JHr(J,r)
  
   R = sparc_object.construct_R(r)
   
   JHr_fun = sparc_object.compute_JHr_formula(z,R)

   print "JHr_com = ",JHr_com
   print "len(JHr_com) = ", len(JHr_com)
   print "JHr_fun = ",JHr_fun
   print "len(JHr_fun) = ", len(JHr_fun)
         
   print "v = ",v
   plt.imshow(np.absolute(J),interpolation="nearest")
   plt.show()   

   plt.imshow(np.absolute(H_comp),interpolation="nearest")
   plt.show()

   plt.imshow(np.absolute(H),interpolation="nearest")
   plt.show()

   plt.imshow(np.absolute(H-H_comp),interpolation="nearest")
   plt.colorbar()
   plt.show()

    
   

