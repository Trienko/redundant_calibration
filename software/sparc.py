import numpy as np
import scipy as sp
import pylab as plt
import simulator
import analytic

class SPARC():

      def __init__(self,N,L,phi,PQ):

          self.N = N
          self.L = L
          self.phi = phi
          self.PQ = PQ
 
          self.v = np.array([])
          self.r = np.array([])

      def compute_v(self,z,phi):
          g = z[:N] 
          y = z[N:]

          counter = 0
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  v[counter] = g[p-1]*np.conjugate(g[q-1])*y[phi[p-1,q-1]-1]

          v = np.vstack([v,v.conjugate()]]
        
          return v

   
if __name__ == "__main__":
   s = simulator.sim() #INSTANTIATE OBJECT
   #s.read_antenna_layout()
   s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
   s.plot_ant(title="HEX") #PLOT THE LAYOUT
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   #s.uv_tracks() #GENERATE UV TRACKS
   #s.plot_uv_coverage(title="HEX") #PLOT THE UV TRACKS
   #point_sources = s.create_point_sources(100,fov=3,a=2) #GENERATE RANDOM SKYMODEL
   #g=s.create_antenna_gains(s.N,0.9,0.1,50,1,5,s.nsteps,plot = True) #GENERATE GAINS
   #D,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=10,w_m=None) #CREATE VIS MATRIX
   #M,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=None,SNR=None,w_m=None) #PREDICTED VIS
   #s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   #s.plot_visibilities([0,1],M,"r",s=True)    
   
   g = np.random.randn(s.N)+1j*np.random.randn(s.N)
   y = np.random.randn(s.L)+1j*np.random.randn(s.L)

   sparc_object = SPARC()

    
   
    
   

