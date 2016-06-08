import numpy as np
import pylab as plt

'''
Simulator that can create uv-tracks and scalar visibilities associated with a regular east-west array
'''

class sim():

      '''
      Constructor
      '''
      def __init__(self):
          pass
	
      '''
      Generates the uv-coverage of a regularly spaced east-west (EW) array
      RETURNS:
      u_m - a N x N matrix containing the baseline u coordinates of a regurlarly spaced EW array
      v_m - a N x N matrix containing the baseline v coordinates of a regurlarly spaced EW array
      w_m - a N x N matrix containing the baseline w coordinates of a regurlarly spaced EW array
      
      INPUTS:
      spacing_in_meters - regular spacing between the array elements in m
      N - the amount of elements in the array
      lam - observational wavelength
      dec - declination of observation
      H_min - starting hour angle (measured in hours)
      H_max - maximum hour angle (measured in hours)
      time_steps - amount of timesteps
      plot - plots the antenna layout and uv-coverage if True
      '''
      def create_uv_regular_line(self,spacing_in_meters=20,N=5,lam=0.21428571428571427,dec=60,H_min=-6,H_max=6,time_steps=600,plot=True):
          b = np.ones((N,),dtype=float)*spacing_in_meters
          b = np.cumsum(b)
          
          if plot:
             plt.plot(b,np.zeros((N,),dtype=float),'ro')
             plt.xlabel("East-West [m]", fontsize=18)
             plt.ylabel("North-South [m]", fontsize=18)
             plt.title("ENU-coordinates of interferometer.", fontsize=18)
             plt.show()
          
          b = b/lam
          
          bas = np.zeros(((N**2-N)/2,),dtype=float)
          bas_2 = np.zeros(((N**2-N)/2,),dtype=float)
          counter = 0
          
          for p in xrange(0,len(b)-1):
              for q in xrange(p+1,len(b)):
                  #bas[counter] = b[q] - b[p] #Taking Oleg's definition of the baseline vector
                  bas[counter] = b[p] - b[q] 
                  bas_2[counter] = q - p
                  counter = counter + 1          
 
          H = np.linspace(H_min,H_max,time_steps)*(np.pi/12) #Hour angle in radians
          delta = dec*(np.pi/180) #Declination in radians

          u = np.zeros((len(bas),len(H)))
          v = np.zeros((len(bas),len(H)))
          w = np.zeros((len(bas),len(H)))
          for k in xrange(len(bas)):
              u[k,:] = bas[k]*np.cos(H)
              v[k,:] = bas[k]*np.sin(H)*np.sin(delta)
              w[k,:] = -1*bas[k]*np.cos(delta)*np.sin(H)
              if plot:
                 plt.plot(u[k,:],v[k,:],"r")
                 plt.plot(-u[k,:],-v[k,:],"b")
                 plt.xlabel("$u$ [rad$^{-1}$]", fontsize=18)
                 plt.ylabel("$v$ [rad$^{-1}$]", fontsize=18)
                 plt.title("$uv$-Coverage", fontsize=18)
                 
          if plot:
             plt.show()

          u_m = np.zeros((len(b),len(b),len(H)))
          v_m = np.zeros((len(b),len(b),len(H)))
          w_m = np.zeros((len(b),len(b),len(H)))

          b_m = np.zeros((len(b),len(b)),dtype=int)

          counter = 0
          for p in xrange(0,len(b)-1):
              for q in xrange(p+1,len(b)):
                  #print "p,q = ",p,",",q
                  
                  u_m[p,q,:] = u[counter,:] #the first two entries denote p and q and the third index denotes time
                  u_m[q,p,:] = -1*u[counter,:]

                  v_m[p,q,:] = v[counter,:]
                  v_m[q,p,:] = -1*v[counter,:]
                  
                  w_m[p,q,:] = w[counter,:]
                  w_m[q,p,:] = -1*w[counter,:]
                  
                  b_m[p,q] = bas_2[counter]
                  counter = counter + 1 

          #plt.plot(u_m[0,1,:],v_m[0,1,:])
          #plt.plot(u_m[0,2,:],v_m[0,2,:])
          #plt.show()
          return u_m,v_m, w_m, b_m	
          
      '''
      Generate flux values of sources (power law distribution)
      RETURNS:
      y - an array of pareto distributed samples of size num_sources
      
      INPUTS:
      a - pareto distribution (power law) shape parameter. 
      num_sources - the amount of values to draw from the pareto distribution.
      plot - if true plot the distribution.
      '''
      def generate_flux(self,a = 1.2,num_sources = 10,plot = False):
          y = np.random.pareto(a, size=num_sources)
          if plot:
             count, bins, ignored = plt.hist(y, 100, normed=True)
             plt.xlabel("$x$")
             plt.ylabel("$p(x)$")
             plt.title("Pareto Distribition, $a = $%.2f"%(a))
             plt.show()
          return y

      '''
      Generate flux values of sources (uniform distribution)
      RETURNS:
      an array of uniformly distributed samples of size num_sources
      
      INPUTS:
      min_v - the lower boundary of the uniform distribution. 
      max_v - the upper boundary of the uniform distribution.
      num_sources - the amount of values to draw from the uniform distribution.
      '''
      def generate_flux_unif(self,min_v=0.1,max_v=1,num_sources=10):
          return np.random.uniform(low=min_v, high=max_v, size = num_sources)

      '''
      Generate position values of sources (uniform distribution)
      
      RETURNS:
      an array of uniformly distributed samples of size num_sources
      
      INPUTS:
      fov - the fov in which the sources reside (in degrees)
      num_sources - the amount of values to draw from the uniform distribution.
      '''
      def generate_pos(self,fov = 3,num_sources=10):
          return np.random.uniform(low=-1*np.absolute(fov), high=np.absolute(fov), size = num_sources)*(np.pi/180)
          
      '''
      Creates an observed visibility matrix
       
      RETURNS:
      D - observed visibility matrix, N x N x timeslots
      
      INPUTS:
      point_sources - list of point sources to simulate
      u_m - N x N x timeslots matrix of u coordinates
      v_m - N x N x timeslots matrix of v coordinates
      g - N x t antenna gains to corrupt the visibilities with
      sig - the noise std
      v_m - N x N x timeslots matrix of w coordinates. If none just do 2D simulation
      '''      
      def create_vis_mat(self,point_sources,u_m,v_m,g,sig,w_m = None):
          D = np.zeros(u_m.shape)
          #print "D.shape = ",D.shape
          #print "G.shape = ",G.shape
          #Step 1: Create Model Visibility Matrix
          for k in xrange(len(point_sources)): #for each point source
              #print "k = ",k
              l_0 = point_sources[k,1]
              m_0 = point_sources[k,2]
              #print "point_sources[k,0] = ",point_sources[k,0]
              if w_m is None:
		D = D + point_sources[k,0]*np.exp(-2*np.pi*1j*(u_m*l_0+v_m*m_0))
	      else:
		n_0 = np.sqrt(1 - l_0**2 - m_0**2)
		D = D + point_sources[k,0]*np.exp(-2*np.pi*1j*(u_m*l_0+v_m*m_0+w_m*n_0))
    
          for t in xrange(D.shape[2]): #for each time-step
              #print "t = ",t
              if g is not None:
                 G = np.diag(g[:,t])
                 #Step 2: Corrupting the Visibilities 
                 D[:,:,t] = np.dot(G,D[:,:,t])
                 D[:,:,t] = np.dot(D[:,:,t],G.conj()) 
                      
              #Step 3: Adding Noise
              D[:,:,t] = D[:,:,t] + sig*np.random.randn(u_m.shape[0],u_m.shape[1]) + sig*np.random.randn(u_m.shape[0],u_m.shape[1])*1j
    
          return D
          
      '''
      Creates a point sources array of dimension: num_sources x 3 
      
      RETURNS:
      point_sources - a point sources array with dimension num_sources x 3, with the second dimension denoting flux, l_0 and m_0 (in
      degrees) respectively.
      
      INPUTS:
      num_sources - number of sources to create
      a - Pareto parameter
      fov - fov of sources
      '''
      def create_point_sources(self,num_sources,fov,a):
	  point_sources = np.zeros((num_sources,3),dtype=float)
	  
	  point_sources[:,0] = self.generate_flux(a = a,num_sources = num_sources,plot = False)
	  point_sources[:,1] = self.generate_pos(fov = fov,num_sources=num_sources)
	  point_sources[:,2] = self.generate_pos(fov = fov,num_sources=num_sources)
	  
	  return point_sources
	  
	
      '''
      Generates the antenna gains via sinusoidal model (mean of real part one; mean of imag part zero)
      
      RETURNS:
      g - N x t antenna gain values
      
      INPUTS:
      N - Number of antennas
      max_amp - Maximum gain amplitude
      min_amp - Minimum gain amplitude
      freq_scale - How many complete periods in timesteps
      time_steps - Number of timesteps to simulate 
      '''
      def create_antenna_gains(self,N,max_amp,min_amp,freq_scale,time_steps,plot = False):
          g = np.zeros((N,time_steps),dtype=complex)
          
          period = 1.0/time_steps
          
          freq = freq_scale*period
          
          t = np.cumsum(np.ones((time_steps,)))
          
          amp_real = np.random.uniform(low=max_amp, high=min_amp, size = N)
          phase_real = np.random.uniform(low=0, high=2*np.pi, size = N)
          
          amp_imag = np.random.uniform(low=max_amp, high=min_amp, size = N)
          phase_imag = np.random.uniform(low=0, high=2*np.pi, size = N)
          
          for t_v in xrange(time_steps):
	      g[:,t_v] = amp_real*np.cos(2*np.pi*freq*t[t_v] + phase_real) + 1 + amp_imag*np.cos(2*np.pi*freq*t[t_v] + phase_imag)*1j
          
          if plot:
	     for n in xrange(g.shape[0]):
	         plt.plot(t,g[n,:].real)
	         plt.plot(t,g[n,:].imag)
	                 
	     plt.xlabel("Time slot")
	     plt.ylabel("$g$")
	     plt.show()
          
          return g

      """
      Plot the visibilities associated with a specific baseline
      
      RETURNS:
      N/A
       
      INPUTS:
      V - Visibility matrix to plot
      baseline - Baseline to plot, i.e. V[p,q,:]
      type_plot - Which type of plot, REAL, IMAG, AMPL or PHAS
      shw - show the plot or not
      """    
      def plot_baseline(self,V,baseline,type_plot = "REAL",shw = False, flag = None):
	  
          
          v_pq = V[baseline[0],baseline[1],:]
	  t = np.cumsum(np.ones((len(v_pq),)))

          if flag is not None:
             t = t[flag]
             v_pq = v_pq[flag]

	  if type_plot == "REAL":
	     plt.plot(t,v_pq.real)
	  elif type_plot == "IMAG":
	     plt.plot(t,v_pq.imag)
	  elif type_plot == "AMPL":
	     plt.plot(t,np.absolute(v_pq))
	  elif type_plot == "PHAS":
	     plt.plot(t,np.angle(v_pq,deg=True))
	  
	  if shw:
	    plt.show()

      """
      Estimates a starting value for y
      
      RETURNS:
      y_0 - An (N-1) x t vector containing the starting value of y_0

      INPUTS:
      M - An N x N x t model visibility matrix
      sig - Amount of noise to add to M
      """  
      def create_y_0_with_M(self,M,sig):
          y_0 = np.zeros((M.shape[0]-1,M.shape[2]),dtype=complex)
          for t in xrange(M.shape[2]):
              M_temp = M[:,:,t] + sig*np.random.randn(M.shape[0],M.shape[1]) + sig*np.random.randn(M.shape[0],M.shape[1])*1j 
              for k in xrange(1,M.shape[0]):
                  y_0[k-1] = np.mean(np.diag(M_temp,k))
          return y_0

	    
if __name__ == "__main__":
   N = 6
   s = sim()
   g = s.create_antenna_gains(N=N,max_amp=1,min_amp=0.1,freq_scale=5,time_steps=600,plot = True)
   point_sources = s.create_point_sources(num_sources=500,fov=3,a=2)
   #print "max = ",np.argmax(point_sources[:,0])
   #count, bins, ignored = plt.hist(point_sources[:,0], 100, normed=True)
   #plt.show()
   u_m, v_m, w_m, b_m = s.create_uv_regular_line(N=N)
   D = s.create_vis_mat(point_sources=point_sources,u_m=u_m,v_m=v_m,g=g,sig=0.01,w_m = None)
   M = s.create_vis_mat(point_sources=point_sources,u_m=u_m,v_m=v_m,g=None,sig=0,w_m = None)
   s.plot_baseline(D,[0,1],shw=False)
   s.plot_baseline(M,[0,1],shw=True)
   #plt.plot(point_sources[:,1]*(180/np.pi),point_sources[:,2]*(180/np.pi),"bx")
   #plt.show()
   #s.generate_flux(a = 10,num_sources = 1000,plot = True)
   #point_sources = 
   
   
   
	    
	    


 
