import numpy as np
import pylab as plt
import matplotlib as mpl
#from matplotlib import cm.get_cmap

'''
Simulator that can create uv-tracks and basic scalar visibilities.
'''

class sim():

      '''
      Constructor
      
      RETURNS:
      None
      
      INPUTS:
      h_min - Minimum hour angle of the observation
      h_max - Maximum hour angle of the observation
      dec - Declination of observation
      lat - Latitude of array
            #lat = (-30 - 43.0/60.0 - 17.34/3600)*(np.pi/180) #KAT7
            #lat = (52 + 55/60.0 + 0.09/3600)*(np.pi/180) #WSRT   
            #lat = (38. + 59./60. + 48./3600.)*(np.pi/180) #College Park
      min_amp - Minimum gain amplitude
      freq - Frequency of observation in Hz
             #freq = 1.4*10**9 1.4 GHz
      layout - Specify the name of the antenna layout file 
             # layout = "HEX" - generate a HEX layout instead of reading from file using bas_len and order (number of rings)
             # layout = "REG" - generate a regular line instead of reading from file using bas_len and order (number of antennas)
             # layout = "SQR" - generate a square layout reading from file using bas_len and order (side length of square)   
      nsteps - Number of timesteps
      bas_len - Fundamental baseline length used to create HEX, SQR and REG layouts
      order - Fundamental parameter which determines the layout size
      '''
      
      def __init__(self,h_min=-1,h_max=1,dec=(-74-39./60-37.481)*(np.pi/180),lat=(-30 - 43.0/60.0 - 17.34/3600)*(np.pi/180),freq=1.4*10**9,layout="HEX",nsteps=600,bas_len=50,order=1,seed=1):
          self.h_min = h_min
          self.h_max = h_max
          self.nsteps = nsteps
          self.h = np.linspace(self.h_min,self.h_max,num=nsteps)*np.pi/12 #Hour angle range
          self.dec = dec
          self.lat = lat
          self.freq = freq
          self.wave = (3*10**8)/self.freq
          self.layout = layout
          self.bas_len = bas_len
          self.order = order

          self.ant = np.array([]) #ANTENNA POSTIONS (N,3) - ENU
          self.N = 0 #NUMEBR OF ANTENNAS
          self.B = 0 #NUMBER OF BASELINES
          self.L = 0 #NUMBER OF REDNUNDANT BASELINES
          self.u_m = None #The uvw composite index matrices
          self.v_m = None
          self.w_m = None

          self.u_f = None #The uvw coordinates associated with a single frequency channel (NB --- Gianni's experiment)
          self.v_f = None
          self.w_f = None

          self.phi = np.array([])
          self.zeta = np.array([])
          if seed is not None:
             np.random.seed(seed)

      '''
      ##############################################################################################################################
      ANTENNA LAYOUT AND UV-COVERAGE       
      ##############################################################################################################################
      '''

      '''
      Generates an hexagonal layout
      
      RETURNS:
      None

      INPUTS:
      hex_dim - The amount of rings in the hexagonal layout
      l - Basic spacing between antennas
      ''' 
      def hex_grid(self,hex_dim,l):
          hex_dim = int(hex_dim)
          side = int(hex_dim + 1)
          ant_main_row = int(side + hex_dim)
        
          elements = 1

          #SUMMING ANTENNAS IN HEXAGONAL RINGS 
          for k in xrange(hex_dim):
              elements = elements + (k+1)*6
          
          #CREATING HEXAGONAL LAYOUT STARTING FROM CENTER      
          ant_x = np.zeros((elements,),dtype=float)
          ant_y = np.zeros((elements,),dtype=float)
          #print "len(ant_x) = ",len(ant_x)
          #print "len(ant_y) = ",len(ant_y)
          x = 0.0
          y = 0.0

          counter = 0
        
          for k in xrange(side):
              x_row = x
              y_row = y
              for i in xrange(ant_main_row):
                  if k == 0:
                     ant_x[counter] = x_row 
                     ant_y[counter] = y_row
                     x_row = x_row + l
                     counter = counter + 1 
                  else:
                     ant_x[counter] = x_row
                     ant_y[counter] = y_row
                     counter = counter + 1
                   
                     ant_x[counter] = x_row
                     ant_y[counter] = -1*y_row
                     x_row = x_row + l
                     counter = counter + 1   
              x = x + l/2.0
              y = y + (np.sqrt(3)/2.0)*l                 
              ant_main_row = ant_main_row - 1
       
          #RESORTING ANTENNA INDICES SO THAT LOWER LEFT CORNER BECOMES THE FIRST ANTENNA
          y_idx = np.argsort(ant_y)
          ant_y = ant_y[y_idx]
          ant_x = ant_x[y_idx]

          slice_value = int(side)
          start_index = 0
          add = True
          ant_main_row = int(side + hex_dim)

          for k in xrange(ant_main_row):
              temp_vec_x = ant_x[start_index:start_index+slice_value]
              x_idx = np.argsort(temp_vec_x)
              temp_vec_x = temp_vec_x[x_idx]
              ant_x[start_index:start_index+slice_value] = temp_vec_x
              if slice_value == ant_main_row:
                 add = False
              start_index = start_index+slice_value 
            
              if add:
                 slice_value = slice_value + 1
              else:
                 slice_value = slice_value - 1  

              #print "slice_value = ",slice_value
              #print "k = ",k

          #SHIFTING CENTER OF ARRAY TO ORIGIN
          max_x = np.amax(ant_x)
          #max_y = np.amax(ant_y)
          
          ant_x = ant_x - max_x/2.0
          #ant_y = ant_y - max_y/2.0

          temp_ant = np.zeros((len(ant_x),3),dtype=float)
          temp_ant[:,0] = ant_x
          temp_ant[:,1] = ant_y
          return temp_ant

      '''
      Generates a square layout
      
      RETURNS:
      None

      INPUTS:
      side - The amount of antennas in one side
      l - Basic spacing between antennas
      '''       
      def square_grid(self,side,l):
          elements = side*side
                
          ant_x = np.zeros((elements,),dtype=float)
          ant_y = np.zeros((elements,),dtype=float)
          #print "len(ant_x) = ",len(ant_x)
          #print "len(ant_y) = ",len(ant_y)
          x = 0.0
          y = 0.0

          counter = 0
        
          for k in xrange(side):
              x = 0.0
              for i in xrange(side):
                  ant_x[counter] = x
                  ant_y[counter] = y 
                  x = x + l
                  counter = counter + 1
              y = y + l
 
          #SHIFTING CENTER OF ARRAY TO ORIGIN
          max_x = np.amax(ant_x)
          max_y = np.amax(ant_y)
          
          ant_x = ant_x - max_x/2.0
          ant_y = ant_y - max_y/2.0
          
          temp_ant = np.zeros((len(ant_x),3),dtype=float)
          temp_ant[:,0] = ant_x
          temp_ant[:,1] = ant_y
          return temp_ant

      '''
      Generates a regular EW layout
      
      RETURNS:
      None

      INPUTS:
      side - The amount of antennas in one side
      l - Basic spacing between antennas
      '''              
      def line_grid(self,elements,l):
                       
          ant_x = np.zeros((elements,),dtype=float)
          ant_y = np.zeros((elements,),dtype=float)
          #print "len(ant_x) = ",len(ant_x)
          #print "len(ant_y) = ",len(ant_y)
          x = 0.0
          y = 0.0

          for k in xrange(elements):
              ant_x[k] = x
              x = x + l

          temp_ant = np.zeros((len(ant_x),3),dtype=float)
          temp_ant[:,0] = ant_x
          temp_ant[:,1] = ant_y
          return temp_ant


      '''
      Generates an antenna layout either from a file or with a standard grid depending on the value of self.layout
      
      RETURNS:
      None

      INPUTS:
      None
      '''
      def generate_antenna_layout(self):
          if self.layout == "HEX":
             self.ant = self.hex_grid(hex_dim=self.order,l=self.bas_len)
          elif self.layout == "SQR":
             self.ant = self.square_grid(side=self.order,l=self.bas_len) 
          elif self.layout == "REG":
             self.ant = self.line_grid(elements=self.order,l=self.bas_len)
          else:
             self.read_antenna_layout()
          self.N = self.ant.shape[0]

      '''
      Reads the antenna layout from a text file
      
      RETURNS:
      None

      INPUTS:
      None
      ''' 
      def read_antenna_layout(self):
          file = open(self.layout, 'r')
          num_ant = file.read().count("\n")
          file.close()
          file = open(self.layout, 'r')
          array_l = np.zeros((num_ant,3))#N E Up (m)
          counter = 0
          for line in file:
              #print "line = ",line
              val = line.split(" ")
              val[-1] = val[-1].split("\n")[0]
              array_l[counter,0] = float(val[0])
              array_l[counter,1] = float(val[1])
              if len(val) > 2 :
                 array_l[counter,2] = float(val[2])
              counter = counter + 1
          file.close()
          #print "array_l = ",array_l
          self.ant = array_l

      '''
      Plots the antenna layout

      RETURNS:
      None

      INPUTS:
      title - title of plot
      labelsize - font size of labels
      ms - marker size
      '''
      def plot_ant(self,title="KAT",label_size=14,ms=10):
          m = np.amax(np.absolute(self.ant))
          mpl.rcParams['xtick.labelsize'] = label_size 
          mpl.rcParams['ytick.labelsize'] = label_size 
          plt.plot(self.ant[:,0],self.ant[:,1],'bo',ms=ms)
          #labels = ["0","1","2","3","4","5","6"]
          #for label, x, y in zip(labels, array_l[:, 0], array_l[:, 1]):
              #plt.annotate(label, xy = (x, y), xytext = (-15, 15), textcoords = 'offset points', fontsize = 12, ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.2), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')) 
          plt.xlabel("W-E [m]",fontsize=label_size)
          plt.ylabel("S-N [m]",fontsize=label_size)
          plt.title(title+"-"+str(self.N),fontsize=label_size)
          plt.axis("equal")
          plt.axis([-1.1*m, 1.1*m,-1.1*m, 1.1*m])
          plt.show() 

      '''
      Converts baseline length, azimuth angle, elevation angle and latitude into XYZ coordinates
      RETURNS:
      XYZ - a vector of size 3 containing the XYZ coordinate of a specific baseline 
            #X - (0,0)
            #Y - (-6,0)
            #Z - NCP
      INPUTS:
      d - baseline length
      az - azimuth angle of baseline
      elev - elevation angle of baseline
      lat - latitude of array
      '''
      def DAE_to_XYZ(self,d,az,elev,lat):
          XYZ = d * np.array([np.cos(lat)*np.sin(elev) - np.sin(lat)*np.cos(elev)*np.cos(az),
          np.cos(elev)*np.sin(az),
          np.sin(lat)*np.sin(elev) + np.cos(lat)*np.cos(elev)*np.cos(az)])
          #print "XYZ = ",XYZ
          return XYZ        


      '''
      Computes the rotation matrix A needed to convert XYZ into uvw at a specific hour angle
        
      INPUTS:
      h - a specific hour angle
      delta - declination of observation

      RETURNs:
      A - rotation matrix capable of converting XYZ into uvw
      '''
      def XYZ_to_uvw(self,h,delta):
          A = np.array([[np.sin(h),np.cos(h),0],[-1*np.sin(delta)*np.cos(h),np.sin(delta)*np.sin(h),np.cos(delta)],[np.cos(delta)*np.cos(h),-1*np.cos(delta)*np.sin(h),np.sin(delta)]])
          return A   

      ''' 
      Computes the uv-track of a specific baseline
      
      RETURNS:
      uvw - The uv-track of a specific baseline over a certain hour angle range
      
      INPUTS:
      h - hour angle range
      dec - declination of observation
      lat - latitude of observation
      az - azimuth angle of baseline
      el - elevation angle of baseline
      d - baseline lenght
      wave_lenght - wavelenght of observation
      '''
      def uv_track(self,h,d,az,el,lat,dec,wave_length):
          uvw = np.zeros((len(h),3))
          for i in xrange(len(h)):
              A = self.XYZ_to_uvw(h[i],dec)
              uvw[i,:] = A.dot(self.DAE_to_XYZ(d,az,el,lat)/wave_length)
          return uvw 
      

      ''' 
      Computes the uv-tracks of an array layout
      RETURNS:
      None

      INPUTS:
      None
      ''' 
      def uv_tracks(self):
          #CALCULATE BASELINE LENGTHS AND AZIMUTH AND ELEVATION ANGLES
          self.N = self.ant.shape[0]
          self.B = (self.N**2-self.N)/2

          DAE = np.zeros((self.B,3))#storing baseline length, azimuth angle and elevation
          
          p = np.zeros((self.B,),dtype=int) 
          q = np.zeros((self.B,),dtype=int)  

          k = 0
          for i in xrange(self.N):
              for j in xrange(i+1,self.N):
                  DAE[k,0] = np.sqrt((self.ant[i,0]-self.ant[j,0])**2+(self.ant[i,1]-self.ant[j,1])**2+(self.ant[i,2]-self.ant[j,2])**2)
                  #WRONG VERSION: DAE[k,1] = np.arctan2(array_l[i,0]-array_l[j,0],array_l[i,1]-array_l[j,1])
                  #NB MUST USE j - i
                  DAE[k,1] = np.arctan2(self.ant[j,0]-self.ant[i,0],self.ant[j,1]-self.ant[i,1])
                  DAE[k,2] = np.arcsin((self.ant[j,2]-self.ant[i,2])/DAE[k,0])
                  p[k] = i
                  q[k] = j
                  k = k + 1 
   
          #CONVERT TO XYZ
          #CONVERT TO UVW
          self.u_m = np.zeros((self.N,self.N,self.nsteps),dtype=float) 
          self.v_m = np.zeros((self.N,self.N,self.nsteps),dtype=float)
          self.w_m = np.zeros((self.N,self.N,self.nsteps),dtype=float)    
   
          
          for i in xrange(self.B):
              X = self.uv_track(self.h,DAE[i,0],DAE[i,1],DAE[i,2],self.lat,self.dec,self.wave)
              self.u_m[p[i],q[i],:] = X[:,0]
              self.u_m[q[i],p[i],:] = -1*X[:,0]
              self.v_m[p[i],q[i],:] = X[:,1]
              self.v_m[q[i],p[i],:] = -1*X[:,1] 
              self.w_m[p[i],q[i],:] = X[:,2]
              self.w_m[q[i],p[i],:] = -1*X[:,2]
      

      ''' 
      Plots the uv-tracks of an array layout
      RETURNS:
      None

      INPUTS:
      title - title of plot
      labelsize - font size of labels
      None
      ''' 
      def plot_uv_coverage(self,title="KAT",label_size=14):
          m_x = np.amax(np.absolute(self.u_m))
          m_y = np.amax(np.absolute(self.v_m)) 
          for i in xrange(self.N):
              for j in xrange(i+1,self.N):
                  plt.plot(self.u_m[i,j,:],self.v_m[i,j,:],"b")  
                  plt.plot(self.u_m[j,i,:],self.v_m[j,i,:],"r")
          plt.ylim(-1.1*m_y,1.1*m_y)
          plt.xlim(-1.1*m_x,1.1*m_x)
          plt.xlabel("$u$ [$\lambda$]",fontsize=label_size)
          plt.ylabel("$v$ [$\lambda$]",fontsize=label_size)
          plt.title(title+"-"+str(self.N),fontsize=label_size)
          plt.show()

      '''
      ##############################################################################################################################
      VISIBILITIES AND NOISE       
      ##############################################################################################################################
      '''
     
      '''
      Determines the power in signal
      INPUTS:
      D - Matrix to calculate the power of
      RETURNS:
      D_pow - The power in each baseline
      d_pow1 - Average power over entire matrix
      d_pow2 - Average of D_pow over baselines
      '''
      def det_power_of_signal(self,D):
          D = np.copy(D)
          for t in xrange(D.shape[2]):
              D[:,:,t] = D[:,:,t]*(np.ones((D.shape[0],D.shape[1]),dtype=float)-np.diag(np.ones((D.shape[0],),dtype=float)))
	  bas = (D.shape[0]**2 - D.shape[0])
	  D_pow = np.mean(np.absolute(D)**2,axis=2)
	  d_pow1 = np.sum(np.absolute(D)**2)/(bas*D.shape[2])
	  d_pow2 = np.sum(D_pow)/bas
	  return D_pow,d_pow1,d_pow2
      
      '''
      Function that generates noise at a certain power level
      http://dsp.stackexchange.com/questions/16216/adding-white-noise-to-complex-signal-complex-envelope
      INPUTS:
      power - power of the noise to generate
      '''
      def generate_noise(self,power,third_dim=None):
	  sig = np.sqrt(power/2)
          #print "sig = ",sig
          if third_dim is None:
             third_dim = self.nsteps  
	  mat = np.zeros((self.N,self.N,third_dim),dtype=complex)
	  for i in xrange(self.N):
              for j in xrange(i+1,self.N):
	          mat[i,j,:] = sig*np.random.randn(third_dim)+sig*np.random.randn(third_dim)*1j
                  mat[j,i,:] = mat[i,j,:].conj()	  
	  return mat
      '''	  
      How much power in the noise is needed to achieve SNR
      INPUTS:
      P_signal - Power in the signal
      SNR - SNR to achieve
      P_noise - Power in the noise      
      '''
      def power_needed_for_SNR(self,P_signal,SNR):
          P_noise = P_signal*10**(-1*(SNR/10.))
          return P_noise
      
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
      sig - the sigma added to generate the noise      

      INPUTS:
      point_sources - list of point sources to simulate
      u_m - N x N x timeslots matrix of u coordinates
      v_m - N x N x timeslots matrix of v coordinates
      g - N x t antenna gains to corrupt the visibilities with
      SNR - the signal to noise ratio in dB (gains are assumed to be signal)
      w_m - N x N x timeslots matrix of w coordinates. If none just do 2D simulation
      '''      
      def create_vis_mat(self,point_sources,u_m,v_m,g,SNR,w_m = None):
          sig = 0
          D = np.zeros(u_m.shape,dtype=complex)
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
		D = D + point_sources[k,0]*np.exp(-2*np.pi*1j*(u_m*l_0+v_m*m_0+w_m*(n_0-1)))
    
          for t in xrange(D.shape[2]): #for each time-step
              #print "t = ",t
              if g is not None:
                 G = np.diag(g[:,t])
                 #Step 2: Corrupting the Visibilities 
                 D[:,:,t] = np.dot(G,D[:,:,t])
                 D[:,:,t] = np.dot(D[:,:,t],G.conj()) 
              #OLD NOISE STEP
              #D[:,:,t] = D[:,:,t] + sig*np.random.randn(u_m.shape[0],u_m.shape[1]) + sig*np.random.randn(u_m.shape[0],u_m.shape[1])*1j
                       
          #Step 3: Adding Noise
          if SNR is not None:
             D_pow,P_signal,pow2 = self.det_power_of_signal(D)
             #print "D_pow = ",D_pow
             #print "P_signal = ",P_signal
             #print "pow2 = ",pow2
             P_noise = self.power_needed_for_SNR(P_signal,SNR)
             #print "P_noise = ",P_noise
             #print "SNR = ",10*np.log10(P_signal/P_noise)
             N = self.generate_noise(P_noise,third_dim=D.shape[2])
             D = D + N 
             sig = (np.sqrt(P_noise/2))

          return D,sig

      '''
      Generates the antenna gains via sinusoidal model
      
      RETURNS:
      g - N x t antenna gain values
      
      INPUTS:
      N - Number of antennas
      max_a - Maximum gain amplitude
      min_a - Minimum gain amplitude
      max_p - Max phase
      min_p - Min phase
      freq_scale - How many complete periods in timesteps
      time_steps - Number of timesteps to simulate 
      '''
      def create_antenna_gains(self,N,max_a,min_a,max_p,min_p,freq_scale,time_steps,plot = False):
          g = np.zeros((N,time_steps),dtype=complex)
          
          period = 1.0/time_steps
          
          freq = freq_scale*period
          
          t = np.cumsum(np.ones((time_steps,)))
          
          amp = np.random.uniform(low=min_a, high=max_a, size = N)
          amp_phase = np.random.uniform(low=0, high=2*np.pi, size = N)
                   
          pha = np.random.uniform(low=(min_p/2.)*(np.pi/180), high=(max_p/2.)*(np.pi/180), size = N) 
          phase_phase = np.random.uniform(low=0, high=2*np.pi, size = N)
          #print "pha = ",pha*(180/np.pi)

          for t_v in xrange(time_steps):
	      g[:,t_v] = (amp*np.cos(2*np.pi*freq*t[t_v] + amp_phase)+1)*np.exp(1j*(pha*np.cos(2*np.pi*freq*t[t_v] + phase_phase)+pha))
          
          if plot:
	     for n in xrange(g.shape[0]):
	         plt.plot(t,np.absolute(g[n,:]))
	         	                 
	     plt.xlabel("Time slot")
	     plt.ylabel("$|g|$")
	     plt.show()

             for n in xrange(g.shape[0]):
	         plt.plot(t,np.angle(g[n,:])*(180/np.pi))
	                 
	     plt.xlabel("Time slot")
	     plt.ylabel("ph$(g)$")
	     plt.show()
          
          return g 

      '''
      Plot the real visibilities
      INPUTS:
      b - baseline
      D - visibility matrix
      c - colout
      s - show
      ''' 
      def plot_visibilities(self,b,D,c,s=False):
          t = np.ones((D.shape[2],),dtype=float)
          t = np.cumsum(t)
          plt.plot(t,D[b[0],b[1],:].real,c)
          if s:
             plt.xlabel("Timeslots")
             plt.ylabel("Jy")
             plt.show()    
              
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
      ##############################################################################################################################
      GIANNI - Freq experiment       
      ##############################################################################################################################
      '''
      
      ''' 
      Computes the uv-points of a single timeslot along the freq dimension
      RETURNS:
      None

      INPUTS:
      timeslot - timeslot for which to calculate freq dim
      num_channels - number of channels to create
      f0 - starting freq
      f1 - ending freq
      ''' 
      def create_uv_f(self,timeslot=1,num_channels=1024,f0=100e6,f1=200e6):
          u = self.u_m[:,:,timeslot]*self.wave 
          v = self.v_m[:,:,timeslot]*self.wave
          w = self.w_m[:,:,timeslot]*self.wave
          
          print "u = ",u
          print "v = ",v

          lam_vec = 3e8*(np.linspace(f0,f1,num=num_channels,endpoint=True,dtype=float)**(-1))
          
          print "lam_vec = ",lam_vec
          

          self.u_f = np.zeros((u.shape[0],u.shape[1],num_channels))
          self.v_f = np.zeros((v.shape[0],v.shape[1],num_channels))
          self.w_f = np.zeros((w.shape[0],w.shape[1],num_channels)) 

          for l in xrange(num_channels):
              self.u_f[:,:,l] = u/lam_vec[l]
              self.v_f[:,:,l] = v/lam_vec[l]
              self.w_f[:,:,l] = w/lam_vec[l]
          print "u_f = ",self.u_f[:,:,0]
       
      '''
      Plots the uv-coverage as a func of freq 
      '''
      def plot_uv_f(self):
          print "u_f = ",self.u_f[:,:,0]
          for p in xrange(self.u_f.shape[0]):
              for q in xrange(p+1,self.u_f.shape[1]):
                  plt.plot(self.u_f[p,q,:],self.v_f[p,q,:],'r.',ms=0.05)
                  #plt.hold()
                  #plt.plot(-1*self.u_f[p,q,:],-1*self.v_f[p,q,:],'b.',ms=0.1)
                  
          plt.show()


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
      def create_antenna_gains_old(self,N,max_amp,min_amp,freq_scale,time_steps,plot = False):
          g = np.zeros((N,time_steps),dtype=complex)
          
          period = 1.0/time_steps
          
          freq = freq_scale*period
          
          t = np.cumsum(np.ones((time_steps,)))
          
          amp_real = np.random.uniform(low=max_amp, high=min_amp, size = N)
          phase_real = np.random.uniform(low=0, high=2*np.pi, size = N)
          
          amp_imag = np.random.uniform(low=max_amp, high=min_amp, size = N)
          phase_imag = np.random.uniform(low=0, high=2*np.pi, size = N)
          
          for t_v in xrange(time_steps):
	      g[:,t_v] = amp_real*np.cos(2*np.pi*freq*t[t_v] + phase_real) + 1.5*amp_real + amp_imag*np.sin(2*np.pi*freq*t[t_v] + phase_imag)*1j
          
          if plot:
	     for n in xrange(g.shape[0]):
	         plt.plot(t,g[n,:].real)
	         plt.plot(t,g[n,:].imag)
	                 
	     plt.xlabel("Time slot")
	     plt.ylabel("$g$")
	     plt.show()
          
          return g


      def generate_phase_slope_gains(self,num_channels=1024,f0=100e6,f1=200e6,l0=5,l1=50):
          phase = np.zeros((self.N,num_channels),dtype=float)

          f = np.linspace(f0,f1,num=num_channels,endpoint=True,dtype=float)

          tau = np.random.uniform(l0,l1,self.N)/3e8

          print "tau = ",tau

          phi = np.zeros((self.N,num_channels),dtype=float)
          
          for n in xrange(self.N):
              #phi[n,:] = 2*np.pi*tau[n]*f
              phi[n,:] = tau[n]*f

          for n in xrange(phi.shape[0]):
              plt.plot(f,phi[n,:])

          plt.show()

          g = np.exp(1j*phi)

          for n in xrange(g.shape[0]):
              plt.plot(f,np.angle(g[n,:]))

          plt.show()  

          for n in xrange(g.shape[0]):
              plt.plot(f,np.angle(np.conjugate(g[0,:])*g[n,:]))

          plt.show()

          return g

      '''
      ##############################################################################################################################
      PHI       
      ##############################################################################################################################
      '''

      '''
      Converts the antenna idices pq into a redundant index
         
      INPUTS:
      red_vec_x - the current list of unique redundant baseline vector (x-coordinate)
      red_vec_y - the current list of unique redundant baseline vector (y-coordinate)
      ant_x_p - the x coordinate of antenna p
      ant_x_q - the x coordinate of antenna q
      ant_y_p - the y coordinate of antenna p
      ant_y_q - the y coordinate of antenna q

      RETURNS:
      red_vec_x - the current list of unique redundant baseline vector (x-coordinate)
      red_vec_y - the current list of unique redundant baseline vector (y-coordinate)
      l - The redundant index associated with antenna p and q
      '''
      def determine_phi_value(self,red_vec_x,red_vec_y,ant_x_p,ant_x_q,ant_y_p,ant_y_q):
          red_x = ant_x_q - ant_x_p
          red_y = ant_y_q - ant_y_p

          for l in xrange(len(red_vec_x)):
              if (np.allclose(red_x,red_vec_x[l]) and np.allclose(red_y,red_vec_y[l])):
                 return red_vec_x,red_vec_y,int(l+1)

          red_vec_x = np.append(red_vec_x,np.array([red_x]))
          red_vec_y = np.append(red_vec_y,np.array([red_y]))
          return red_vec_x,red_vec_y,int(len(red_vec_x)) 

      '''
      Returns the mapping phi, from pq indices to redundant indices.
      INPUTS:
      ant_x - vector containing the x-positions of all the antennas
      ant_y - vector containing the y-positions of all the antennas 

      RETURNS:
      phi - the mapping from pq indices to redundant indices
      zeta - the symmetrical counterpart  
      '''
      def calculate_phi(self,ant_x,ant_y):
          phi = np.zeros((len(ant_x),len(ant_y)),dtype=int)
          zeta = np.zeros((len(ant_x),len(ant_y)),dtype=int)
          red_vec_x = np.array([])
          red_vec_y = np.array([])
          for k in xrange(len(ant_x)):
              for j in xrange(k+1,len(ant_x)):
                  red_vec_x,red_vec_y,phi[k,j]  = self.determine_phi_value(red_vec_x,red_vec_y,ant_x[k],ant_x[j],ant_y[k],ant_y[j])           
                  zeta[k,j] = phi[k,j]
                  zeta[j,k] = zeta[k,j]
          self.L = np.amax(zeta)
          self.phi = phi
          self.zeta = zeta   
          return phi,zeta

      '''
      Plot zeta
      INPUTS:
      data - redundant matrix to plot
      step1 - step size on colorbar
      step2 - step size on axis
      label_size - label size
      cs - color scheme
      
      RETURNS:
      None 
      '''
      def plot_zeta(self,data,step1,step2,label_size,cs):
          mpl.rcParams['xtick.labelsize'] = label_size 
          mpl.rcParams['ytick.labelsize'] = label_size 
          #get discrete colormap
          cmap = plt.get_cmap(cs, np.max(data)-np.min(data)+1)
          # set limits .5 outside true range
          mat = plt.matshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
          #tell the colorbar to tick at integers
          ticks = ticks=np.arange(np.min(data),np.max(data)+1,step1)
          if ticks[-1] <> np.max(data):
             ticks = np.append(ticks,np.array([np.max(data)])) 
          cax = plt.colorbar(mat, ticks=ticks)
          x = np.arange(len(self.ant[:,0]),step=step2,dtype=int)
          if x[-1] <> len(self.ant[:,0])-1:
             x = np.append(x,np.array([len(self.ant[:,0])-1]))
          plt.xticks(x, x+1)
          y = np.arange(len(self.ant[:,1]),step=step2,dtype=int)
          if y[-1] <> len(self.ant[:,0])-1:
             y = np.append(y,np.array([len(self.ant[:,0])-1]))
          plt.yticks(y, y+1)
          plt.xlabel("$q$",fontsize=label_size+5)
          plt.ylabel("$p$",fontsize=label_size+5)
          #plt.title("$\zeta_{pq}$",fontsize=label_size+5)
          cax.set_label("$R(\zeta_{pq})$", size=label_size+5)
          plt.show()

      '''
      Returns the q if the redundant index and p are given
      INPUTS:
      i - p
      j - redundant index
      phi - mapping from pq to redundant index
      
      RETURNS:
      q - q
      Found - If q was found in phi?
      '''
      def psi_func_eval(self,i,j,phi):
          row = phi[i,:]

          for q in xrange(len(row)):
              if row[q] == j:
                 return q,True

          return 0,False

      '''
      Returns the p if the redundant index and q are given
      INPUTS:
      i - q
      j - redundant index
      phi - mapping from pq to redundant index
      
      RETURNS:
      p - p
      Found - If p was found in phi?
      '''
      def xi_func_eval(self,i,j,phi):
          column = phi[:,i]

          #print "column = ",column
          #print "j = ",j

          for p in xrange(len(column)):
              if column[p] == j:
                 return p,True

          return 0,False


      '''
      Returns the the dictionary PQ containing all the pq index sets associated with each redundant spacing
      INPUTS:
      phi - mapping from pq to redundant index
      L - maximum entry in phi

      RETURNS:
      PQ - dictionary of pq index sets associated with each redundant spacing      
     '''
      def create_PQ(self,phi,L):
          PQ = {}
          for i in xrange(L):
              pq = zip(*np.where(phi == (i+1)))
              #print "pq2 = ",pq        
              PQ[str(i)]=pq
              #print "PQ = ",PQ    
          return PQ

      '''
      Takes the y-vector and constructs an NxN matrix M

      INPUTS:
      PQ - dictionary of pq index sets associated with each redundant spacing 
      y - vector from which M is constructed
      N - number of antennas

      OUTPUTS:
      M - NxN matrix from y
      '''
      def convert_y_to_M(self,PQ,y,N):
    
          M = np.zeros((N,N),dtype=complex)
          for i in xrange(len(y)):
              pq = PQ[str(i)]
              #print "pq = ",pq
                          
              for k in xrange(len(pq)):
                  p = pq[k][0]
                  q = pq[k][1]
             
                  M[p,q] = y[i]
                  M[q,p] = np.conjugate(y[i])  
                  #from IPython import embed; embed() 
          return M
          
"""
Function to verify the analytic expression for calculating the number of redundancies from the number of antennas in an HEXAGONAL 
layout

INPUTS:
min_v - the minumum number of rings in the layout
max_v - the maximum number of rings in the layout
"""
def func_N_to_L_hex(min_v=1,max_v=6):
    r_v  = np.arange(min_v,max_v)

    N = np.zeros((len(r_v),))
    L = np.zeros((len(r_v),))
   
    #GENERATES L NUMERICALLY 
    for k in xrange(len(r_v)): 
        s = sim(layout="HEX",order=r_v[k])
        s.generate_antenna_layout()
        phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
        N[k] = s.N
        L[k] = s.L
    #array([  7,  19,  37,  61,  91, 127, 169, 217, 271, 331, 397, 469, 547])
    #z = np.polyfit(r_v, L, 2) #FITTING A POLYNOMIAL TO DATA    
    #r_t = (-3 + np.sqrt(12*N-3))/6 #LINK BETWEEN AMOUNT OF RINGS AND NUMBER OF ANTENNAS
    #L_t = 6*r_v**2 + 3*r_v #ANALYTIC LINK BETWEEN NUMBER OF RINGS AND REDUNDANT BASELINES
    plt.plot(N,L)
    plt.plot(N,2*N-0.5*np.sqrt(12*N-3)-0.5,"r")#ANALYTIC LINK BETWEEN NUMBER OF RINGS AND BASELINES
    plt.xlabel("$N$")
    plt.ylabel("$L$")
    plt.show()
    
"""
Function to verify the analytic expression for calculating the number of redundancies from the number of antennas in an HEXAGONAL 
layout

INPUTS:
min_v - the minumum number of side
max_v - the maximum number of side
"""
def func_N_to_L_SQR(min_v=2,max_v=6):
    r_v  = np.arange(min_v,max_v)

    N = np.zeros((len(r_v),))
    L = np.zeros((len(r_v),))
   
    #GENERATES L NUMERICALLY 
    for k in xrange(len(r_v)): 
        s = sim(layout="SQR",order=r_v[k])
        s.generate_antenna_layout()
        phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
        N[k] = s.N
        L[k] = s.L
        
    #z = np.polyfit(r_v, L, 2) #FITTING A POLYNOMIAL TO DATA  
    #r_t = np.sqrt(N) #LINK BETWEEN SIDE AND NUMBER OF ANTENNAS
    #L_t = 2*r_v**2 - 2*r_v #ANALYTIC LINK BETWEEN SIDE AND REDUNDANT BASELINES
    #plt.plot(r_v,L)
    #plt.plot(r_v,2*r_v**2-2*r_v,"r")
    
    plt.plot(N,L)
    plt.plot(N,2*N-2*np.sqrt(N),"r")#ANALYTIC LINK BETWEEN SIDE AND BASELINES
        
    plt.xlabel("$N$")
    plt.ylabel("$L$")
    plt.show()

'''
Function that showcases the functionality of this simulator

INPUT:
None

RETRUNS:
None
'''
def example_usage():
    s = sim() #INSTANTIATE OBJECT
    #s.read_antenna_layout()
    s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
    s.plot_ant(title="HEX") #PLOT THE LAYOUT
    s.uv_tracks() #GENERATE UV TRACKS
    s.plot_uv_coverage(title="HEX") #PLOT THE UV TRACKS
    point_sources = s.create_point_sources(100,fov=3,a=2) #GENERATE RANDOM SKYMODEL
    g=s.create_antenna_gains(s.N,0.99,0.95,2,1,5,s.nsteps,plot = True) #GENERATE GAINS
    D,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=20,w_m=None) #CREATE VIS MATRIX
    M,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=None,SNR=None,w_m=None) #PREDICTED VIS
    s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
    s.plot_visibilities([0,1],M,"r",s=True)

def plot_paper_layouts():
    s = sim(layout="HEX",order=5) #INSTANTIATE OBJECT
    s.generate_antenna_layout()
    s.plot_ant(title="HEX")
    phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
    s.plot_zeta(zeta,10,10,14,'jet') 
    print "s.L = ",s.L
    print "s.N = ",s.N

    s = sim(layout="SQR",order=10) #INSTANTIATE OBJECT
    s.generate_antenna_layout()
    s.plot_ant(title="SQR")
    phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
    s.plot_zeta(zeta,10,10,14,'jet')
    print "s.L = ",s.L
    print "s.N = ",s.N

    s = sim(layout="REG",order=100) #INSTANTIATE OBJECT
    s.generate_antenna_layout()
    s.plot_ant(title="REG")
    phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
    s.plot_zeta(zeta,10,10,14,'jet')
    print "s.L = ",s.L
    print "s.N = ",s.N

def Gianni_freq_exp():
    s = sim() #INSTANTIATE OBJECT
    s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
    s.plot_ant(title="HEX") #PLOT THE LAYOUT
    s.uv_tracks() #GENERATE UV TRACKS (time)
    s.create_uv_f() #GENERATE UV POINTS f (for one time-slot)
    s.plot_uv_f()
    g = s.generate_phase_slope_gains()
    point_sources = np.array([(1,0,0)])
    D,sig = s.create_vis_mat(point_sources,s.u_f,s.v_f,g=g,SNR=1000,w_m=None) #CREATE VIS MATRIX
    plt.plot(D[0,1,:].real) 
    plt.show() 

if __name__ == "__main__":
   #example_usage()
   #plot_paper_layouts()
   Gianni_freq_exp()
   

