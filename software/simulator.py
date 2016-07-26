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
      NSteps - Number of timesteps
      bas_len - Fundamental baseline length used to create HEX, SQR and REG layouts
      order - Fundamental parameter which determines the layout size
      '''
      
      def __init__(self,h_min=-6,h_max=6,dec=(-74-39./60-37.481)*(np.pi/180),lat=(-30 - 43.0/60.0 - 17.34/3600)*(np.pi/180),freq=1.4*10**9,layout="HEX",nsteps=100,bas_len=50,order=19):
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
          self.phi = np.array([])
          self.zeta = np.array([])

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
          print "len(ant_x) = ",len(ant_x)
          print "len(ant_y) = ",len(ant_y)
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
          print "len(ant_x) = ",len(ant_x)
          print "len(ant_y) = ",len(ant_y)
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
    
          plt.ylim(-1.1*m,1.1*m)
          plt.xlim(-1.1*m,1.1*m)
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
          


if __name__ == "__main__":
   #s = sim()
   #s.read_antenna_layout()
   #s.generate_antenna_layout()
   #s.plot_ant(title="HEX")
   #phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   #s.plot_zeta(zeta,5,5,12,"jet")
   
   r_v  = np.arange(1,6)

   N = np.zeros((len(r_v),))
   L = np.zeros((len(r_v),))

   for k in xrange(len(r_v)):
       print "k = ",k
       s = sim(order=r_v[k])
       s.generate_antenna_layout()
       phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
       N[k] = s.N
       L[k] = s.L
 
   print "N = ",N
   print "L = ",L

   r_t = (-3 + np.sqrt(12*N-3))/6

   L_t_2 = 6*((-3 + np.sqrt(12*N-3))/6)**2+3*((-3 + np.sqrt(12*N-3))/6)
   print "r_t = ",r_t
   print "r_v = ",r_v 

   z = np.polyfit(r_v, L, 2)
   print "z = ",z

   #L_t = 2*N-np.sqrt(12*N-3)+4
   print "L_t = ",2*N-0.5*np.sqrt(12*N-3)-0.5
   print "L_t_2 = ",L_t_2
   
   plt.plot(r_v,L)
   plt.plot(r_v,6*r_v**2 + 3*r_v,"r")
   plt.show()
   plt.plot(N,L)
   plt.plot(N,2*N-0.5*np.sqrt(12*N-3)-0.5,"r")
   #print "L2 = ",z[0]*r_v**2 + z[1]*r_v+z[0]
   #plt.plot(N,z[0]*N+z[1],"g")
   #print "L2 = ",7./4*N-13./4
   #print "L3 = ",z[0]*N+z[1]
   plt.show()
   #s.uv_tracks()
   #s.plot_uv_coverage(title="SQR")
