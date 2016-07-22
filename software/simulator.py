import numpy as np
import pylab as plt
import matplotlib as mpl

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
      
      def __init__(self,h_min=-6,h_max=6,dec=(-74-39./60-37.481)*(np.pi/180),lat=(-30 - 43.0/60.0 - 17.34/3600)*(np.pi/180),freq=1.4*10**9,layout="KAT7.txt",nsteps=300,bas_len=None,order=None):
          self.h_min = h_min
          self.h_max = h_max
          self.nsteps = nsteps
          self.h = np.linspace(self.h_min,self.h_max,num=nsteps)*np.pi/12
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
      def plot_ant(self,title="KAT7",label_size=14,ms=10):
          m = np.amax(np.absolute(self.ant))
          mpl.rcParams['xtick.labelsize'] = label_size 
          mpl.rcParams['ytick.labelsize'] = label_size 
          plt.plot(self.ant[:,0],self.ant[:,1],'bo',ms=ms)
          #labels = ["0","1","2","3","4","5","6"]
          #for label, x, y in zip(labels, array_l[:, 0], array_l[:, 1]):
              #plt.annotate(label, xy = (x, y), xytext = (-15, 15), textcoords = 'offset points', fontsize = 12, ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.2), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')) 
          plt.xlabel("W-E [m]",fontsize=label_size)
          plt.ylabel("S-N [m]",fontsize=label_size)
          plt.title(title,fontsize=label_size)
    
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
      def DAE_to_XYZ(d,az,elev,lat):
          XYZ = d * np.array([np.cos(lat)*np.sin(elev) - np.sin(lat)*np.cos(elev)*np.cos(az),
          np.cos(elev)*np.sin(az),
          np.sin(lat)*np.sin(elev) + np.cos(lat)*np.cos(elev)*np.cos(az)])
          #print "XYZ = ",XYZ
          return XYZ        


      '''
      Converts XYZ into uvw coordinates
      
      '''
      def XYZ_to_uvw(h,delta):
          A = np.array([[np.sin(h),np.cos(h),0],[-1*np.sin(delta)*np.cos(h),np.sin(delta)*np.sin(h),np.cos(delta)],[np.cos(delta)*np.cos(h),-  np.cos(delta)*np.sin(h),np.sin(delta)]])
          return A   

def uvTrack(h,d,az,el,lat,dec,wave_length):
    uvw = np.zeros((len(h),3))
    for i in xrange(len(h)):
        A = XYZ_to_uvw(h[i],dec)
        uvw[i,:] = A.dot(DAE_to_XYZ(d,az,el,lat)/wave_length)
    return uvw  


if __name__ == "__main__":
   s = sim()
   s.read_antenna_layout()
   s.plot_ant()
