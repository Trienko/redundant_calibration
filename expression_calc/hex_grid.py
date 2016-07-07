import numpy as np
import pylab as plt

class Hgrid():

    def __init__(self):
        pass

    def hex_grid(self,hex_dim,l):
        side = hex_dim + 1
        ant_main_row = side + hex_dim
        
        elements = 1

        #summing the antennas in hexactoganal rings 
        for k in xrange(hex_dim):
            elements = elements + (k+1)*6
                
        ant_x = np.zeros((elements,),dtype=float)
        ant_y = np.zeros((elements,),dtype=float)
        print "len(ant_x) = ",len(ant_x)
        print "len(ant_y) = ",len(ant_y)
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
    
        return ant_x,ant_y

    def hex_grid_ver2(self,hex_dim,l):
        hex_dim = int(hex_dim)
        side = int(hex_dim + 1)
        ant_main_row = int(side + hex_dim)
        
        elements = 1

        #summing the antennas in hexactoganal rings 
        for k in xrange(hex_dim):
            elements = elements + (k+1)*6
                
        ant_x = np.zeros((elements,),dtype=float)
        ant_y = np.zeros((elements,),dtype=float)
        print "len(ant_x) = ",len(ant_x)
        print "len(ant_y) = ",len(ant_y)
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

            print "slice_value = ",slice_value
            print "k = ",k  

        return ant_x,ant_y

    def determine_phi_value(self,red_vec_x,red_vec_y,ant_x_p,ant_x_q,ant_y_p,ant_y_q):
        red_x = ant_x_q - ant_x_p
        red_y = ant_y_q - ant_y_p

        for l in xrange(len(red_vec_x)):
            if (np.allclose(red_x,red_vec_x[l]) and np.allclose(red_y,red_vec_y[l])):
               return red_vec_x,red_vec_y,l+1

        red_vec_x = np.append(red_vec_x,np.array([red_x]))
        red_vec_y = np.append(red_vec_y,np.array([red_y]))
        return red_vec_x,red_vec_y,len(red_vec_x) 

    def calculate_phi(self,ant_x,ant_y,plot=True):
        phi = np.zeros((len(ant_x),len(ant_y)))
        red_vec_x = np.array([])
        red_vec_y = np.array([])
        for k in xrange(len(ant_x)):
            for j in xrange(k+1,len(ant_x)):
                red_vec_x,red_vec_y,phi[k,j]  = self.determine_phi_value(red_vec_x,red_vec_y,ant_x[k],ant_x[j],ant_y[k],ant_y[j])           
                phi[j,k] = phi[k,j]

        if plot:
           plt.imshow(phi,interpolation="nearest")
           x = np.arange(len(ant_x))
           plt.xticks(x, x+1)
           y = np.arange(len(ant_x))
           plt.yticks(y, y+1)
           plt.colorbar() 
           #plt.yticks(y, y+1)
           
           plt.show()

        print "phi = ",phi


if __name__ == "__main__":
   h = Hgrid()
   ant_x,ant_y = h.hex_grid_ver2(1,20)
   print "ant_x = ",ant_x
   print "ant_y = ",ant_y
   plt.plot(ant_x,ant_y,"ro")
   plt.show()
   h.calculate_phi(ant_x,ant_y)
   #print "Hallo"

