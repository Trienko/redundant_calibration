import numpy as np
import pylab as plt

class Hgrid():

    def __init__(self):
        pass

    def hex_grid(self,hex_dim,l):
        side = hex_dim + 1
        ant_main_row = side + (side+1)
        ant_x = np.zeros((6*hex_dim+1,),dtype=float)
        ant_y = np.zeros((6*hex_dim+1,),dtype=float)
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


if __name__ == "__main__":
   h = Hgrid()
   ant_x,ant_y = h.hex_grid(1,20)
   plt.plot(ant_x,ant_y,"ro")
   plt.show()
   print "Hallo"

