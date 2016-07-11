import numpy as np
import pylab as plt
import pickle
from scipy import optimize
import matplotlib as mpl
from copy import deepcopy
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering


class factor():

    def __init__(self,
                type_factor = "g",
                index = 1, 
                exponent = 1,
                conjugate_value = False,
                mag = False,
                ant_p = 0,
                ant_q = 0,
                print_f = True):
                self.type_factor = type_factor
                self.index = index
                self.exponent = exponent
                self.conjugate_value = conjugate_value
                self.mag = mag
                self.ant_p = ant_p
                self.ant_q = ant_q
                self.print_f = print_f 
             

    def to_string(self):
        if self.print_f:
           if self.type_factor == "y":
              string_out = self.type_factor + "_{" + str(self.ant_p)+str(self.ant_q)+"}"
           else:
              string_out = self.type_factor + "_" + str(self.index)
        else:   
           string_out = self.type_factor + "_" + str(self.index)
        if self.conjugate_value:
           string_out = string_out + "^" + "*"
           if self.exponent > 1:
              string_out = "("+string_out+")"+"^"+str(self.exponent)
        else:
           if self.mag:
              string_out = "|"+string_out+"|^2"
           else:  
              if self.exponent > 1:
                 string_out = string_out+"^"+str(self.exponent)
        return string_out   
           
    def conjugate(self):
        if self.conjugate_value:
           self.conjugate_value = False
        else:
           self.conjugate_value = True

    def equal(self,factor_in):
        if self.type_factor == factor_in.type_factor:
           if self.index == factor_in.index:
              if self.exponent == factor_in.exponent:
                 if self.conjugate_value == factor_in.conjugate_value:
                    return true
        return false

class term():
      def __init__(self):
          self.g_array = np.array([],dtype=object)
          self.y_array = np.array([],dtype=object)
          self.yc_array = np.array([],dtype=object) 
          self.gc_array = np.array([],dtype=object)
          self.ga_array = np.array([],dtype=object)
          self.ya_array = np.array([],dtype=object)
          self.zero = False

      def substitute(self,g_v,y_v):
          if self.zero:
             return 0.0
          number = 1
          for factor in self.g_array:
              indx = factor.index
              number = number * g_v[indx-1]
          for factor in self.y_array:
              indx = factor.index
              number = number * y_v[indx-1]
          for factor in self.gc_array:
              indx = factor.index
              number = number * g_v[indx-1].conj()
          for factor in self.yc_array:
              indx = factor.index
              number = number * y_v[indx-1].conj()

          #if number == 1:
          #   number = 0.0

          return number 

      def append_factor(self,factor_in):
          factor_in = deepcopy(factor_in)
          if factor_in.type_factor == "g":
             if factor_in.conjugate_value:
                self.gc_array = np.append(self.gc_array,factor_in)
             else:
                self.g_array = np.append(self.g_array,factor_in)
          elif  factor_in.type_factor == "y":
             if factor_in.conjugate_value:
                self.yc_array = np.append(self.yc_array,factor_in)
             else:
                self.y_array = np.append(self.y_array,factor_in) 
       
      def extract_index_array(self,array_type):
          indx_array = np.array([],dtype=int)
          if array_type == "g":
             for factor in self.g_array:
                 indx_array = np.append(indx_array,factor.index)
          elif array_type == "gc":
             for factor in self.gc_array:
                 indx_array = np.append(indx_array,factor.index) 
          elif array_type == "y":
             for factor in self.y_array:
                 indx_array = np.append(indx_array,factor.index)  
          elif array_type == "yc":
             for factor in self.yc_array:
                 indx_array = np.append(indx_array,factor.index)
          return indx_array
 
      def sort_factor(self,array_type):
          indx = self.extract_index_array(array_type)
          s_ind = np.argsort(indx)
          if array_type == "g":
             self.g_array = self.g_array[s_ind]
          elif array_type == "gc":
             self.gc_array = self.gc_array[s_ind]
          elif array_type == "y":
             self.y_array = self.y_array[s_ind]
          elif array_type == "yc":
             self.yc_array = self.yc_array[s_ind]
      
      def sort_factors(self):
          self.sort_factor("g")
          self.sort_factor("gc")
          self.sort_factor("y")
          self.sort_factor("yc")     

      def conjugate(self):
          #print "self.g_array[0].to_string = ",self.g_array[0].to_string()
          g_t = deepcopy(self.g_array)
          self.g_array = deepcopy(self.gc_array)
          self.gc_array = g_t
          #print "self.g_array[0].to_string = ",self.g_array[0].to_string()

          for k in xrange(len(self.g_array)):
              self.g_array[k].conjugate() 

          for k in xrange(len(self.gc_array)):
              self.gc_array[k].conjugate() 

          y_t = deepcopy(self.y_array)
          self.y_array = deepcopy(self.yc_array)
          self.yc_array = y_t

          for k in xrange(len(self.y_array)):
              self.y_array[k].conjugate() 

          for k in xrange(len(self.yc_array)):
              self.yc_array[k].conjugate() 

      def simplify_conjugates(self):
          ind1 = self.extract_index_array("g")
          ind2 = self.extract_index_array("gc")
          del1 = np.ones((len(ind1),),dtype=bool)
          del2 = np.ones((len(ind2),),dtype=bool)
 
          for k in xrange(len(ind2)):
              i = ind2[k]
              temp = np.where(i==ind1)[0]
              if not (len(temp) == 0):
                 del1[temp[0]] = 0
                 del2[k] = 0
                 temp = deepcopy(self.g_array[temp[0]])
                 temp.mag = True 
                 self.ga_array = np.append(self.ga_array,temp)
          self.g_array = self.g_array[del1]
          self.gc_array = self.gc_array[del2]

          ind1 = self.extract_index_array("y")
          ind2 = self.extract_index_array("yc")
          del1 = np.ones((len(ind1),),dtype=bool)
          del2 = np.ones((len(ind2),),dtype=bool)
 
          for k in xrange(len(ind2)):
              i = ind2[k]
              temp = np.where(i==ind1)[0]
              if not (len(temp) == 0):
                 del1[temp[0]] = 0
                 del2[k] = 0
                 temp = deepcopy(self.y_array[temp[0]])
                 temp.mag = True
                 self.ya_array = np.append(self.ya_array,temp)
          self.y_array = self.y_array[del1]
          self.yc_array = self.yc_array[del2]
           
      def multiply_arrays(self,a1,ind1,a2,ind2):
          product = deepcopy(a1)
          
          for k in xrange(len(ind2)):
              i = ind2[k]
              temp = np.where(i==ind1)[0] 
              if len(temp) == 0:
                 product = np.append(product,a2[k])
              else:
                 product[temp[0]].exponent = product[temp[0]].exponent+a2[k].exponent

          return product

      def setZero(self):
          self.zero = True
          self.g_array = np.array([],dtype=object)
          self.y_array = np.array([],dtype=object)
          self.yc_array = np.array([],dtype=object) 
          self.gc_array = np.array([],dtype=object) 

      def multiply_terms(self,in_term):
          if (self.zero) or (in_term.zero):
             self.zero = True
             self.g_array = np.array([],dtype=object)
             self.y_array = np.array([],dtype=object)
             self.yc_array = np.array([],dtype=object) 
             self.gc_array = np.array([],dtype=object) 
          else:
             ind1 = self.extract_index_array("g")
             ind2 = in_term.extract_index_array("g")
             self.g_array = self.multiply_arrays(self.g_array,ind1,in_term.g_array,ind2) 
             ind1 = self.extract_index_array("gc")
             ind2 = in_term.extract_index_array("gc")
             self.gc_array = self.multiply_arrays(self.gc_array,ind1,in_term.gc_array,ind2)
             ind1 = self.extract_index_array("y")
             ind2 = in_term.extract_index_array("y")
             self.y_array = self.multiply_arrays(self.y_array,ind1,in_term.y_array,ind2)
             ind1 = self.extract_index_array("yc")
             ind2 = in_term.extract_index_array("yc")
             self.yc_array = self.multiply_arrays(self.yc_array,ind1,in_term.yc_array,ind2)
             self.sort_factors() 
          #return self                  
            
      def diffirentiate_factor(self,f):#ASSUMING THE EXPONENT IS ALWAYS ONE FOR DIFFERENTIATION
          if self.zero:
             return

          if f.type_factor == "g":
             if f.conjugate_value:
                if len(self.gc_array) == 0:
                   self.setZero()
                   return
                ind = self.extract_index_array("gc")
                temp = np.where(ind==f.index)[0] 
                if len(temp) == 0:
                   self.setZero()
                   return
                else:
                   sel_ind = np.ones((len(ind),),dtype=bool)
                   sel_ind[temp[0]] = 0
                   self.gc_array = self.gc_array[sel_ind]     
                   return 
             else:  
                if len(self.g_array) == 0:
                   self.setZero()
                   return
                ind = self.extract_index_array("g")
                temp = np.where(ind==f.index)[0] 
                if len(temp) == 0:
                   self.setZero()
                   return
                else:
                   sel_ind = np.ones((len(ind),),dtype=bool)
                   sel_ind[temp[0]] = 0
                   self.g_array = self.g_array[sel_ind]     
                   return

          if f.type_factor == "y":
            if f.conjugate_value:
               if len(self.yc_array) == 0:
                  self.setZero()
                  return
               ind = self.extract_index_array("yc")
               temp = np.where(ind==f.index)[0] 
               if len(temp) == 0:
                  self.setZero()
                  return
               else:
                 sel_ind = np.ones((len(ind),),dtype=bool)
                 sel_ind[temp[0]] = 0
                 self.yc_array = self.yc_array[sel_ind]     
                 return   
            else:   
               if len(self.y_array) == 0:
                  self.setZero()
                  return
               ind = self.extract_index_array("y")
               temp = np.where(ind==f.index)[0] 
               if len(temp) == 0:
                  self.setZero()
                  return
               else:
                  sel_ind = np.ones((len(ind),),dtype=bool)
                  sel_ind[temp[0]] = 0
                  self.y_array = self.y_array[sel_ind]     
                  return

      def to_string(self,simplify=False):
          if simplify:
             self.simplify_conjugates()
          string_out = ""
          if self.zero:
             string_out = "0"
             return string_out
          if len(self.g_array) <> 0:
             for factor in self.g_array:
                 string_out = string_out+factor.to_string()
          if len(self.y_array) <> 0:  
             for factor in self.y_array:
                 string_out = string_out+factor.to_string()
          if len(self.yc_array) <> 0:  
             for factor in self.yc_array:
                 string_out = string_out+factor.to_string()
          if len(self.gc_array) <> 0:  
             for factor in self.gc_array:
                 string_out = string_out+factor.to_string()
          if len(self.ga_array) <> 0:  
             for factor in self.ga_array:
                 string_out = string_out+factor.to_string()
          if len(self.ya_array) <> 0:  
             for factor in self.ya_array:
                 string_out = string_out+factor.to_string() 
          return string_out 

class expression():
      def __init__(self,terms):
          self.terms = deepcopy(terms)

      def dot(self,exp_in):

          for k in xrange(len(exp_in.terms)):
              self.terms[k].multiply_terms(exp_in.terms[k]) 

      def to_string(self,simplify=False):
          string_out = ""
          for term in self.terms:
              if not term.zero: 
                 string_out = string_out + " + " + term.to_string(simplify)

          if string_out == "":
             string_out = "0" 
          else:
             string_out = string_out[3:]
          return string_out

      def number_of_terms(self):
          counter = 0
          for term in self.terms:
              if not term.zero:
                 counter = counter + 1

          return counter 

      def substitute(self,g_v,y_v):
          number = 0j
          for term in self.terms:
              if not term.zero:
                 number = number + term.substitute(g_v,y_v)
          return number 
           
class redundant():
      def __init__(self,N):
          self.N = N
          self.regular_array = np.array([],dtype=object)
          self.J1 = np.array([],dtype=object)
          self.Jc1 = np.array([],dtype=object)
          self.J2 = np.array([],dtype=object)
          self.Jc2 = np.array([],dtype=object)
          self.J = np.array([],dtype=object)
          self.JH = np.array([],dtype=object)
          self.H = np.array([],dtype=object)
          self.phi = np.array([])
         
      def create_J1(self,type_v="RED"):
          rows = ((self.N*self.N) - self.N)/2

          if type_v == "RED":
             columns = self.N + (self.N-1)
          elif type_v == "HEX":
             columns = self.N + int(np.amax(self.phi))
          else:
             columns = self.N
 
          self.J1 = np.empty((rows,columns),dtype=object)

          column_vector = np.array([],dtype=object)

          counter = 1
          for k in xrange(columns):
              if k <= self.N-1:
                 fact = factor("g",counter,1,False)
              else:
                 fact = factor("y",counter,1,False)
              column_vector = np.append(column_vector,fact) 
              counter = counter + 1
              if k == self.N-1:
                 counter = 1

          for c in xrange(columns):
              #print "column J1: ",c
              #print "###############"
              for r in xrange(rows):
                  
                  t_temp = deepcopy(self.regular_array[r])
                  #print "*********************"
                  #print "t_temp = ",t_temp.to_string()
                  #print "column_vector = ",column_vector[c].to_string()
                  t_temp.diffirentiate_factor(column_vector[c])
                  #print "t_temp_d = ",t_temp.to_string()
                  self.J1[r,c] = deepcopy(t_temp)
                  #print self.J1[r,c].to_string()
                  #print "*********************"   
                   
              #print "###############"

      def create_J2(self,type_v="RED"):
          
          rows = ((self.N*self.N) - self.N)/2
          if type_v == "RED":
             columns = self.N + (self.N-1)
          elif type_v == "HEX":
             columns = self.N + int(np.amax(self.phi))
          else:
             columns = self.N
          self.J2 = np.empty((rows,columns),dtype=object)

          column_vector = np.array([],dtype=object)

          counter = 1
          for k in xrange(columns):
              if k <= self.N-1:
                 fact = factor("g",counter,1,True)
              else:
                 fact = factor("y",counter,1,True)
              column_vector = np.append(column_vector,fact) 
              counter = counter + 1
              if k == self.N-1:
                 counter = 1

          for c in xrange(columns):
              #print "column J2: ",c
              #print "###############"
              for r in xrange(rows):
                  
                  t_temp = deepcopy(self.regular_array[r])
                  #print "*********************"
                  #print "t_temp = ",t_temp.to_string()
                  #print "column_vector = ",column_vector[c].to_string()
                  t_temp.diffirentiate_factor(column_vector[c])
                  #print "t_temp_d = ",t_temp.to_string()
                  self.J2[r,c] = deepcopy(t_temp)
                  #print self.J2[r,c].to_string()
                  #print "*********************"   
                   
              #print "###############"

      def create_J(self,type_v="RED"):
          rows = ((self.N*self.N) - self.N)
          if type_v == "RED":
             columns = 2*(self.N + (self.N-1))
          elif type_v == "HEX":
             columns = 2*(self.N + int(np.amax(self.phi)))
          else:
             columns = 2*(self.N)
          self.J = np.empty((rows,columns),dtype=object)
          self.J[:rows/2,:columns/2] = self.J1
          self.J[:rows/2,columns/2:] = self.J2
          self.J[rows/2:,:columns/2] = self.Jc2
          self.J[rows/2:,columns/2:] = self.Jc1

      #def create_J_L(self,type_v="RED"):
      #    rows = ((self.N*self.N) - self.N)
      #    if type_v == "RED":
      #       columns = (self.N + (self.N-1))
      #    elif type_v == "HEX":
      #       columns = (self.N + int(np.amax(self.phi)))
      #    else:
      #       columns = (self.N)
      #    self.J = np.empty((rows,columns),dtype=object)
      #    self.J[:rows/2,:columns] = self.J1
      #    self.J[:rows/2,columns/2:] = self.J2
      #    self.J[rows/2:,:columns/2] = self.Jc2
      #    self.J[rows/2:,columns/2:] = self.Jc1

      #def create_special_J(self,type_v="RED"):
      #    rows = ((self.N*self.N) - self.N)
      #    if type_v == "RED":
      #       columns = (self.N + (self.N-1))
      #    elif type_v == "HEX":
      #       columns = (self.N + int(np.amax(self.phi)))
      #    else:
      #       columns = (self.N)
      #    self.J = np.empty((rows,columns),dtype=object)
      #    self.J[:rows/2,:columns] = self.J1
      #    self.J[:rows/2,columns/2:] = self.J2
      #    self.J[rows/2:,:columns/2] = self.Jc2
      #    self.J[rows/2:,columns/2:] = self.Jc1 
     
      def to_string_J1(self):
          string_out = " J1 = ["
          for r in xrange(self.J1.shape[0]):
              for c in xrange(self.J1.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.J1[r,c].to_string() + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out  

      def to_string_J(self):
          string_out = " J = ["
          for r in xrange(self.J.shape[0]):
              for c in xrange(self.J.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.J[r,c].to_string() + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out  

      def to_string_JH(self):
          string_out = " J^H = ["
          for r in xrange(self.JH.shape[0]):
              for c in xrange(self.JH.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.JH[r,c].to_string() + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out  

      def to_string_J2(self):
          string_out = " J2 = ["
          for r in xrange(self.J2.shape[0]):
              for c in xrange(self.J2.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.J2[r,c].to_string() + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out  

      def to_string_Jc2(self):
          string_out = " Jc2 = ["
          for r in xrange(self.Jc2.shape[0]):
              for c in xrange(self.Jc2.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.Jc2[r,c].to_string() + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out 

      def to_string_Jc1(self):
          string_out = " Jc1 = ["
          for r in xrange(self.Jc1.shape[0]):
              for c in xrange(self.Jc1.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.Jc1[r,c].to_string() + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out           

      def conjugate_J1_J2(self):
          self.Jc1 = deepcopy(self.J1)
          self.Jc2 = deepcopy(self.J2)
          for r in xrange(self.Jc1.shape[0]):
              for c in xrange(self.Jc1.shape[1]): 
                  self.Jc1[r,c].conjugate()

          for r in xrange(self.Jc2.shape[0]):
              for c in xrange(self.Jc2.shape[1]): 
                  self.Jc2[r,c].conjugate()

      def hermitian_transpose_J(self):
          J_temp = deepcopy(self.J)
          
          for r in xrange(J_temp.shape[0]):
              for c in xrange(J_temp.shape[1]): 
                  J_temp[r,c].conjugate()
          self.JH = J_temp.transpose()   
          
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

      def calculate_phi(self,ant_x,ant_y,plot=False):
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
          return phi

      def create_hexagonal(self,hex_dim,l,print_pq=False):
          self.regular_array=np.array([],dtype=object)

          ant_x,ant_y = self.hex_grid_ver2(hex_dim,l)
          self.N = len(ant_x)
          #plt.plot(ant_x,ant_y,"ro")
          #plt.show()
          self.phi = self.calculate_phi(ant_x,ant_y)
          
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  g_fact = factor("g",p,1,False,False,0,0,False) 
                  y_fact = factor("y",int(self.phi[p-1,q-1]),1,False,False,p,q,print_pq)          
                  gc_fact = factor("g",q,1,True,False,0,0,False)
                  t_temp = term()
                  t_temp.append_factor(g_fact)
                  t_temp.append_factor(y_fact)
                  t_temp.append_factor(gc_fact)
                  self.regular_array = np.append(self.regular_array,t_temp)

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
 
          return ant_x,ant_y
                  
      def create_square(self,side,l,print_pq=False):
          self.regular_array=np.array([],dtype=object)

          ant_x,ant_y = self.square_grid(side,l)
          self.N = len(ant_x)
          #plt.plot(ant_x,ant_y,"ro")
          #plt.show()
          self.phi = self.calculate_phi(ant_x,ant_y)
          
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  g_fact = factor("g",p,1,False,False,0,0,False) 
                  y_fact = factor("y",int(self.phi[p-1,q-1]),1,False,False,p,q,print_pq)          
                  gc_fact = factor("g",q,1,True,False,0,0,False)
                  t_temp = term()
                  t_temp.append_factor(g_fact)
                  t_temp.append_factor(y_fact)
                  t_temp.append_factor(gc_fact)
                  self.regular_array = np.append(self.regular_array,t_temp)

      def create_regular(self,print_pq=False):
          spacing_vector = np.arange(self.N)[1:]  
          redundant_baselines = spacing_vector[::-1]          
          
          self.regular_array=np.array([],dtype=object)
          
          for k in xrange(len(redundant_baselines)):
              ant1 = 1
              for j in xrange(redundant_baselines[k]):
                  ant2 = ant1 + spacing_vector[k]
                  g_fact = factor("g",ant1,1,False,False,0,0,False)
                  y_fact = factor("y",spacing_vector[k],1,False,False,ant1,ant1,print_pq)
                  gc_fact = factor("g",ant2,1,True,False,0,0,False)
                  t_temp = term()
                  t_temp.append_factor(g_fact)
                  t_temp.append_factor(y_fact)
                  t_temp.append_factor(gc_fact)
                  self.regular_array = np.append(self.regular_array,t_temp)
                  ant1 = ant1 + 1

      def create_normal(self,print_pq=False):
          self.regular_array=np.array([],dtype=object)
          
          for p in xrange(1,self.N):
              y_counter = 1
              for q in xrange(p+1,self.N+1):
                  g_fact = factor("g",p,1,False,False,0,0,False) 
                  y_fact = factor("y",y_counter,1,False,False,p,q,print_pq)          
                  gc_fact = factor("g",q,1,True,False,0,0,False)
                  t_temp = term()
                  t_temp.append_factor(g_fact)
                  t_temp.append_factor(y_fact)
                  t_temp.append_factor(gc_fact)
                  self.regular_array = np.append(self.regular_array,t_temp)
                  y_counter = y_counter+1

      def create_regular_config2(self,print_pq=False):
          self.regular_array=np.array([],dtype=object)
          
          for p in xrange(1,self.N):
              y_counter = 1
              for q in xrange(p+1,self.N+1):
                  g_fact = factor("g",p,1,False,False,0,0,False) 
                  y_fact = factor("y",y_counter,1,False,False,p,q,print_pq)          
                  gc_fact = factor("g",q,1,True,False,0,0,False)
                  t_temp = term()
                  t_temp.append_factor(g_fact)
                  t_temp.append_factor(y_fact)
                  t_temp.append_factor(gc_fact)
                  self.regular_array = np.append(self.regular_array,t_temp)
                  y_counter = y_counter+1
     
      def to_string_regular(self):
          string_out = "regular_array = ["
          for entry in self.regular_array:
              string_out = string_out + entry.to_string()+","
          string_out = string_out[:-1]
          string_out = string_out+"]"
          return string_out

      def compute_H(self,type_v="RED"):
          print "ymax = ",np.amax(self.phi)
          if type_v == "RED":
             parameters = 2*(self.N + (self.N-1)) 
          elif type_v == "HEX":
             parameters = 2*(self.N + int(np.amax(self.phi)))
          else:
             parameters = 2*(self.N) 
          self.H = np.empty((parameters,parameters),dtype=object)  
          for r in xrange(parameters):
              for c in xrange(parameters):
                  row = expression(self.JH[r,:])
                  row_temp = deepcopy(row)
                  column = expression(self.J[:,c])
                  row_temp.dot(column)
                  self.H[r,c] = row_temp

      def substitute_H(self,g,y,type_v="RED"):
          if type_v == "RED":
             parameters = 2*(self.N + (self.N-1)) 
          elif type_v == "HEX":
             parameters = 2*(self.N + int(np.amax(self.phi)))
          else:
             parameters = 2*(self.N) 
          H_numerical = np.zeros((parameters,parameters),dtype=complex)  
          for r in xrange(parameters):
              for c in xrange(parameters):
                  H_numerical[r,c] = self.H[r,c].substitute(g,y)   
          return H_numerical        
                  
      def substitute_J(self,g,y,type_v="RED"):
          if type_v == "RED":
             parameters = 2*(self.N + (self.N-1))
             equations = (self.N**2 -self.N) 
          elif type_v == "HEX":
             parameters = 2*(self.N + int(np.amax(self.phi)))
             equations = (self.N**2 -self.N)
          else:
             parameters = 2*(self.N) 
             equations = (self.N**2 -self.N)
          J_numerical = np.zeros((equations,parameters),dtype=complex)  
          for r in xrange(equations):
              #print "r = ",r
              for c in xrange(parameters):
                  J_numerical[r,c] = self.J[r,c].substitute(g,y)   
          return J_numerical  

      def to_string_H(self,simplify=False):
          H_temp = deepcopy(self.H)
          string_out = " H = ["
          for r in xrange(H_temp.shape[0]):
              for c in xrange(H_temp.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + H_temp[r,c].to_string(simplify) + ","
              string_out = string_out[:-1]
              string_out = string_out+"\n" 
          string_out = string_out[:-1]
          string_out = string_out + "]"                  
          return string_out  

      def to_latex_H(self,simplify=False):
          file = open("H_"+str(self.N)+".txt", "w")
          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{H}_1 = \n\\begin{bmatrix}\n")
          string_out = ""
          H_temp = deepcopy(self.H[:2*self.N-1,:2*self.N-1])
          for r in xrange(H_temp.shape[0]):
              for c in xrange(H_temp.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  if r <> c:
                     string_out = string_out + H_temp[r,c].to_string(simplify) + "&"
                  else:
                     string_out = string_out + "a_{"+str(r)+"}" + "&"  
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")
 
          string_out = ""
          file.write("\\begin{eqnarray}\n")
          for r in xrange(H_temp.shape[0]):
              string_out = string_out + "a_{"+str(r)+"} &=& " + H_temp[r,r].to_string(simplify)+"\\\\\n"
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{eqnarray}\n")
               
          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{H}_2 = \n\\begin{bmatrix}\n")
          string_out = ""
          H_temp = deepcopy(self.H[:2*self.N-1,2*self.N-1:])
          for r in xrange(H_temp.shape[0]):
              for c in xrange(H_temp.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  #if r <> c:
                  string_out = string_out + H_temp[r,c].to_string(simplify) + "&"
                  #else:
                  #string_out = string_out + "b_{"+str(r)+"}" + "&"  
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          #string_out = ""
          #file.write("\\begin{eqnarray}\n")
          #for r in xrange(H_temp.shape[0]):
          #    string_out = string_out + "b_{"+str(r)+"} &=& " + H_temp[r,r].to_string(simplify)+"\\\\\n"
          #string_out = string_out[:-3]
          #file.write(string_out)
          #file.write("\n\\end{eqnarray}\n")

          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{H}_3 = \n\\begin{bmatrix}\n")
          string_out = ""
          H_temp = deepcopy(self.H[2*self.N-1:,2*self.N-1:])
          for r in xrange(H_temp.shape[0]):
              for c in xrange(H_temp.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  if r <> c:
                     string_out = string_out + H_temp[r,c].to_string(simplify) + "&"
                  else:
                     string_out = string_out + "b_{"+str(r)+"}" + "&"  
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          string_out = ""
          file.write("\\begin{eqnarray}\n")
          for r in xrange(H_temp.shape[0]):
              string_out = string_out + "b_{"+str(r)+"} &=& " + H_temp[r,r].to_string(simplify)+"\\\\\n"
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{eqnarray}\n")

          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{H}_4 = \n\\begin{bmatrix}\n")
          string_out = ""
          H_temp = deepcopy(self.H[2*self.N-1:,:2*self.N-1])
          for r in xrange(H_temp.shape[0]):
              for c in xrange(H_temp.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  #if r <> c:
                  string_out = string_out + H_temp[r,c].to_string(simplify) + "&"
                  #else:
                  #string_out = string_out + "d_{"+str(r)+"}" + "&"  
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          #string_out = ""
          #file.write("\\begin{eqnarray}\n")
          #for r in xrange(H_temp.shape[0]):
          #    string_out = string_out + "d_{"+str(r)+"} &=& " + H_temp[r,r].to_string(simplify)+"\\\\\n"
          #string_out = string_out[:-3]
          #file.write(string_out)
          #file.write("\n\\end{eqnarray}\n")

          file.close()

      def to_latex_JH(self,simplify=False):
          file = open("JH_"+str(self.N)+".txt", "w")
          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_1^H = \n\\begin{bmatrix}\n")
          string_out = ""
          JH_temp = deepcopy(self.JH[:2*self.N-1,:((self.N)**2-(self.N))/2])
          for r in xrange(JH_temp.shape[0]):
              for c in xrange(JH_temp.shape[1]):
                  string_out = string_out + JH_temp[r,c].to_string(simplify) + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")
 
          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_2^H = \n\\begin{bmatrix}\n")
          string_out = ""
          JH_temp = deepcopy(self.JH[:2*self.N-1,((self.N)**2-(self.N))/2:])
          for r in xrange(JH_temp.shape[0]):
              for c in xrange(JH_temp.shape[1]):
                  string_out = string_out + JH_temp[r,c].to_string(simplify) + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          #string_out = ""
          #file.write("\\begin{eqnarray}\n")
          #for r in xrange(H_temp.shape[0]):
          #    string_out = string_out + "b_{"+str(r)+"} &=& " + H_temp[r,r].to_string(simplify)+"\\\\\n"
          #string_out = string_out[:-3]
          #file.write(string_out)
          #file.write("\n\\end{eqnarray}\n")

          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_3^H = \n\\begin{bmatrix}\n")
          string_out = ""
          JH_temp = deepcopy(self.JH[2*self.N-1:,((self.N)**2-(self.N))/2:])
          for r in xrange(JH_temp.shape[0]):
              for c in xrange(JH_temp.shape[1]):
                  string_out = string_out + JH_temp[r,c].to_string(simplify) + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_4^H = \n\\begin{bmatrix}\n")
          string_out = ""
          JH_temp = deepcopy(self.JH[2*self.N-1:,:((self.N)**2-(self.N))/2])
          for r in xrange(JH_temp.shape[0]):
              for c in xrange(JH_temp.shape[1]):
                  string_out = string_out + JH_temp[r,c].to_string(simplify) + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          file.close()

      def to_latex_J(self):
          file = open("J_"+str(self.N)+".txt", "w")
          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_1 = \n\\begin{bmatrix}\n")
          string_out = ""
          J_temp = deepcopy(self.J1)
          for r in xrange(J_temp.shape[0]):
              for c in xrange(J_temp.shape[1]):
                  string_out = string_out + J_temp[r,c].to_string() + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")
 
          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_2 = \n\\begin{bmatrix}\n")
          string_out = ""
          J_temp = deepcopy(self.J2)
          for r in xrange(J_temp.shape[0]):
              for c in xrange(J_temp.shape[1]):
                  string_out = string_out + J_temp[r,c].to_string() + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_3 = \n\\begin{bmatrix}\n")
          string_out = ""
          J_temp = deepcopy(self.Jc2)
          for r in xrange(J_temp.shape[0]):
              for c in xrange(J_temp.shape[1]):
                  string_out = string_out + J_temp[r,c].to_string() + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          file.write("\\begin{equation}\n")
          file.write("\\boldsymbol{J}_4 = \n\\begin{bmatrix}\n")
          string_out = ""
          J_temp = deepcopy(self.Jc1)
          for r in xrange(J_temp.shape[0]):
              for c in xrange(J_temp.shape[1]):
                  string_out = string_out + J_temp[r,c].to_string() + "&"
              string_out = string_out[:-1]
              string_out = string_out+"\\\\\n" 
          string_out = string_out[:-3]
          file.write(string_out)
          file.write("\n\\end{bmatrix}\n")
          file.write("\\end{equation}\n")

          file.close()

      def to_int_H(self):
          H_int = np.zeros(self.H.shape,dtype=int)
          for r in xrange(self.H.shape[0]):
              for c in xrange(self.H.shape[1]):
                  H_int[r,c] = self.H[r,c].number_of_terms()
          return H_int   

                    
if __name__ == "__main__":
   r = redundant(0)
   r.create_hexagonal(1,20)
   #r.create_regular_config2(print_pq=True)
   #r.create_regular()
   #r.create_normal()
   print r.to_string_regular()
   r.create_J1(type_v="HEX")
   r.create_J2(type_v="HEX")
   print r.to_string_J1()
   print r.to_string_J2()
   r.conjugate_J1_J2()
   r.create_J(type_v="HEX")
   r.hermitian_transpose_J()
   r.compute_H(type_v="HEX")
   H_int = r.to_int_H()
   
   #print r.to_string_J1()
   #print r.to_string_J2()
   #print r.to_string_Jc1()
   #print r.to_string_Jc2()
   print r.to_string_J()
   #print r.to_string_JH()
   print r.to_string_H()
   print r.to_string_H(simplify=True)
   r.to_latex_H(simplify=True)
   #r.to_latex_J()
   #r.to_latex_JH()
   
   plt.imshow(H_int,interpolation="nearest")
   plt.colorbar()
   plt.show()

   H_int[H_int <> 0] = 1
   plt.imshow(H_int,interpolation="nearest")
   plt.colorbar()
   plt.show()
   x = np.sum(H_int)
   print "nonzero = ",x
   y = H_int.shape[0]
   y = y*y
   print "y = ",y
   z = y - x
   print "z = ",z
   print (1.*x)/y 
   #H_int_old = np.copy(H_int)
   #H_int[H_int <= 1] = 0
   #G = nx.Graph(H_int)
   #rcm = list(reverse_cuthill_mckee_ordering(G))
   #rcm = list(reverse_cuthill_mckee_ordering(G))
   #print "rcm = ",rcm
   #A1 = H_int_old[rcm, :][:, rcm]
   #plt.imshow(A1,interpolation='nearest')
   #plt.colorbar()
   #plt.show()
   """
   f1 = factor("g",1,1,True)
   f2 = factor("y",1,1,False)
   f3 = factor("g",2,1,False)  
   
   print "f1 = ",f1.to_string()
   print "f2 = ",f2.to_string()
   print "f3 = ",f3.to_string()

   t1 = term()
   t1.append_factor(f1)
   t1.append_factor(f2)
   t1.append_factor(f3)

   f4 = factor("g",1,1,True)
   f5 = factor("y",2,1,False)
   f6 = factor("g",3,1,False) 

   t2 = term()
   t2.append_factor(f4)
   t2.append_factor(f5)
   t2.append_factor(f6) 
 
   t1.multiply_terms(t2)

   t3 = term()
   t3.setZero()

   e1 = expression(np.array([t1,t2],dtype=object)) 
   e2 = expression(np.array([t1,t3],dtype=object)) 

   print "e1 = ",e1.to_string()
   print "e2 = ",e2.to_string()   

   e1.dot(e2)   
 
   print "e1 = ",e1.to_string()
 
   print "t1 = ",t1.to_string()

   #t1.conjugate()
   
   #print "t1 = ",t1.to_string()
   #print "t1 = ",t1.to_string()
   """
