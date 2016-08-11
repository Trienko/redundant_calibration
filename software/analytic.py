import numpy as np
import pylab as plt
import pickle
from scipy import optimize
import matplotlib as mpl
from copy import deepcopy
import simulator
#import networkx as nx
#from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering

'''
Analytic expression code, with which one can derive the analytic expressions for complex redundant calibration, LOGCAL,
LINCAL and redundant SteFCal (HESSIAN AND JACOBIAN).
'''

###############################################################################

'''
Class that represents a single algebraic factor in a term. Each term object (defined later in file) consists out of algebraic factors.
'''

class factor():

    '''
    Constructor
      
    RETURNS:
    None
      
    INPUTS:
    type_factor - The algebraic letter. It can be a g (antenna gain). Or y - the redundant visibilities.
    Or a and b the positions of the antennas. Or c indicating the factor is a constant.
    index - Index of factor. In the case of g it is the antenna to which this gain belongs, if it is 
    exponent - The factor raised to which power.
    conjugate_value - Indicating that this factor should be conjugated.
    mag - If mag is true the factor is interpreted as |factor|^2 when calling to_string
    ant_p - The first antenna used to form factor. If a g then meaningless.
    ant_q - The second antenna used to form factor. If a g then meaningless.
    print_f - Print the antennas that formed y instead of its redundant index.
    value - If factor is a constant this is its value. If value is 0 (default) it is not a constant.
    '''
    def __init__(self,
                type_factor = "g",
                index = 1, 
                exponent = 1,
                conjugate_value = False,
                mag = False,
                ant_p = 0,
                ant_q = 0,
                print_f = True,
                value = 0):
                self.type_factor = type_factor
                self.index = index
                self.exponent = exponent
                self.conjugate_value = conjugate_value
                self.mag = mag
                self.ant_p = ant_p
                self.ant_q = ant_q
                self.print_f = print_f 
                self.value = value
             
    '''
    Converts factor to a string
    INPUTS:
    None

    OUTPUTS:
    None
    '''
    def to_string(self):
        if self.print_f:
           if self.type_factor == "y":
              string_out = self.type_factor + "_{" + str(self.ant_p)+str(self.ant_q)+"}"
           elif self.type_factor == "c":
              string_out = "("+str(self.value)+")" + "_{" + str(self.ant_p)+str(self.ant_q)+"}"
           else:
              string_out = self.type_factor + "_" + str(self.index)
        else:   
           if self.type_factor == "c":
              string_out = "("+str(self.value) + ")" + "_" + str(self.index)
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
        
    '''
    Takes the conjugate of a factor
    INPUTS:
    None

    OUTPUTS:
    None
    '''   
    def conjugate(self):
        if self.conjugate_value:
           self.conjugate_value = False
        else:
           self.conjugate_value = True

    '''
    Determines if a factor is equal to factor_in
    INPUTS:
    factor_in - The input factor to compare against

    OUTPUTS:
    true or false depending if factors are equal
    '''
    def equal(self,factor_in):
        if self.type_factor == factor_in.type_factor:
           if self.index == factor_in.index:
              if self.exponent == factor_in.exponent:
                 if self.conjugate_value == factor_in.conjugate_value:
                    if self.value == factor_in.value:
                       return true
        return false

###############################################################################

'''
Class that represents an algebraic term (a term consists of many factors previously defined object). 
'''

#NB - NOT SURE IF THE CONSTANT, A or B arrays are empty the program will not crash did not build in any fail safe for this case yet. NEED TO TEST DIFFERENT SCENARIOS TO SEE IF IT CAN HANDLE EDGE CASES.

#NB - FIRST IMPLEMENTATION OF A,B and CONSTANT ARRAY DONE, NEED TO STILL UPDATE to_string OF factor TO HANDLE CHANGE --> NEXT STEP

#NB - COMMENT ALL THE FUNCTIONS

class term():
      '''
      Constructor
      
      RETURNS:
      None
      
      INPUTS:
      None
      '''

      def __init__(self):
          self.g_array = np.array([],dtype=object) #The g factors contained in the term
          self.y_array = np.array([],dtype=object) #The redundant visibilities y in the term
          self.yc_array = np.array([],dtype=object) # Conjugate of y 
          self.gc_array = np.array([],dtype=object) # Conjugate of g
          self.ga_array = np.array([],dtype=object) # If we simplify terms and g and its conjugate appear we store the result of the multi in here
          self.ya_array = np.array([],dtype=object) # Similar to ga only for y
          self.constant_array = np.array([],dtype=object) # The list of constants in the term
          self.a_array = np.array([],dtype=object) # The list of x antenna positions
          self.b_array = np.array([],dtype=object) # The list of y antenna positions
          self.zero = False # The term is equal to zero
          self.const = 0 #The final constant if we decide to multiply all the constants in the term together


      '''
      Substitutes real values of g and y into term to obtain result

      RETURNS:
      number - The product of the term after sub
      
      INPUTS:
      g_v - The g vector
      y_v - The y vector 
      ''' 
      def substitute(self,g_v,y_v): #STILL NEEDS TO BE FIXED
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
      '''
      Adds another factor to the term
      
      RETURNS:
      None
 
      INPUTS:
      factor_in - The factor to add to the term   
      '''  
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
          elif  factor_in.type_factor == "c":
                self.constant_array = np.append(self.constant_array,factor_in) 
          elif  factor_in.type_factor == "a":
                self.a_array = np.append(self.a_array,factor_in)
          elif  factor_in.type_factor == "b":
                self.b_array = np.append(self.b_array,factor_in)
      
 
      '''
      For a given variable it returns all the indices of that variable contained in the algebraic term

      RETURNS:
      indx_array - The indices for a specific variable

      INPUTS:
      array_type - The variable for which all indices needs to be returned. Can be
                   "g" - antenna gains
                   "y" - redundant spacings
                   "gc" - complex antenna gains
                   "yc" - complex redundant spacings
                   "a" - x antenna positions
                   "b" - y antenna positions
                   "c" - constant array
      '''
      def extract_index_array(self,array_type):
          indx_array = np.array([],dtype=int)
          if array_type == "g":
             if len(self.g_array) <> 0:
                for factor in self.g_array:
                    indx_array = np.append(indx_array,factor.index)
          elif array_type == "gc":
               if len(self.gc_array) <> 0: 
                  for factor in self.gc_array:
                      indx_array = np.append(indx_array,factor.index) 
          elif array_type == "y":
               if len(self.y_array) <> 0:
                  for factor in self.y_array:
                      indx_array = np.append(indx_array,factor.index)  
          elif array_type == "yc":
               if len(self.yc_array) <> 0:
                  for factor in self.yc_array:
                      indx_array = np.append(indx_array,factor.index)
          elif array_type == "c":
               if len(self.constant_array) <> 0:
                  for factor in self.constant_array:
                      indx_array = np.append(indx_array,factor.index)
          elif array_type == "a":
               if len(self.a_array) <> 0:
                  for factor in self.a_array:
                      indx_array = np.append(indx_array,factor.index)
          elif array_type == "b":
               if len(self.b_array) <> 0:
                  for factor in self.b_array:
                      indx_array = np.append(indx_array,factor.index)
          return indx_array
 
      '''
      Sort the different arrays according to index positions

      RETRUNS:
      None
 
      INPUTS:
      array_type - The array to sort (either the antenna gains, the redundant visibilities, the antenna positions or constants)       
      '''
      def sort_factor(self,array_type):
          indx = self.extract_index_array(array_type)
          if len(indx) <> 0:
             s_ind = np.argsort(indx)
             if array_type == "g":
                self.g_array = self.g_array[s_ind]
             elif array_type == "gc":
                self.gc_array = self.gc_array[s_ind]
             elif array_type == "y":
                self.y_array = self.y_array[s_ind]
             elif array_type == "yc":
                self.yc_array = self.yc_array[s_ind]
             elif array_type == "c":
                self.constant_array = self.constant_array[s_ind]
             elif array_type == "a":
               self.a_array = self.a_array[s_ind]
             elif array_type == "b":
               self.b_array = self.b_array[s_ind]
      '''
      Sort all the factors of all types according to their index postions 

      RETURNS:
      None

      INPUTS: 
      None
      ''' 
      def sort_factors(self):
          self.sort_factor("g")
          self.sort_factor("gc")
          self.sort_factor("y")
          self.sort_factor("yc") 
          self.sort_factor("c")
          self.sort_factor("a")
          self.sort_factor("b")      

     
      '''
      Takes the conjugate of all factors in the term
       
      Returns:
      None
     
      INPUTS: 
      None
      ''' 
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

      '''
      Calculates the total constant factor multiplies all the factors in the term together
      
      Returns:
      None
     
      INPUTS: 
      None
      ''' 
      def simplify_constant(self):
          self.const = 1.
          if len(self.constant_array) <> 0:
             for factor in self.constant_array:
                 self.const = self.const*factor.value**(factor.exponent)
          self.constant_array = np.array([])

      '''
      Searches the g and gc arrays to find an index match removes it from both and add to ga. Calculates the amplitudes
      if a factor and its conjugate is in the term. Only does this for g and gc; y and yc. 
      
      Returns:
      None
     
      INPUTS: 
      None
      '''       
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
           
      '''
      Multiply a1 (can be gains, red vis, x, y or const) and a2 toghether using the indices to do matching
      
      RETURNS:
      product - returns the product of the two arrays

      INPUTS:
      a1 - first array to multiply
      ind1 - the indices of the items in a1
      a2 - second array with which to multiply
      ind2 - the indices of the items in a2
      '''
      def multiply_arrays(self,a1,ind1,a2,ind2):
          product = deepcopy(a1)

          #if len(ind1) == 0:
          #   product = deepcopy(a2)
          #   return product
          
          for k in xrange(len(ind2)):
              i = ind2[k]
              temp = np.where(i==ind1)[0] 
              if len(temp) == 0:
                 product = np.append(product,a2[k])
              else:
                 
                 if product[temp[0]].type_factor == "c":
                    if (product[temp[0]].value == a2[k].value) and (product[temp[0]].ant_p == a2[k].ant_p) and (product[temp[0]].ant_q == a2[k].ant_q):
                       product[temp[0]].exponent = product[temp[0]].exponent+a2[k].exponent
                    else:
                       product = np.append(product,a2[k])  
                 else:
                    product[temp[0]].exponent = product[temp[0]].exponent+a2[k].exponent

          return product

      '''
      Set the term equal to zero
      
      INPUT:
      NONE
  
      OUTPUT:
      NONE
      '''
      def setZero(self):
          self.zero = True
          self.g_array = np.array([],dtype=object)
          self.y_array = np.array([],dtype=object)
          self.yc_array = np.array([],dtype=object) 
          self.gc_array = np.array([],dtype=object)
          self.a_array = np.array([],dtype=object)
          self.b_array = np.array([],dtype=object) 
          self.const = 0
          self.constant_array = np.array([],dtype=object)

      '''
      Multiply two terms together 

      INPUTS: 
      in_term - the term with which to multiply
      ''' 
      def multiply_terms(self,in_term):
          if (self.zero) or (in_term.zero):
             self.zero = True
             self.g_array = np.array([],dtype=object)
             self.y_array = np.array([],dtype=object)
             self.yc_array = np.array([],dtype=object) 
             self.gc_array = np.array([],dtype=object) 
             self.constant_array = np.array([],dtype=object)
             self.a_array = np.array([],dtype=object)
             self.b_array = np.array([],dtype=object)     
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
             ind1 = self.extract_index_array("c")
             ind2 = in_term.extract_index_array("c")
             self.constant_array = self.multiply_arrays(self.constant_array,ind1,in_term.constant_array,ind2)
             ind1 = self.extract_index_array("a")
             ind2 = in_term.extract_index_array("a")
             self.a_array = self.multiply_arrays(self.a_array,ind1,in_term.a_array,ind2)
             ind1 = self.extract_index_array("b")
             ind2 = in_term.extract_index_array("b")
             self.b_array = self.multiply_arrays(self.b_array,ind1,in_term.b_array,ind2) 
             self.sort_factors() 
          #return self                  
            
      '''
      Differentiate with respect to a factor
      
      RETURNS:
      None
        
      INPUT:
      f - differential
      '''
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


      '''
      Determines if the term is just a constant
      
      INPUTS:
      None

      RETURNS:
      is_constant - Returns True if the term is a constant
      '''
      def is_constant(self):
          if (len(self.g_array) == 0) and (len(self.gc_array) == 0) and (len(self.y_array) == 0) and (len(self.yc_array) == 0) and (len(self.a_array) == 0) and (len(self.b_array) == 0) and (len(self.const_array) <> 0):
             return True
          else:
             return False

      '''
      Converts the term into a printable string
      
      INPUTS:
      simplify - Simplifies the conjugates
      simplify_constant - Simplifies the constants

      OUTPUTS:
      string_out - The output string
      '''
      def to_string(self,simplify=False,simplify_constant=False):
          #print "HALLO"
          #print "self.zero = ",self.zero
          
          if simplify:
             self.simplify_conjugates()
          if simplify_constant:
             self.simplify_constant()

          string_out = ""
          if self.zero:
             string_out = "0"
             return string_out
          if not (np.iscomplex([self.const])[0]):
             self.const = self.const.real
             if self.const > 1:
                string_out = str(self.const)
             elif (len(self.g_array) == 0) and (len(self.gc_array) == 0) and (len(self.y_array) == 0) and (len(self.yc_array) == 0) and (len(self.a_array) == 0) and (len(self.b_array) == 0) and (np.allclose([self.const],[1])):
                   string_out = str(self.const)
                   return string_out
             elif self.const < 1:
                  string_out = str(self.const)
          else:
              string_out = str(self.const)
          if self.const == 0:
             if len(self.constant_array) <> 0:
                for factor in self.constant_array:
                    string_out = string_out+factor.to_string()
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
          if len(self.a_array) <> 0:
             for factor in self.a_array:
                 string_out = string_out+factor.to_string()
          if len(self.b_array) <> 0:
             for factor in self.b_array:
                 string_out = string_out+factor.to_string()
           
    
          return string_out 

###############################################################################

'''
Class that represents an algebraic expressions (an algebraic expression consists of terms which inturn consists out of many factors). 
'''

class expression():

      '''
      Constructor

      INPUTS:
      terms - an array of terms 

      RETURNS:
      None
      ''' 
      def __init__(self,terms):
          self.terms = deepcopy(terms)

      '''
      Takes the dot product of two equal length expressions

      INPUTS:
      exp_in - second expression in the dot product. The first is the current object itself.
      
      RETURNS:
      None
      ''' 
      def dot(self,exp_in):

          for k in xrange(len(exp_in.terms)):
              self.terms[k].multiply_terms(exp_in.terms[k]) 

      '''
      Converts an expression into a string

      INPUTS: 
      Simplify - Simplifies conjugates
      Simplify_const - Simplify constants

      RETURNS:
      string_out - The output string
      '''
      def to_string(self,simplify=False,simplify_const=False):
          string_out = ""
          for term in self.terms:
              if not term.zero: 
                 string_out = string_out + " + " + term.to_string(simplify=simplify,simplify_constant=simplify_const)

          if string_out == "":
             string_out = "0" 
          else:
             string_out = string_out[3:]
          return string_out

      '''
      Calculates the number of non-zero terms in an expression

      INPUTS:
      None

      RETURNS:
      counter - The number of terms in expression
      '''
      def number_of_terms(self):
          counter = 0
          for term in self.terms:
              if not term.zero:
                 counter = counter + 1

          return counter 

      '''
      Substitutes the g and y vector
      '''
      def substitute(self,g_v,y_v): #NB STILL NEED TO ADD OTHER VARIABLES AND CONSTANTS
          number = 0j
          for term in self.terms:
              if not term.zero:
                 number = number + term.substitute(g_v,y_v)
          return number 
           
class redundant():
      def __init__(self):
          self.N = 0
          self.L = 0
          self.phi = np.array([],dtype=int)
          self.zeta = np.array([],dtype=int)
          self.regular_array = np.array([],dtype=object)
          self.J1 = np.array([],dtype=object)
          self.Jc1 = np.array([],dtype=object)
          self.J2 = np.array([],dtype=object)
          self.Jc2 = np.array([],dtype=object)
          self.J = np.array([],dtype=object)
          self.JH = np.array([],dtype=object)
          self.H = np.array([],dtype=object)
          self.phi = np.array([])

      def create_J_LOGCAL(self,layout="REG",order=5,print_f=False):
          # GENERATES PHI --- NEED TO MAKE IT GLOBAL SO AS TO NOT RECALCULATE IT
          s = simulator.sim(layout=layout,order=order)
          s.generate_antenna_layout()
          phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
          self.N = s.N
          self.L = s.L 
          self.phi = phi
          self.zeta = zeta

          rows = (self.N**2 - self.N)/2
          columns = self.N + self.L

          self.J = np.zeros((rows,columns),dtype=object)

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
              phi_v = self.phi[p,q]
              for c in xrange(columns):
                  #print "r = ",r
                  #print "c = ",c
                  if c == p:
                     f = factor("c",p,1,False,ant_p=0,ant_q=0,print_f=False,value=1)
                     t = term()
                     t.append_factor(f)
                     self.J[r,c] = deepcopy(t)
                  elif c == q:
                     f = factor("c",q,1,False,ant_p=0,ant_q=0,print_f=False,value=1)
                     t = term()
                     t.append_factor(f)
                     self.J[r,c] = deepcopy(t)
                  elif (c == (self.N-1 + phi_v)):
                     f = factor("c",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=1)
                     t = term()
                     t.append_factor(f)
                     
                     self.J[r,c] = deepcopy(t)
                  else:
                     t = term()
                     t.setZero()
                     self.J[r,c] = deepcopy(t)
                     #print "Hallo_zero"
                     #print "t.zero = ",t.zero
                  self.J[r,c].to_string()
                  #print "self.J[r,c] = ",self.J[r,c]

      def create_J_LINCAL(self,layout="REG",order=5,print_f=False):
          # GENERATES PHI --- NEED TO MAKE IT GLOBAL SO AS TO NOT RECALCULATE IT
          s = simulator.sim(layout=layout,order=order)
          s.generate_antenna_layout()
          phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
          self.N = s.N
          self.L = s.L 
          self.phi = phi
          self.zeta = zeta

          rows = (self.N**2 - self.N)/2
          columns = 2*self.N + 2*self.L

          self.J = np.zeros((rows,columns),dtype=object)

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
              phi_v = self.phi[p,q]
              for c in xrange(columns):
                  #print "r = ",r
                  #print "c = ",c
                  if c == p:
                     f1 = factor("g",p,1,False,ant_p=0,ant_q=0,print_f=False,value=0)
                     f2 = factor("g",q,1,True,ant_p=0,ant_q=0,print_f=False,value=0)
                     f3 = factor("y",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=0)     
                     t = term()
                     t.append_factor(f1)
                     t.append_factor(f2)
                     t.append_factor(f3) 
                     self.J[r,c] = deepcopy(t)
                  elif c == q:
                     f1 = factor("g",p,1,False,ant_p=0,ant_q=0,print_f=False,value=0)
                     f2 = factor("g",q,1,True,ant_p=0,ant_q=0,print_f=False,value=0)
                     f3 = factor("y",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=0)     
                     t = term()
                     t.append_factor(f1)
                     t.append_factor(f2)
                     t.append_factor(f3) 
                     self.J[r,c] = deepcopy(t)
                  elif (c == (self.N + p)):
                     f1 = factor("c",p,1,False,ant_p=0,ant_q=0,print_f=print_f,value=1j)
                     f2 = factor("g",p,1,False,ant_p=0,ant_q=0,print_f=False,value=0)
                     f3 = factor("g",q,1,True,ant_p=0,ant_q=0,print_f=False,value=0)
                     f4 = factor("y",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=0)     
                     t = term()
                     t.append_factor(f1)
                     t.append_factor(f2)
                     t.append_factor(f3) 
                     t.append_factor(f4)
                     self.J[r,c] = deepcopy(t) 
                  elif (c == (self.N + q)):
                     f1 = factor("c",q,1,False,ant_p=0,ant_q=0,print_f=print_f,value=-1j)
                     f2 = factor("g",p,1,False,ant_p=0,ant_q=0,print_f=False,value=0)
                     f3 = factor("g",q,1,True,ant_p=0,ant_q=0,print_f=False,value=0)
                     f4 = factor("y",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=0)     
                     t = term()
                     t.append_factor(f1)
                     t.append_factor(f2)
                     t.append_factor(f3) 
                     t.append_factor(f4)
                     self.J[r,c] = deepcopy(t)
                  elif (c == (2*(self.N) - 1 + phi_v)):
                     f1 = factor("g",p,1,False,ant_p=0,ant_q=0,print_f=False,value=0)
                     f2 = factor("g",q,1,True,ant_p=0,ant_q=0,print_f=False,value=0)
                     f3 = factor("y",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=0)     
                     t = term()
                     t.append_factor(f1)
                     t.append_factor(f2)
                     t.append_factor(f3) 
                     self.J[r,c] = deepcopy(t)
                  elif (c == (2*(self.N) + self.L - 1 + phi_v)):
                     f1 = factor("c",p,1,False,ant_p=0,ant_q=0,print_f=print_f,value=1j) 
                     f2 = factor("g",p,1,False,ant_p=0,ant_q=0,print_f=False,value=0)
                     f3 = factor("g",q,1,True,ant_p=0,ant_q=0,print_f=False,value=0)
                     f4 = factor("y",phi_v,1,False,ant_p=p,ant_q=q,print_f=print_f,value=0)     
                     t = term()
                     t.append_factor(f1)
                     t.append_factor(f2)
                     t.append_factor(f3) 
                     t.append_factor(f4)
                     self.J[r,c] = deepcopy(t)  
                  else:
                     t = term()
                     t.setZero()
                     self.J[r,c] = deepcopy(t)
                     #print "Hallo_zero"
                     #print "t.zero = ",t.zero
                  self.J[r,c].to_string()
                  #print "self.J[r,c] = ",self.J[r,c]

       
      def create_J1(self,type_v="RED"):
          rows = ((self.N*self.N) - self.N)/2

          if type_v == "RED":
             columns = self.N + (self.N-1)
          elif type_v == "HEX":
             columns = self.N + int(np.amax(self.phi))
          elif type_v == "SQR":
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
          elif type_v == "SQR":
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

      def create_J(self,type_v="RED",phi=None):
          if phi <> None:
             self.phi = phi
          rows = ((self.N*self.N) - self.N)
          if type_v == "RED":
             columns = 2*(self.N + (self.N-1))
          elif type_v == "HEX":
             columns = 2*(self.N + int(np.amax(self.phi)))
          elif type_v == "SQR":
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

      def to_string_J(self,simplify=False,simplify_const=False):
          string_out = " J = ["
          for r in xrange(self.J.shape[0]):
              for c in xrange(self.J.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + self.J[r,c].to_string(simplify,simplify_const) + ","
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

      def compute_H(self):
          parameters = self.J.shape[1]
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
          elif type_v == "SQR":
             parameters = 2*(self.N + int(np.amax(self.phi)))
          else:
             parameters = 2*(self.N) 
          H_numerical = np.zeros((parameters,parameters),dtype=complex)  
          for r in xrange(parameters):
              for c in xrange(parameters):
                  H_numerical[r,c] = self.H[r,c].substitute(g,y)   
          return H_numerical        
                  
      def substitute_J(self,g,y,type_v="RED",phi=None):
          if phi <> None:
             self.phi = phi
          if type_v == "RED":
             parameters = 2*(self.N + (self.N-1))
             equations = (self.N**2 -self.N) 
          elif type_v == "HEX":
             parameters = 2*(self.N + int(np.amax(self.phi)))
             equations = (self.N**2 -self.N)
          elif type_v == "SQR":
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

      def to_string_H(self,simplify=False,simplify_const=False):
          H_temp = deepcopy(self.H)
          string_out = " H = ["
          for r in xrange(H_temp.shape[0]):
              for c in xrange(H_temp.shape[1]):
                  #print "r = ",r
                  #print "c = ",c
                  #print "self.J1[r,c] = ", self.J1[r,c].to_string()
                  string_out = string_out + H_temp[r,c].to_string(simplify=simplify,simplify_const=simplify_const) + ","
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
   r = redundant()
   #r.create_J_LOGCAL()
   r.create_J_LINCAL(layout="HEX",order=1)
   print r.to_string_J()
   r.JH = r.J.transpose()
   r.compute_H()
   print r.to_string_H(simplify=True,simplify_const=True)
   H_int = r.to_int_H()
   plt.imshow(H_int,interpolation="nearest")
   plt.show()
   print "H_int = ",H_int
   '''
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
   '''
   '''
   f1 = factor("g",1,1,True)
   f2 = factor("y",1,1,False,print_f=False)
   f3 = factor("a",3,1,False)
   f4 = factor("c",3,1,False,ant_p=3,ant_q=6,print_f=True,value=1)  
   
   print "f1 = ",f1.to_string()
   print "f2 = ",f2.to_string()
   print "f3 = ",f3.to_string()
   print "f4 = ",f4.to_string()

   t1 = term()
   t1.append_factor(f1)
   t1.append_factor(f2)
   t1.append_factor(f3)
   t1.append_factor(f4)

   f4 = factor("g",1,1,True)
   f5 = factor("y",2,1,False,print_f=False)
   f6 = factor("b",3,1,False) 
   f7 = factor("c",3,1,False,ant_p=3,ant_q=5,print_f=True,value=1)

   t2 = term()
   t2.append_factor(f4)
   t2.append_factor(f5)
   t2.append_factor(f6) 
   t2.append_factor(f7) 
 
   print "t1 = ",t1.to_string()
   print "t2 = ",t2.to_string()
  
   t1.multiply_terms(t2)

   print "t1 = ",t1.to_string(simplify_constant=True)

   t3 = term()
   t3.setZero()

   print "t3 = ",t3.to_string()

   #e1 = expression(np.array([t1,t2],dtype=object)) 
   #e2 = expression(np.array([t1,t3],dtype=object)) 

   #print "e1 = ",e1.to_string()
   #print "e2 = ",e2.to_string()   

   #e1.dot(e2)   
 
   #print "e1 = ",e1.to_string()
 
   #print "t1 = ",t1.to_string()

   #t1.conjugate()
   
   #print "t1 = ",t1.to_string()
   #print "t1 = ",t1.to_string()
   '''
