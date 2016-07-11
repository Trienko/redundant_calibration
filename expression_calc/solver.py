import numpy as np
import pylab as plt
import H_gen
import simulator
from scipy import sparse
from scipy import optimize
import os
import pickle


'''
Solver class that can use different methods to do redundant calibration. The aim being to compare these methods with one another
'''

class solver():
      
      '''
      Constructor 
      '''
      def __init__(self):
          self.J_sym = np.array([])
          self.J = np.array([])
          self.JH = np.array([])
          self.J_int = np.array([])
          self.H_sym = np.array([])
          self.H = np.array([])
          self.H_int = np.array([])
          #self.r_vec = np.array([])
          self.itr = 0
          
          self.itr_vec_mean = np.array([])
          self.itr_vec_std = np.array([])
 
          self.itr_list = []

          self.directory = "EXPERIMENT_NAME"
          self.txtfile = "TXTFILE.txt"
          self.verbose = True

      def write_to_current_textfile(self,str_to_write):
          if self.verbose:
             if not os.path.isdir("./"+self.directory):
                os.system("mkdir "+self.directory)
             if not os.path.isfile("./"+self.directory+"/"+self.txtfile):
                f = open("./"+self.directory+"/"+self.txtfile,"w")
             else:
                f = open("./"+self.directory+"/"+self.txtfile,"a")
          
             f.write(str_to_write+"\n")
             f.close()
            
      '''
      Creates the redundant Jacobian from variables. Stores the result in self.J_sym
      
      RETURNS:
      N/A
 
      INPUTS:
      N - Number of elements in array.     
      '''
      def construct_sym_J(self,N):
          r = H_gen.redundant(N)
          r.create_regular()
          r.create_J1()
          r.create_J2()
          r.conjugate_J1_J2()
          r.create_J()
          self.J_sym = r.J
          
      '''
      Substitutes the vector z into the symbolic redundant Jacobian. Stores the result in self.J
      
      RETURNS:
      N/A
 
      INPUTS:
      z - Input vector.
      N - Number of elements in array.
      plot - Plot the resultant matrices if True.     
      '''
      def substitute_sym_J(self,z,N,plot=False):
          g = z[:N]
          y = z[N:]
          r = H_gen.redundant(N)
          r.J = self.J_sym
          self.J = r.substitute_J(g,y)
          temp_J = np.copy(self.J)
          temp_J[np.absolute(temp_J) > 1e-12] = 1.0
          self.J_int = temp_J.real
          #print "self.J = ",self.J
          if plot:
             plt.imshow(np.absolute(self.J),interpolation="nearest")
             plt.show()
          if plot:
             plt.imshow(self.J_int,interpolation="nearest")
             plt.show() 


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

          return ant_x,ant_y

      def determine_phi_value(self,red_vec_x,red_vec_y,ant_x_p,ant_x_q,ant_y_p,ant_y_q):
          red_x = ant_x_q - ant_x_p
          red_y = ant_y_q - ant_y_p

          for l in xrange(len(red_vec_x)):
              if (np.allclose(red_x,red_vec_x[l]) and np.allclose(red_y,red_vec_y[l])):
                 return red_vec_x,red_vec_y,int(l+1)

          red_vec_x = np.append(red_vec_x,np.array([red_x]))
          red_vec_y = np.append(red_vec_y,np.array([red_y]))
          return red_vec_x,red_vec_y,int(len(red_vec_x)) 

      def calculate_phi(self,ant_x,ant_y,plot=True):
          phi = np.zeros((len(ant_x),len(ant_y)))
          zeta = np.zeros((len(ant_x),len(ant_y)))
          red_vec_x = np.array([])
          red_vec_y = np.array([])
          for k in xrange(len(ant_x)):
              for j in xrange(k+1,len(ant_x)):
                  red_vec_x,red_vec_y,phi[k,j]  = self.determine_phi_value(red_vec_x,red_vec_y,ant_x[k],ant_x[j],ant_y[k],ant_y[j])           
                  zeta[k,j] phi[k,j]
                  zeta[j,k] = zeta[k,j]
                  #phi[j,k] = phi[k,j]

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
          return phi,zeta

      def xi_func_eval(self,i,j,phi):
          column = phi[:,i]

          for p in xrange(len(column)):
              if column[p] == j:
                 return p

          return 0

      def psi_func_eval(self,i,j,phi):
          row = phi[i,:]

          for q in xrange(len(row)):
              if column[q] == j:
                 return q

          return 0

      '''
      INPUTS:
      z - Input vector.
      N - Number of elements in array.
      plot - Plot the resultant matrices if True.     
      '''
      def generate_H_formula(self,z,N,plot=False,dim=1,l=20,layout="HEX"):
          g = z[:N]
          y = z[N:]
          R = len(y)
          if layout == "HEX":
             ant_x,ant_y = self.hex_grid_ver2(dim,l)
          elif layout == "LIN":
             ant_x,ant_y = self.line_grid(dim,l)
          else:
             ant_x,ant_y = self.square_grid(dim,l)
  
          phi,zeta = self.calculate_phi(ant_x,ant_y,plot=False)
          
          #CONSTRUCTING C
          #**************
          C = np.zeros((N,N))
          for i in xrange(N):
              sum_v = 0
              for k in xrange(N):
                  if k != i:
                     sum_v = np.absolute(g[k])**2*np.absolute(y[zeta[k,i]-1])**2 + sum_v
              C[i,i] = sum_v


          #CONSTRUCTING E
          #**************
          E = np.zeros((R,R))
          for i in xrange(R):
              pq = zip(*np.where(phi == (i+1))) #[(0, 0), (0, 1), (1, 2), (1, 3)]
              sum_v = 0
              for k in xrange(len(pq)):
                  sum_v = np.absolute(g[pq[k][0]])^2*np.absolute(g[pq[k][1]])^2 + sum_v
              E[i,i] = sum_v

          #CONSTRUCTING D
          #**************
          D = np.zeros((N,R))
          for i in xrange(N):
              for j in xrange(R):
                  psi = psi_func_eval(i,j+1,phi)
                  if psi <> 0:
                     D[i,j] = g[i]*y[j]*np.absolute(g[psi])**2 

          DH = D.transpose.conj()   



         

      '''
      Creates the redundant Hessian from variables. Stores the result in self.H_sym
      
      RETURNS:
      N/A
 
      INPUTS:
      N - Number of elements in array.     
      '''
      def construct_sym_H(self,N):
          r = H_gen.redundant(N)
          r.create_regular()
          r.create_J1()
          r.create_J2()
          r.conjugate_J1_J2()
          r.create_J()
          r.hermitian_transpose_J()
          r.compute_H()
          self.H_sym = r.H
      
      '''
      Substitutes the vector z into the symbolic redundant Hessian. Stores the result in self.H
      
      RETURNS:
      N/A
 
      INPUTS:
      z - Input vector.
      N - Number of elements in array.
      plot - Plot the resultant matrices if True.     
      '''
      def substitute_sym_H(self,z,N,plot=False):
          g = z[:N]
          y = z[N:]
          r = H_gen.redundant(N)
          r.H = self.H_sym
          self.H_int = r.to_int_H()
          self.H = r.substitute_H(g,y)
          if plot:
             plt.imshow(np.absolute(self.H),interpolation="nearest")
             plt.show()
          if plot:
             plt.imshow(self.H_int,interpolation="nearest")
             plt.show()

      '''
      Computes the conjugate transpose of  the Jacobian
            
      RETURNS:
      N/A
 
      INPUTS:
      N/A   
      '''
      def compute_JH(self):
          self.JH = self.J.transpose().conj()

      '''
      Computes the Hessian

      RETURNS:
      N/A
 
      INPUTS:
      sparse_v - If true use sparse matrix format. 
      '''
      def compute_H(self,sparse_v=True):
          if sparse_v:
             JH_sparse = sparse.csr_matrix(self.JH)
             J_sparse = sparse.csc_matrix(self.J)
             res_sparse = JH_sparse.dot(J_sparse)
             self.H = res_sparse.toarray()
          else:
             self.H = np.dot(self.JH,self.J)



      '''
      Vectorizes a matrix by appending the upper diagonals of a matrix together. Starting from the diagonal just above the main diagonal

      RETURNS:
      d - The vectorized result
 
      INPUTS:
      V - The matrix to vectorize 
      '''
      def vectorize_matrix(self,V):
          N = V.shape[0]
          for k in xrange(1,N):
              d_temp = np.diag(V,k)
              if k == 1:
                 d = d_temp
              else:
                 d = np.append(d,d_temp) 
          return d

      '''
      Creates a matrix by filling the upper diagonals with the elements in vec

      RETURNS:
      M_temp - The created matrix.
 
      INPUTS:
      vec - The input vector. 
      '''
      def make_matrix_from_vector(self,vec,dtype=complex):
          M_temp = np.zeros((len(vec)+1,len(vec)+1),dtype=dtype) 
          for k in xrange(1,len(vec)+1):
                m_one = np.ones((len(np.diag(M_temp,k)),),dtype=dtype)
                M_temp = M_temp + np.diag(m_one*vec[k-1],k)
          return M_temp


      def make_Y(self,y):
          Y = np.zeros((y.shape[0]+1,y.shape[0]+1,y.shape[1]),dtype=complex)
          for t in xrange(Y.shape[2]):
              Y[:,:,t] = self.make_matrix_from_vector(y[:,t])
          return Y    

      '''
      Creates a matrix by computing gg^H

      RETURNS:
      g - Input vector
 
      INPUTS:
      G - gg^H. 
      '''
      def construct_G(self,g):
          temp =np.ones((len(g),len(g)) ,dtype=complex)
          G_temp = np.dot(np.diag(g),temp)
          G_temp = np.dot(G_temp,np.diag(g.conj()))
          return G_temp
      
      def make_G(self,g):
          G = np.zeros((g.shape[0],g.shape[0],g.shape[1]),dtype=complex)
          for t in xrange(G.shape[2]):
              G[:,:,t] = self.construct_G(g[:,t])
          return G  

      '''
      Creates the v (predicted visibilities) vector for a regular EW-array
     
      RETURNS:
      v - Predicted visibility vector.
      
      INPUTS:
      z - Input parameter vector consisting of gains and model visibilities.
      N - Number of elements in the array. 
      '''    
      def compute_v(self,z,N):
          g = z[:N]
          y = z[N:]
          Y = self.make_matrix_from_vector(y)
          G = self.construct_G(g)
          V = G*Y
          v = self.vectorize_matrix(V)
          return v

      def err_func(self,z0_new,d):
          N = int(0.5 + 0.5*np.sqrt(1+4*len(d)))
          g = z0_new[0:N]+1j*z0_new[N:2*N]
          y = z0_new[2*N:2*N+N-1] + 1j*z0_new[2*N+N-1:]
          Y = self.make_matrix_from_vector(y)
          G = self.construct_G(g)
          V = G*Y
          v = self.vectorize_matrix(V)
          v = np.hstack([v.real,v.imag])
          return d-v

      '''
      Vectorizes the observed visibility matrix
      
      RETRUNS:
      d - The vectrozed observed visibility matrix
      
      INPUTS:
      D - The observed visibility matrix
      ''' 
      def compute_d(self,D):
          d = self.vectorize_matrix(D)
          return d

      '''
      Computes the residual vector
      
      RETRUNS:
      r - The residual visibility vector
      
      INPUTS:
      D - The observed visibility matrix
      z - z - Input parameter vector consisting of gains and model visibilities.
      '''  
      def compute_r(self,D,z):
          N = D.shape[0]
          d = self.compute_d(D)
          v = self.compute_v(z,N)
          r = d-v
          return r
      
      '''
      Computes the product between the hermitian transpose of the Jacobian and the residual vectror

      RETURNS:
      N/A
 
      INPUTS:
      r - Residual visibility vector
      sparse_v - If true use sparse matrix format. 
      '''
      def compute_JHr(self,r,sparse_v=True):
          r = np.hstack([r,r.conj()])
          if  sparse_v:
              JH_sparse = sparse.csr_matrix(self.JH)
              self.JHr = JH_sparse.dot(r)
          else:
              self.JHr = np.dot(self.JH,r)

      def Anorm(self,x_true, xk, A):
          e = x_true - xk
          mat_vec_prod = np.dot(A,e)
          e_value = np.sqrt(np.sum(e.conj()*mat_vec_prod))
          return e_value

      def report(self,xk):
          if self.verbose:
             self.write_to_current_textfile("CG;itr;"+str(self.itr))
             x0 = np.ones(self.x_true.shape,dtype=complex)
             e0 = self.Anorm(self.x_true, x0, self.A) 
             ek = self.Anorm(self.x_true, xk, self.A)
             self.write_to_current_textfile("CG;e0;"+str(e0))
             self.write_to_current_textfile("CG;ek;"+str(ek))
             self.write_to_current_textfile("CG;rel_error;"+str(np.absolute(ek/e0)))  
          self.itr = self.itr + 1

      def timeslots_LM_complex(self,D,z0,maxit=5400,lam = 1,K = 10,eps1 = 1e-6,eps2 = 1e-6,eps3 = 1e-6,type_inverse="PCG",sparse_v=True):
          z_return = np.zeros((len(z0),D.shape[2]),dtype=complex)
          flags = np.zeros((D.shape[2],),dtype=bool)
          self.itr_vec_mean = np.zeros((D.shape[2],),dtype=float)
          self.itr_vec_std = np.zeros((D.shape[2],),dtype=float) 
          for t in xrange(D.shape[2]):
              print "t = ",t
              self.itr_list = []
              self.txtfile = "N_"+str(D.shape[0])+"_"+"t_"+str(t)+".txt"              
              self.write_to_current_textfile("####")
              D_temp = D[:,:,t]
              z_return[:,t],info = self.LM_solver_complex(D_temp,z0,maxit=maxit,lam=lam,K=K,eps1=eps1,eps2=eps2,eps3=eps3,type_inverse=type_inverse,sparse_v=sparse_v)
              print "info = ",info
              if len(self.itr_list) <> 0:
                 self.itr_vec_mean[t] = np.mean(self.itr_list)  
                 self.itr_vec_std[t] = np.std(self.itr_list)
              self.write_to_current_textfile("LM; z_return[:,t];"+str(z_return[:,t]))
              self.write_to_current_textfile("LM; info;"+str(info))
              self.write_to_current_textfile("####")
              if (info == 1):
                 flags[t] = False
              else:
                 flags[t] = True
          self.itr_vec_mean = self.itr_vec_mean[flags]
          self.itr_vec_std = self.itr_vec_std[flags]
          print "self.itr_vec_std = ",self.itr_vec_std
          mean_final = np.mean(self.itr_vec_mean)
          std_final = np.sqrt(1.0*np.sum(self.itr_vec_std**2)/len(self.itr_vec_std)) 
          print "f = ",std_final
          return z_return,flags,mean_final,std_final    
      
      def timeslots_LM_real_imag(self,D,z0):
          z_return = np.zeros((len(z0),D.shape[2]),dtype=complex)
          flags = np.zeros((D.shape[2],),dtype=bool)
          for t in xrange(D.shape[2]):
              print "t = ",t
              
              self.txtfile = "N_"+str(D.shape[0])+"_"+"t_"+str(t)+".txt"              
              self.write_to_current_textfile("####")
              D_temp = D[:,:,t]
              z_return[:,t],info = self.LM_solver_real_imaginary(D_temp,z0)
              self.write_to_current_textfile("LM; z_return[:,t];"+str(z_return[:,t]))
              self.write_to_current_textfile("LM; info;"+str(info))
              self.write_to_current_textfile("####")
              print "info = ",info
              if (info > 4):
                 flags[t] = False
              else:
                 flags[t] = True

          return z_return,flags 

      def LM_solver_real_imaginary(self,D,z0):
          N = D.shape[0]
          
          g = z0[0:N]
          y = z0[N:]

          z0_new = np.hstack([g.real,g.imag,y.real,y.imag])

          d = self.compute_d(D)
          d = np.hstack([d.real,d.imag])
          
          z_lstsqr_temp = optimize.leastsq(self.err_func, z0_new, args=(d), full_output = 1)
          self.write_to_current_textfile("LM; msg;"+z_lstsqr_temp[-2])
          self.write_to_current_textfile("LM; itrs;"+str(z_lstsqr_temp[2]['nfev']))
          #print "z_lstsqr_temp = ",z_lstsqr_temp[-1]
          #print "z_lstsqr_temp2 = ",z_lstsqr_temp[-2]
          #print "z_lstsqr_temp3 = ",z_lstsqr_temp[2]['nfev']
          z_lstsqr = z_lstsqr_temp[0]

          g = z_lstsqr[0:N]+1j*z_lstsqr[N:2*N]
          y = z_lstsqr[2*N:2*N+N-1] + 1j*z_lstsqr[2*N+N-1:]

          z = np.hstack([g,y]),z_lstsqr_temp[-1]
          
          return z

      def LM_solver_complex(self,D,z0,maxit=5400,lam = 0.01,K = 10,eps1 = 1e-6,eps2 = 1e-6,eps3 = 1e-6,type_inverse="PCG",sparse_v=True):
          z = np.copy(z0)
          r = self.compute_r(D,z)
          old_chi = np.linalg.norm(r)
          olam = lam
          itr = 0
          while True:
                
                self.write_to_current_textfile("LM;itr;"+str(itr)) 
                self.write_to_current_textfile("LM;lam;"+str(lam))
                self.write_to_current_textfile("LM;old_chi;"+str(old_chi))
                self.write_to_current_textfile("LM;maxit;"+str(maxit))  
                           
                dz = self.compute_update_step(D,z,type_inverse=type_inverse,lambda_fac=lam,sparse_v=sparse_v,tol=eps3)
                z += dz

                r = self.compute_r(D,z)
                new_chi = np.linalg.norm(r)
 
                self.write_to_current_textfile("LM;new_chi;"+str(new_chi))
                
                if new_chi < eps1:
                   return z,0
                elif np.linalg.norm(dz) < eps2*(np.linalg.norm(z)+eps2):
                   return z,0 

                if new_chi > old_chi:
                   z -= dz
                   lam = lam*K
                else:
                   old_chi = new_chi
                   lam = olam
 
                itr += 1 
                 
                #print "itr = ",itr
                #print "lam = ",lam
               
                if itr > maxit:
                   return z,1

      def timeslots_GN_complex(self,D,z0,maxit=5400,eps1 = 1e-6,eps2 = 1e-6,eps3 = 1e-6,type_inverse="PCG",sparse_v=True):
          z_return = np.zeros((len(z0),D.shape[2]),dtype=complex)
          flags = np.zeros((D.shape[2],),dtype=bool)
          for t in xrange(D.shape[2]):
              print "t = ",t
              self.txtfile = "N_"+str(D.shape[0])+"_"+"t_"+str(t)+".txt"              
              self.write_to_current_textfile("####")
              D_temp = D[:,:,t]
              z_return[:,t],info = self.GN_solver_complex(D_temp,z0,maxit=maxit,eps1=eps1,eps2=eps2,eps3=eps3,type_inverse=type_inverse,sparse_v=sparse_v)
              self.write_to_current_textfile("GN; z_return[:,t];"+str(z_return[:,t]))
              self.write_to_current_textfile("GN; info;"+str(info))
              self.write_to_current_textfile("####")
              print "info = ",info
              if (info == 1):
                 flags[t] = False
              else:
                 flags[t] = True

          return z_return,flags

      def GN_solver_complex(self,D,z0,maxit=5400,eps1 = 1e-6,eps2 = 1e-6,eps3 = 1e-6,type_inverse="PCG",sparse_v=True):
          z = np.copy(z0)
          itr = 0
          while True:
                
                self.write_to_current_textfile("GN;itr;"+str(itr)) 
                self.write_to_current_textfile("GN;maxit;"+str(maxit))  
                           
                dz = self.compute_update_step(D,z,type_inverse=type_inverse,lambda_fac=None,sparse_v=sparse_v,tol=eps3)
                z += 0.5*dz

                r = self.compute_r(D,z)
                new_chi = np.linalg.norm(r)
 
                self.write_to_current_textfile("LM;new_chi;"+str(new_chi))
                
                if new_chi < eps1:
                   return z,0
                elif np.linalg.norm(dz) < eps2*(np.linalg.norm(z)+eps2):
                   return z,0 

                itr += 1 
                 
                if itr > maxit:
                   return z,1

      def compute_update_step(self,D,z,type_inverse = "PCG",lambda_fac = None,sparse_v = True,tol = 1e-6):
          N = D.shape[0]
          #self.itr_list = []
          self.itr = 0
          self.write_to_current_textfile("****")
          self.substitute_sym_J(z,N)
          self.compute_JH()
          r = self.compute_r(D,z)
          self.compute_JHr(r,sparse_v=sparse_v)
          self.compute_H(sparse_v=sparse_v)
          
          if lambda_fac is not None:
             H_temp = self.H + np.diag(lambda_fac*np.diag(self.H))
          else:
             H_temp = np.copy(self.H)

          if type_inverse == "FULL":
             H_inv = np.pinv(H_temp)
             dz = np.dot(H_inv,self.JHr)
          elif type_inverse == "DIAG":
               H_inv = np.diag(np.diag(H_temp)**(-1))
               if sparse_v:
                  H_inv_sparse = sparse.dia_matrix(H_inv)
                  dz = H_inv_sparse.dot(self.JHr)
               else:
                  dz = np.dot(H_inv,self.JHr)
          elif type_inverse == "PCG":
               if self.verbose:
                     self.A = np.dot(np.diag(np.diag(H_temp)**(-1)),H_temp)
                     self.A2 = np.dot(self.A,self.A)
                     
                     m = np.sum(np.absolute(np.diag(self.A)))/self.A.shape[0]
                     s = np.sqrt(np.sum(np.absolute(np.diag(self.A2)))/self.A.shape[0] - m**2)
                     
                     lam_min_l = m - s*np.sqrt(self.A.shape[0]-1)
                     lam_min_u = m - s/(np.sqrt(self.A.shape[0]-1))

                     lam_max_l = m + s/(np.sqrt(self.A.shape[0]-1))
                     lam_max_u = m + s*np.sqrt(self.A.shape[0]-1)
                     
                     self.x_true = np.dot(np.linalg.pinv(H_temp),self.JHr)  
                     w = np.linalg.eigvals(self.A)
                     self.write_to_current_textfile("CG;w;"+str(w))
                     w = w.real
                     wb0 = w[w>1e-6]
                     lam_max = np.amax(wb0)
                     lam_min = np.amin(wb0)

                     column_sum = np.sum(np.absolute(self.A),axis = 0)
                     row_sum = np.sum(np.absolute(self.A),axis = 1)                     
                     max_c = np.amax(column_sum)
                     max_r = np.amax(row_sum)

                     min_rc = np.amin(np.array([max_c,max_r]))
                     self.write_to_current_textfile("CG;m;"+str(m)) 
                     self.write_to_current_textfile("CG;s;"+str(s)) 
                     self.write_to_current_textfile("CG;trA2;"+str(np.absolute(np.diag(self.A2)))) 
                     self.write_to_current_textfile("CG;lam_min_l;"+str(lam_min_l)) 
                     self.write_to_current_textfile("CG;lam_min_u;"+str(lam_min_u)) 
                     self.write_to_current_textfile("CG;lam_max_l;"+str(lam_max_l)) 
                     self.write_to_current_textfile("CG;lam_max_u;"+str(lam_max_u))
                     
                                         
                     self.write_to_current_textfile("CG;A_row;"+str(row_sum)) 
                     self.write_to_current_textfile("CG;A_column;"+str(column_sum))
                     self.write_to_current_textfile("CG;min_rc;"+str(min_rc))
                     #self.write_to_current_textfile("CG;A;"+str(self.A)) 
                     #self.write_to_current_textfile("CG;self.JHr;"+str(self.JHr)) 
                     self.write_to_current_textfile("CG;lam_max;"+str(lam_max)) 
                     self.write_to_current_textfile("CG;lam_min;"+str(lam_min))
                     self.write_to_current_textfile("CG;lam_max/lam_min;"+str(lam_max/lam_min)) 
                     self.write_to_current_textfile("CG;lam;"+str(w))
               if sparse_v:
                  H_m = sparse.csr_matrix(H_temp)
                  H_inv = sparse.dia_matrix(np.diag(np.diag(H_temp)**(-1)))
               else:
                  H_m = np.copy(H_temp)
                  H_inv = np.diag(np.diag(H_temp)**(-1)) 
               dz,info = sparse.linalg.cg(A=H_m,b=self.JHr,x0=np.ones(H_m.shape[0]),tol=tol,M=H_inv,callback=self.report)
               self.itr_list.append(self.itr)
               #print "info = ",info
          elif type_inverse == "CG":
               if self.verbose:
                     self.A = np.copy(H_temp)
                     H_inv = np.linalg.pinv(H_temp)
                     self.x_true = np.dot(H_inv,self.JHr)  
                     w = np.linalg.eigvals(self.A)
                     w = w.real
                     wb0 = w[w>1e-6]
                     lam_max = np.amax(wb0)
                     lam_min = np.amin(wb0)
                     self.write_to_current_textfile("CG;lam_max;"+str(lam_max)) 
                     self.write_to_current_textfile("CG;lam_min;"+str(lam_min))
                     self.write_to_current_textfile("CG;lam;"+str(w))  
               if sparse_v:
                  H_m = sparse.csr_matrix(H_temp)
               else:
                  H_m = np.copy(H_temp)
               dz,info = sparse.linalg.cg(A=H_m,b=self.JHr,x0=np.ones(H_m.shape[0]),tol=tol,callback=self.report)
               self.itr_list.append(self.itr)
          self.write_to_current_textfile("****")
          dz = dz[0:len(dz)/2] 
          return dz       

def experiment_as_func_N(N_min=5,N_max = 16, step_size = 5, max_amp=0.1,min_amp=0.05,freq_scale=5,time_steps=10,sig1=0.01,sig2=0.01,num_sources=10,fov=3,a=2,type_inverse="CG",pickle_name="test.p"):
    
    
    N_v = np.arange(N_min,N_max,step_size)
    mean_N = np.zeros(N_v.shape)
    std_N = np.zeros(N_v.shape)

    for k in xrange(len(N_v)):
          N_t = N_v[k]
          print "N_t = ",N_t
          s = simulator.sim() 
          g = s.create_antenna_gains(N=N_t,max_amp=max_amp,min_amp=min_amp,freq_scale=freq_scale,time_steps=time_steps,plot=False)
          point_sources = s.create_point_sources(num_sources=num_sources,fov=fov,a=a)
          u_m, v_m, w_m, b_m = s.create_uv_regular_line(N=N_t,time_steps=time_steps,plot=False)
          D = s.create_vis_mat(point_sources=point_sources,u_m=u_m,v_m=v_m,g=g,sig=sig1,w_m=None)
          M = s.create_vis_mat(point_sources=point_sources,u_m=u_m,v_m=v_m,g=None,sig=0,w_m=None)
          g_0 = np.ones((N_t,))
          y_0 = s.create_y_0_with_M(M,sig=sig2)
          z_0 = np.hstack([g_0,y_0[:,0]])
          solver_object = solver()
          solver_object.verbose = False
          solver_object.construct_sym_J(N_t)
          z,flags,mean_N[k],std_N[k] = solver_object.timeslots_LM_complex(D,z_0,type_inverse=type_inverse)
          
     
    filehandler = open(pickle_name,"wb")
    pickle.dump(N_v,filehandler)
    pickle.dump(mean_N,filehandler)
    pickle.dump(std_N,filehandler) 
    filehandler.close()
    #plt.plot(N_v,mean_N) 
    #plt.show()       

def plot_graph2(p1,p2):
    filehandler = open(p1,"rb")
    N_v = pickle.load(filehandler)
    mean_N = pickle.load(filehandler)
    std_N = pickle.load(filehandler)
    #plt.errorbar(x=N_v, y=mean_N, yerr=std_N,label="PCG")
    #print "yerr = ",std_N
    plt.plot(N_v,mean_N,"b",label="PCG")
    plt.fill_between(N_v, mean_N-std_N , mean_N+std_N, facecolor='blue', interpolate=True, alpha = 0.1)

    filehandler = open(p2,"rb")
    N_v = pickle.load(filehandler)
    mean_N = pickle.load(filehandler)
    std_N = pickle.load(filehandler)
    #plt.errorbar(x=N_v, y=mean_N, yerr=std_N,label="PCG")
    #print "yerr = ",std_N
    plt.plot(N_v,mean_N,"r",label="CG")
    plt.fill_between(N_v, mean_N-std_N , mean_N+std_N, facecolor='red', interpolate=True,alpha=0.1)
    plt.legend()
    plt.xlabel("Antennas [$N$]")
    plt.ylabel("Nbr of itr [$i$]")

    plt.show()

def plot_graph(p1,p2):
    filehandler = open(p1,"rb")
    N_v = pickle.load(filehandler)
    mean_N = pickle.load(filehandler)
    std_N = pickle.load(filehandler)
    plt.errorbar(x=N_v, y=mean_N, yerr=std_N,label="PCG")
    print "yerr = ",std_N
    filehandler.close()

    filehandler = open(p2,"rb")
    N_v = pickle.load(filehandler)
    mean_N = pickle.load(filehandler)
    std_N = pickle.load(filehandler)
    plt.errorbar(x=N_v, y=mean_N, yerr=std_N,label="CG")
    print "yerr = ",std_N
    filehandler.close()

    plt.legend()
    plt.xlabel("Antennas [$N$]")
    plt.ylabel("Nbr of itr [$i$]")

    plt.show()
   
if __name__ == "__main__":
   #experiment_as_func_N(N_min=5,N_max = 16, step_size = 5, max_amp=0.1,min_amp=0.05,freq_scale=5,time_steps=10,sig1=0.01,sig2=0.01,num_sources=10,fov=3,a=2,type_inverse="CG",pickle_name="test.p"):
   #experiment_as_func_N(N_min=5,N_max=40,step_size=5,max_amp=1.1,min_amp=0.9,time_steps=10,sig1=0.01,sig2=0.01,num_sources=10,type_inverse="PCG",pickle_name="PCG_HSNR.p")
   #experiment_as_func_N(N_min=5,N_max=60,step_size=5,max_amp=1.1,min_amp=0.9,time_steps=200,sig1=8,sig2=8,num_sources=100,type_inverse="PCG",pickle_name="PCG_LSNR.p")
   #experiment_as_func_N(N_min=5,N_max=40,step_size=5,max_amp=1.1,min_amp=0.9,time_steps=10,sig1=0.01,sig2=0.01,num_sources=10,type_inverse="CG",pickle_name="CG_HSNR.p")
   #experiment_as_func_N(N_min=5,N_max=60,step_size=5,max_amp=1.1,min_amp=0.9,time_steps=200,sig1=8,sig2=8,num_sources=100,type_inverse="CG",pickle_name="CG_LSNR.p")
   #plot_graph2("PCG_HSNR.p","CG_HSNR.p")
   
   N = 7
   s = simulator.sim()
   g = s.create_antenna_gains(N=N,max_amp=0.1,min_amp=0.05,freq_scale=5,time_steps=600,plot=True)
   point_sources = s.create_point_sources(num_sources=100,fov=3,a=2)
   u_m, v_m, w_m, b_m = s.create_uv_regular_line(N=N)
   D = s.create_vis_mat(point_sources=point_sources,u_m=u_m,v_m=v_m,g=g,sig=0.01,w_m=None)
   M = s.create_vis_mat(point_sources=point_sources,u_m=u_m,v_m=v_m,g=None,sig=0,w_m=None)
   s.plot_baseline(D,[0,1],shw=False)
   s.plot_baseline(M,[0,1],shw=True)

   g_0 = np.ones((N,))
   y_0 = s.create_y_0_with_M(M,sig=0.01)
   z_0 = np.hstack([g_0,y_0[:,0]])

   solver_object = solver()
   

   solver_object.construct_sym_J(N)
   #solver_object.construct_sym_H(N)
   z,flags,v1,v2 = solver_object.timeslots_LM_complex(D,z_0)
   #z = solver_object.timeslots_LM_real_imag(D,z_0)
   g = z[0:N,:]
   y = z[N:,:]

   Y = solver_object.make_Y(y)
   G = solver_object.make_G(g)

   s.plot_baseline(D,[0,1],shw=False)
   #s.plot_baseline(Y*G,[0,1],shw=True)
   s.plot_baseline(Y*G,[0,1],shw=True,flag=flags)
   #z = solver_object.LM_solver_real_imaginary(D[:,:,0],z_0)
   #z,info = solver_object.LM_solver_complex(D[:,:,0],z_0)
   print "z = ",z
   print "flags = ",flags
   #dz = solver_object.compute_update_step(D[:,:,0],z_0)
   #print "dz = ",dz
   #print "len(dz) = ",len(dz)
   #print "itr = ",solver_object.itr
   """
   #solver_object.substitute_sym_J(z_0,N,plot=True)
   #solver_object.substitute_sym_H(z_0,N,plot=True)
   #H_temp = np.copy(solver_object.H)
   #solver_object.compute_JH()
   #solver_object.compute_H()
   #res_H = H_temp - solver_object.H
   #plt.imshow(np.absolute(res_H),interpolation="nearest")
   #plt.show() 
   #print "res_H = ",res_H

   #v = solver_object.compute_v(z_0,N)
   #d = solver_object.compute_d(D[:,:,0])

   #print "v = ",v
   #print "D = ",D[:,:,0]
   #print "d = ",d
   
   #solver_object.compute_r(D[:,:,0],z_0)
   #solver_object.compute_JHr()
   #print "JHr = ",solver_object.JHr
   """


    
   
          
          
          
