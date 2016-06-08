import scipy as sp
import numpy as np
import pylab as plt
import pickle
from scipy import optimize
import matplotlib as mpl
from copy import deepcopy
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering

class test_redundant_calibration():

      def __init__(self):
          self.itr = 0
          self.itr_predict = 0
          self.mylist = []
          self.mylist2 = []
          self.compute_error = False
          self.error_list = []
          self.M = np.array([])
          pass
      
      def create_vis_mat(self,point_sources,u_m,v_m,g,sig):
          D = np.zeros(u_m.shape)
          G = np.diag(g)
          print "D.shape = ",D.shape
          print "G.shape = ",G.shape
          #Step 1: Create Model Visibility Matrix
          for k in xrange(len(point_sources)): #for each point source
              
              l_0 = point_sources[k,1]
              m_0 = point_sources[k,2]
              print "point_sources[k,0] = ",point_sources[k,0]
              D = D + point_sources[k,0]*np.exp(-2*np.pi*1j*(u_m*l_0+v_m*m_0))
    
          for t in xrange(D.shape[2]): #for each time-step
        
              #Step 2: Corrupting the Visibilities 
              D[:,:,t] = np.dot(G,D[:,:,t])
              D[:,:,t] = np.dot(D[:,:,t],G.conj()) 
        
              #Step 3: Adding Noise
              D[:,:,t] = D[:,:,t] + sig*np.random.randn(u_m.shape[0],u_m.shape[1]) + sig*np.random.randn(u_m.shape[0],u_m.shape[1])*1j
    
          return D

      def create_uv_regular_line(self,spacing_in_meters=20,antennas=5,lam=0.21428571428571427,dec=60,H_min=-6,H_max=6,time_steps=600,plot=True):
          b = np.ones((antennas,),dtype=float)*spacing_in_meters
          b = np.cumsum(b)
          if plot:
             plt.plot(b,np.zeros((antennas,),dtype=float),'ro')
             plt.xlabel("East-West [m]", fontsize=18)
             plt.ylabel("North-South [m]", fontsize=18)
             plt.title("ENU-coordinates of interferometer.", fontsize=18)
             plt.show()
          b = b/lam
          #print "b = ",b

          bas = np.zeros(((antennas**2-antennas)/2,),dtype=float)
          bas_2 = np.zeros(((antennas**2-antennas)/2,),dtype=float)
          counter = 0
          for p in xrange(0,len(b)-1):
              for q in xrange(p+1,len(b)):
                  bas[counter] = b[q] - b[p] 
                  bas_2[counter] = q - p
                  counter = counter + 1          
 
          H = np.linspace(H_min,H_max,time_steps)*(np.pi/12) #Hour angle in radians
          delta = dec*(np.pi/180) #Declination in radians

          u = np.zeros((len(bas),len(H)))
          v = np.zeros((len(bas),len(H)))
          for k in xrange(len(bas)):
              u[k,:] = bas[k]*np.cos(H)
              v[k,:] = bas[k]*np.sin(H)*np.sin(delta)
              if plot:
                 plt.plot(u[k,:],v[k,:],"r")
                 plt.plot(-u[k,:],-v[k,:],"b")
                 plt.xlabel("$u$ [rad$^{-1}$]", fontsize=18)
                 plt.ylabel("$v$ [rad$^{-1}$]", fontsize=18)
                 plt.title("$uv$-Coverage of three element interferometer", fontsize=18)
                 
          if plot:
             plt.show()

          u_m = np.zeros((len(b),len(b),len(H)))
          v_m = np.zeros((len(b),len(b),len(H)))

          b_m = np.zeros((len(b),len(b)),dtype=int)

          counter = 0
          for p in xrange(0,len(b)-1):
              for q in xrange(p+1,len(b)):
                  print "p,q = ",p,",",q
                  
                  u_m[p,q,:] = u[counter,:] #the first two entries denote p and q and the third index denotes time
                  u_m[q,p,:] = -1*u[counter,:]

                  v_m[p,q,:] = v[counter,:]
                  v_m[q,p,:] = -1*v[counter,:]
                  b_m[p,q] = bas_2[counter]
                  counter = counter + 1 

          #plt.plot(u_m[0,1,:],v_m[0,1,:])
          #plt.plot(u_m[0,2,:],v_m[0,2,:])
          #plt.show()
          return u_m,v_m,b_m

      def plot_redundant_visibilities(self,M,b_m,antennas=5,time_slot=0):

          M_temp = M[:,:,time_slot]

          str1 = "rbgycmk"
          str2 = "osx<>p"

          counter1 = 0
          counter2 = 0

          for k in xrange(1,antennas):
              if (counter1 == 7):
                 counter1 = 0
              if (counter2 == 6):
                 counter2 = 0 
              red_vis = M_temp[b_m==k]
              plt.plot(red_vis.real,red_vis.imag,str1[counter1]+str2[counter2],label=str(k))
              counter1 = counter1 + 1
              counter2 = counter2 + 1

          #plt.legend()
          #plt.show() 

      def plot_baseline(self,p,q,M,c,real=True):
          t = np.ones((len(M[p,q]),))
          t = np.cumsum(t)
          if real:
             plt.plot(t,M[p,q].real,c) 
          else:
             plt.plot(t,M[p,q].imag,c) 
      

      def err_func(self,g,d):
          #g contains the antenna gains and skymodel
          Nm = len(d)/2
          N = int(np.sqrt(Nm))
          #print "N = ",N
          G = np.diag(g[0:N]+1j*g[N:2*N])
          #print "G = ",G
          y = g[2*N:2*N+N-1] + 1j*g[2*N+N-1:]
          #print "y = ",y
          Y = np.zeros((N,N),dtype=complex)
          for k in xrange(1,N):
              #print "k = ",k
              temp_var = np.ones((N-k,),dtype=complex)*y[k-1]
              Y = Y + np.diag(temp_var,k)

          Y = Y + Y.transpose().conjugate()

          #print "Y = ",Y
           
          D = np.reshape(d[0:Nm],(N,N))+np.reshape(d[Nm:],(N,N))*1j #matrization
          T = np.dot(G,Y)
          T = np.dot(T,G.conj())
          #print "T = ",T
          #print "D = ",D
          R = D - T
          r_r = np.ravel(R.real) #vectorization
          r_i = np.ravel(R.imag)
          r = np.hstack([r_r,r_i])
          #print "r = ",r
          return r

      def invert_G(self,G_in):
          G = np.copy(G_in)
          for t in xrange(G.shape[2]):
              for r in xrange(G.shape[0]):
                  for c in xrange(G.shape[1]):
                      if r <> c:
                         G[r,c,t] = G[r,c,t]**(-1)
          return G         

      def plot_error(self,D,G,Y,t):
          N = D.shape[0] #number of antennas
          z_diag = np.ones((N,N),dtype=complex) - np.diag(np.ones((N,),dtype=complex))
          E = (D[:,:,t] - G[:,:,t]*Y[:,:,t])*z_diag
          plt.imshow(np.absolute(E),interpolation="nearest")
          plt.colorbar()
          plt.show()
          
      def computeJ(self,z,plot=False):#hardcoding the 5 antenna case
          g = z[0:5]
          gc = g.conjugate()
          y = z[5:]
          J1 = np.array([[y[0]*gc[1],0,0,0,0,g[0]*gc[1],0,0,0],[0,y[0]*gc[2],0,0,0,g[1]*gc[2],0,0,0],[0,0,y[0]*gc[3],0,0,g[2]*gc[3],0,0,0],[0,0,0,y[0]*gc[4],0,g[3]*gc[4],0,0,0],[y[1]*gc[2],0,0,0,0,0,g[0]*gc[2],0,0],[0,y[1]*gc[3],0,0,0,0,g[1]*gc[3],0,0],[0,0,y[1]*gc[4],0,0,0,g[2]*gc[4],0,0],[y[2]*gc[3],0,0,0,0,0,0,g[0]*gc[3],0],[0,y[2]*gc[4],0,0,0,0,0,g[1]*gc[4],0],[y[3]*gc[4],0,0,0,0,0,0,0,g[0]*gc[4]]])
          #print "J1 = ",J1

          J2 = np.array([[0,g[0]*y[0],0,0,0,0,0,0,0],[0,0,g[1]*y[0],0,0,0,0,0,0],[0,0,0,g[2]*y[0],0,0,0,0,0],[0,0,0,0,g[3]*y[0],0,0,0,0],[0,0,g[0]*y[1],0,0,0,0,0,0],[0,0,0,g[1]*y[1],0,0,0,0,0],[0,0,0,0,g[2]*y[1],0,0,0,0],[0,0,0,g[0]*y[2],0,0,0,0,0],[0,0,0,0,g[1]*y[2],0,0,0,0],[0,0,0,0,g[0]*y[3],0,0,0,0]])
          #print "J2 = ",J2

          J = np.zeros((20,18),dtype=complex)
  
          J[0:10,0:9] = J1
          J[0:10:,9:] = J2
          J[10:,0:9] = J2.conjugate()
          J[10:,9:] = J1.conjugate()

          #print "J = ",J
          if plot:
             plt.imshow(np.absolute(J),interpolation="nearest")
             plt.colorbar()
             plt.show()
          return J

      def computev(self,z):
          g = z[0:5]
          gc = g.conjugate()
          y = z[5:]

          v = np.array([g[0]*y[0]*gc[1],g[1]*y[0]*gc[2],g[2]*y[0]*gc[3],g[3]*y[0]*gc[4],g[0]*y[1]*gc[2],g[1]*y[1]*gc[3],g[2]*y[1]*gc[4],g[0]*y[2]*gc[3],g[1]*y[2]*gc[4],g[0]*y[3]*gc[4]])
          return v    
          
      def computer(self,d,z):
	  r_t = d-self.computev(z)
	  r = np.hstack([r_t,r_t.conjugate()])
	  return r
	  
      def computeJHr(self,d,z):
          JH = self.computeJH(z,plot=False)
          r = self.computer(d,z)
          JHr = np.dot(JH,r)
          return JHr

      def computeH(self,z,plot=False):
          J = self.computeJ(z,plot=False)
          JH = self.computeJH(z,plot=False)
          H = np.dot(JH,J)  
          if plot:
             plt.imshow(np.absolute(H),interpolation="nearest")
             plt.colorbar()
             plt.show()
          return H 

      def computeJH(self,z,plot=False):
          J =  self.computeJ(z,plot=False)
          JH = J.transpose()
          JH = JH.conjugate()
          if plot:
             plt.imshow(np.absolute(J),interpolation="nearest")
             plt.colorbar()
             plt.show()

          return JH
          
      def construct_fake_d(self,z,sig=20):
	  spacing_vector = np.arange(5)[1:]  
          redundant_baselines = spacing_vector[::-1]          
          g = z[0:5]
          #print "g = ",g
          y = z[5:]
          #print "y = ",y
          d = np.zeros((10,),dtype=complex)
          
          counter = 0
          for k in xrange(len(redundant_baselines)):
              ant1 = 0
              for j in xrange(redundant_baselines[k]):
                  
                  ant2 = ant1 + spacing_vector[k]
                  #print "ant1 , ant2 = ",ant1," , ",ant2
                  #print "spacing_vector[k] = ", spacing_vector[k]-1
                  d[counter] = g[ant1]*y[spacing_vector[k]-1]*g[ant2].conj() + sig*np.random.randn() + sig*np.random.randn()*1j  
                  ant1 = ant1 + 1
                  counter = counter + 1
                  
          return d        
          
      def update_step(self,d,z,sparse=False,cg=True):
	  z_t = z[0:len(z)/2]
	  H = self.computeH(z_t,plot=False)
          ones = np.ones((H.shape[0],),dtype=float)
          I = np.diag(ones)
          if sparse: 
             H_inv = np.linalg.pinv(H*I)
             #print "Hallo"
          else:
	     H_inv = np.linalg.pinv(H)
	  JHr = self.computeJHr(d,z_t)
          if not cg:
	     d_z = np.dot(H_inv,JHr)
          else:
             d_z = np.dot(H_inv,JHr)
             A_m = np.dot(np.linalg.pinv(H*I),H)
             #A_m = np.dot(H,np.sqrt(np.linalg.pinv(H*I).real))
             
             #c = np.linalg.cond(H)
             w = np.linalg.eigvals(H)
             print "w = ",w
             #print "max_w = ",np.amax(w)
             #print "min_w = ",np.amin(w)
             #print "dim = ",A_m.shape
             #print "condition number before= ",c
             #c = np.linalg.cond(A_m)
             #print "condition number after= ",c
             w = np.linalg.eigvals(A_m)
             #print "w = ",w
             lam_max = np.amax(w.real[w.real>1e-6])
             lam_min = np.absolute(np.min(w.real[w.real>1e-6]))
             lam = np.sort(w.real)
             #plt.plot(lam,np.ones(w.shape),"bo")
             #plt.show()
             print "NEW"
             print "*******"
             print "min_w = ",lam_max
             print "max_w = ",lam_min
             print "lam_max/lam_min = ",lam_max/lam_min
             print "w = ",w
             print "*******"
             #print "NEW"
             #print "*******"
             

             def report(xk):
                 #print "IN"
                 self.itr = self.itr + 1
                 if self.compute_error:
                    #print "len xk = ",len(xk)
                    e = self.x_c - xk
                    e_temp = np.dot(self.M,e)
                    #print "e_temp = ",len(e_temp)
                    e_v = np.sqrt(np.sum(e_temp*e.conj()))
                    #print "e_v = ",len(e_v)
                    e2 = self.x_c - np.ones(self.x_c.shape)
                    e_temp2 = np.dot(self.M,e2)   
                    e_v2 = np.sqrt(np.sum(e_temp2*e2.conj()))       

                    self.error_list.append(np.absolute(e_v)/np.absolute(e_v2))
             d_z,info = sp.sparse.linalg.cg(A=H,b=JHr,x0=np.ones(H.shape[0]),tol=1e-12,M=np.linalg.pinv(H*I)*I)
             #d_z,info = sp.sparse.linalg.cg(A=H,b=JHr,x0=np.ones(H.shape[0]),tol=1e-12)
             print "d_z1 = ",d_z
             self.compute_error = True
             if self.compute_error:
                temp_H = H*I
                H_diag = np.diag(temp_H)
                H_diag = H_diag**(-1)
                temp_H_inv = np.diag(H_diag)
                self.M = np.dot(temp_H_inv*I,H)
                self.error_list = []
                self.x_c = np.copy(d_z)
             self.itr = 0
             #d_z,info = sp.sparse.linalg.cg(A=H,b=JHr,x0=np.ones(H.shape[0]),tol=1e-5,callback=report)
             d_z,info = sp.sparse.linalg.cg(A=H,b=JHr,x0=np.ones(H.shape[0]),tol=1e-5,M=np.linalg.pinv(H*I)*I,callback=report)
             print "d_z2 = ",d_z
             #d_z,info = sp.sparse.linalg.cg(A=H,b=JHr,x0=d_z,callback=report)
             if self.compute_error:
                error_list_ar = np.array(self.error_list)
                K = np.cumsum(np.ones(error_list_ar.shape))
                plt.semilogy(K,error_list_ar)
                plt.semilogy(K,2*np.exp(-(2./np.sqrt(lam_max/lam_min))*K))
                plt.show()
             self.compute_error = False
             self.itr_predict = 0.5*np.log(2./1e-5)*np.sqrt(lam_max/lam_min)# np.log(2./1e-5)
             #print "self.itr = ",self.itr
             #print "info = ",info
	  return d_z
          
      def solve(self,d,z_0,iter_num=100,avg=True,sparse=True):
	  z = np.copy(z_0)
	  z = np.hstack([z,z.conj()])
	  #z_odd = 0.0
          #for k in xrange(1,iter_num):
          k = 1
          #iterations_new = 
          while True:
              dz = self.update_step(d,z,sparse)
              self.mylist = np.append(self.mylist,self.itr)
              self.mylist2 = np.append(self.mylist2,self.itr_predict) 
              z_old = np.copy(z)
              if avg:
                 #z_old = np.copy(z)
                 z += 0.1*dz
                 #z = (z + z_old)/2
                 #if (k%2) == 0:
                 #   z+= 0.5*dz
                 #else:
                 #   z+= dz
                 #if (k%2) == 0:
                 #   print "z_before_1 = ",z
                 #   z = z + dz
                 #   print "*****************"
                 #   print "k = ",k
                 #   print "dz = ",dz
                 #   print "z_before = ",z
                 #   print "z_odd = ",z_odd  
                 #   z = (z + z_odd)/2.0
                 #   print "z_after = ",z 
                 #   print "*****************"
                 #else:
                 #   print "#####################"
                 #   print "k = ",k
                 #   print "z_before = ",z
                 #   z += dz
                 #   print "z_after = ",z  
                 #   z_odd = np.copy(z)
                 #   print "z_odd = ",z_odd
                 #   print "#####################"
                  
              else:
                 z = z + dz
              
              if  np.sqrt(np.sum((np.absolute(z-z_old))**2))/np.sqrt(np.sum((np.absolute(z))**2)) < 1e-8:
                  break
              if  k > iter_num:
                  break
              if  np.sqrt(np.sum(np.absolute(self.computer(d,z) - self.computer(d,z_old))**2))/np.sqrt(np.sum(np.absolute(self.computer(d,z))**2)) < 1e-8:
                  break
              if  (np.sum(np.absolute(self.computer(d,z))**2) > np.sum(np.absolute(self.computer(d,z_old))**2)) and (k>2):
                  return z_old[0:len(z_0)]
                  break
              #print "np.sum(np.absolute(self.computer(d,z))**2) = ",np.sum(np.absolute(self.computer(d,z))**2)
              #print "np.sum(np.absolute(self.computer(d,z_old))**2) = ",np.sum(np.absolute(self.computer(d,z_old))**2)
              #print "x-y = ",np.sum(np.absolute(self.computer(d,z))**2)-np.sum(np.absolute(self.computer(d,z_old))**2) 
              k = k + 1
          print "k = ",k    
          return z[0:len(z_0)]

      #def plot_error2(p,q,D,G_o,Y_o):
      #    real_part = D[p,q,:].real - self.invert_g(G_o).real      

      def create_G_LM(self,D,M=np.array([]),use_M=False,sig=1.0):
          N = D.shape[0] #number of antennas

          z_diag = np.ones((N,N),dtype=complex) - np.diag(np.ones((N,),dtype=complex)) 
          
          temp = np.ones((D.shape[0],D.shape[1]) ,dtype=complex)
          G = np.zeros(D.shape,dtype=complex)
          Y = np.zeros(D.shape,dtype=complex)

          g = np.zeros((D.shape[0],D.shape[2]),dtype=complex)
          y = np.zeros((D.shape[0]-1,D.shape[2]),dtype=complex)
          r_out = np.zeros((D.shape[2],),dtype=float)
          r_in = np.zeros((D.shape[2],),dtype=float)
          for t in xrange(D.shape[2]): #perform calibration per time-slot
              D_temp = D[:,:,t]*z_diag
              if len(M) > 0:
                 M_temp = (M[:,:,t] + sig*np.random.randn(D_temp.shape[0],D_temp.shape[1]) + sig*np.random.randn(D_temp.shape[0],D_temp.shape[1])*1j)*z_diag 
              #print "M_temp = ",M_temp
              #first estimate of visibilities
              y_0_r = np.zeros((N-1,))
              for k in xrange(1,N):
                  if not use_M:
                     y_0_r[k-1] = np.mean(np.diag(D_temp.real,k))
                  else:
                     y_0_r[k-1] = np.mean(np.diag(M_temp.real,k))
              y_0_i = np.zeros((N-1,))
              for k in xrange(1,N):
                  if not use_M:
                     y_0_i[k-1] = np.mean(np.diag(D_temp.imag,k))
                  else:
                     y_0_i[k-1] = np.mean(np.diag(M_temp.imag,k))
              y_0 = np.hstack([y_0_r,y_0_i])
              #print "y_0 = ",y_0
              g_0 = np.ones((2*N,)) # first antenna gain guess 
              g_0[N:] = 0

              g_0 = np.hstack([g_0,y_0])
              #print "g_0 = ",g_0
              print "g_0.shape = ",g_0.shape
              print "t = ",t
              d_r = np.ravel(D_temp.real) #vectorization of observed + seperating real and imag
              d_i = np.ravel(D_temp.imag)
              d = np.hstack([d_r,d_i])
              #print "r_before = ",self.err_func(g_0,d)
              #print "r_before_val = ",np.sum(self.err_func(g_0,d)**2) 
              r_in[t] = np.sum(self.err_func(g_0,d)**2)
              g_lstsqr_temp = optimize.leastsq(self.err_func, g_0, args=(d))
              g_lstsqr = g_lstsqr_temp[0]          
              #print "r_after = ",self.err_func(g_lstsqr,d)
              #print "r_after_val = ",np.sum(self.err_func(g_lstsqr,d)**2) 
              r_out[t] = np.sum(self.err_func(g_lstsqr,d)**2)
              #print "g_lstsqr = ",g_lstsqr
              
              G_m = np.dot(np.diag(g_lstsqr[0:N]+1j*g_lstsqr[N:2*N]),temp)
              G_m = np.dot(G_m,np.diag((g_lstsqr[0:N]+1j*g_lstsqr[N:2*N]).conj())) 

              g[:,t] = g_lstsqr[0:N]+1j*g_lstsqr[N:2*N] #creating antenna gain vector       
              G[:,:,t] = G_m

              y[:,t] = g_lstsqr[2*N:2*N+N-1] + 1j*g_lstsqr[2*N+N-1:]

              Y_m = np.zeros((N,N),dtype=complex)
              for k in xrange(1,N):
                  temp_var = np.ones((N-k,),dtype=complex)*y[k-1,t]
                  Y_m = Y_m + np.diag(temp_var,k)

              Y_m = Y_m + Y_m.transpose().conjugate()

              Y[:,:,t] = Y_m
         
          return g,G,y,Y,r_out,r_in

      '''This function finds argmin G ||R-GMG^H|| using StEFCal.
      R is your observed visibilities matrx.
      M is your predicted visibilities.
      imax maximum amount of iterations.
      tau stopping criteria.
      g the antenna gains.
      G = gg^H.'''
      def create_G_stef(self,R,M,imax,tau):
          N = R.shape[0]
          temp =np.ones((R.shape[0],R.shape[1]) ,dtype=complex)
          G = np.zeros(R.shape,dtype=complex)
          g = np.zeros((R.shape[0],R.shape[2]),dtype=complex)

          for t in xrange(R.shape[2]):
              g_temp = np.ones((N,),dtype=complex) 
              for i in xrange(imax):
                  g_old = np.copy(g_temp)
                  for p in xrange(N):
                      z = g_old*M[:,p,t]
                      g_temp[p] = np.sum(np.conj(R[:,p,t])*z)/(np.sum(np.conj(z)*z))
                  if  (i%2 == 0):
                      if (np.sqrt(np.sum(np.absolute(g_temp-g_old)**2))/np.sqrt(np.sum(np.absolute(g_temp)**2)) <= tau):
                          break
                      else:
                          g_temp = (g_temp + g_old)/2
           
              G_m = np.dot(np.diag(g_temp),temp)
              G_m = np.dot(G_m,np.diag(g_temp.conj()))           

              g[:,t] = g_temp       
              G[:,:,t] = G_m
         
          return g,G

      '''This function finds argmin G ||R-GMG^H|| using StEFCal.
      R is your observed visibilities matrx.
      M is your predicted visibilities.
      imax maximum amount of iterations.
      tau stopping criteria.
      g the antenna gains.
      G = gg^H.'''
      def create_red_stef(self,D,imax,tau):

          N = D.shape[0]

          z_diag = np.ones((N,N),dtype=complex) - np.diag(np.ones((N,),dtype=complex))
          
          temp =np.ones((D.shape[0],D.shape[1]) ,dtype=complex)
          G = np.zeros(D.shape,dtype=complex)
          g = np.zeros((D.shape[0],D.shape[2]),dtype=complex)
              
          M = np.zeros(D.shape,dtype=complex) 
          m = np.zeros((D.shape[0]-1,D.shape[2]),dtype=complex)

          for t in xrange(D.shape[2]):
              print "t = ",t
              D_temp = z_diag*D[:,:,t]

              g_temp = np.ones((N,),dtype=complex) 
              G_temp = np.dot(np.diag(g_temp),temp)
              G_temp = np.dot(G_temp,np.diag(g_temp.conj()))

              m_temp = np.zeros((N-1,),dtype=complex) 
              M_temp = np.zeros((D.shape[0],D.shape[1]),dtype=complex)                  

              for i in xrange(1,imax):
                  m_old = np.copy(m_temp)
                  M_old = np.copy(M_temp)
                  m_temp = np.zeros((N-1,),dtype=complex) 
                  M_temp = np.zeros((D.shape[0],D.shape[1]),dtype=complex) 
                      
                  for k in xrange(1,N):
                      m_temp[k-1] = np.sum(np.conj(np.diag(G_temp,k))*np.diag(D_temp[:,:],k))/np.sum(np.absolute(np.diag(G_temp,k)))
                      m_one = np.ones((len(np.diag(D_temp[:,:],k)),),dtype=complex)
                      M_temp = M_temp + np.diag(m_one*m_temp[k-1],k)
                   
                  M_temp = M_temp + np.transpose(M_temp.conj())
                  #print "M_temp = ",M_temp
                  
                  #if (i%2 == 0):
                  #   M_temp = (M_temp + M_old)/2 
             
                  g_old = np.copy(g_temp)
                  for p in xrange(N):
                      z = g_old*M_temp[:,p]
                      g_temp[p] = np.sum(np.conj(D_temp[:,p])*z)/(np.sum(np.conj(z)*z))
                      
                  if (i%2 == 0):
                     z_temp = np.hstack([g_temp,m_temp])
                     z_old = np.hstack([g_old,m_old]) 
                     if (np.sqrt(np.sum(np.absolute(z_temp-z_old)**2))/np.sqrt(np.sum(np.absolute(z_temp)**2)) <= tau):
                        print "i = ",i
                        break
                     else:
                        #pass
                        g_temp = (g_temp + g_old)/2
           
                  G_temp = np.dot(np.diag(g_temp),temp)
                  G_temp = np.dot(G_temp,np.diag(g_temp.conj()))           

              g[:,t] = g_temp       
              G[:,:,t] = G_temp
              m[:,t] = m_temp
              M[:,:,t] = M_temp
         
          return g,G,m,M


      def create_G_sparse(self,D,M=np.array([]),use_M=False,sig=0.1,iter_num=1,avg=True,sparse=True):
           N = D.shape[0]
           temp = np.ones((D.shape[0],D.shape[1]) ,dtype=complex)
           G = np.zeros(D.shape,dtype=complex)
           Y = np.zeros(D.shape,dtype=complex)

           g = np.zeros((D.shape[0],D.shape[2]),dtype=complex)
           y = np.zeros((D.shape[0]-1,D.shape[2]),dtype=complex)
           for t in xrange(D.shape[2]): #perform calibration per time-slot
              print "t = ",t 
              #CREATING d
              D_temp = np.copy(D[:,:,t])
              for k in xrange(1,N):
                  d_temp = np.diag(D_temp,k)
                  if k == 1:
                     d = d_temp
                  else:
                     d = np.append(d,d_temp)
              

              #CREATING y_0
              if len(M) > 0:
                 M_temp = (M[:,:,t] + sig*np.random.randn(D_temp.shape[0],D_temp.shape[1]) + sig*np.random.randn(D_temp.shape[0],D_temp.shape[1])*1j) 
              y_0 = np.zeros((N-1,),dtype=complex)
              for k in xrange(1,N):
                  if not use_M:
                     y_0[k-1] = np.mean(np.diag(D_temp,k))
                  else:
                     y_0[k-1] = np.mean(np.diag(M_temp,k))
              
              g_0 = np.ones((N,),dtype=complex)  
              z_0 = np.hstack([g_0,y_0])

              z_t = self.solve(d,z_0,iter_num=iter_num,avg=avg,sparse=sparse) 
              
              G_m = np.dot(np.diag(z_t[0:N]),temp)
              G_m = np.dot(G_m,np.diag(z_t[0:N]).conj()) 

              g[:,t] = z_t[0:N] #creating antenna gain vector       
              G[:,:,t] = G_m

              y[:,t] = z_t[N:]

              Y_m = np.zeros((N,N),dtype=complex)
              for k in xrange(1,N):
                  temp_var = np.ones((N-k,),dtype=complex)*y[k-1,t]
                  Y_m = Y_m + np.diag(temp_var,k)

              Y_m = Y_m + Y_m.transpose().conjugate()

              Y[:,:,t] = Y_m
         
           return g,G,y,Y         


if __name__ == "__main__":
   r = test_redundant_calibration()
   u_m,v_m, b_m = r.create_uv_regular_line()
   point_sources = np.array([(3,0,0),(15,(1*np.pi)/180,(0*np.pi)/180),(1.5,(1*np.pi)/180,(-1*np.pi)/180),(1.5,(-1*np.pi)/180,(0*np.pi)/180),(2.5,(-2*np.pi)/180,(0*np.pi)/180),(10.5,(-1*np.pi)/180,(-2*np.pi)/180),(10.5,(-4*np.pi)/180,(-2*np.pi)/180),(20.5,(-5*np.pi)/180,(-2*np.pi)/180)]) #l and m are measured in radians
   g = np.array([2.+0.03j,1.007-0.8j,1.009,3.-0.8004j,1.1])
   sig = 0.1
   D = r.create_vis_mat(point_sources,u_m,v_m,g,sig) #we corrupt our data and we add noise
   g_un = np.array([1.,1.,1.,1.,1.])
   M = r.create_vis_mat(point_sources,u_m,v_m,g_un,0)
 
   #g_r,G_r,m_r,M_r = r.create_red_stef(D,100,1e-6)
   #g_o,G_o,y_o,Y_o,r_out,r_in = r.create_G_LM(D,M,use_M=True)
   g_s,G_s,y_s,Y_s = r.create_G_sparse(D,M,use_M=True)
   print "len = ",len(r.mylist)

   ar_v = np.array(r.mylist) 
   ar_v2 = np.array(r.mylist2) 
   
   plt.plot(np.cumsum(np.ones(ar_v.shape)),ar_v)
   plt.show()
   n, bins, patches = plt.hist(ar_v, 5, normed=1, facecolor='green', alpha=0.75)  
   plt.show()
   plt.plot(np.cumsum(np.ones(ar_v2.shape)),ar_v2)
   plt.show()
   n, bins, patches = plt.hist(ar_v2, 5, normed=1, facecolor='green', alpha=0.75)  
   plt.show()
   print "mean = ",np.mean(ar_v)
   print "mean = ",np.mean(ar_v2)
   #r.plot_baseline(0,1,M,"r",real=True)
   #r.plot_baseline(1,2,M,"b",real=True)
   
   #r.plot_baseline(1,2,D,"g",real=True)
   #r.plot_baseline(0,1,M,"c",real=True)
   #r.plot_baseline(1,2,D,"c",real=True)

   #r.plot_baseline(0,1,Y_o,"k",real=True)
   #r.plot_baseline(1,2,Y_o,"y",real=True)
   
   #r.plot_baseline(1,2,Y_o*G_o,"m",real=True)
   #r.plot_baseline(1,2,Y_s*G_s,"k",real=True)
   #r.plot_baseline(1,2,D*G_o[1,2]**(-1),"y",real=True)

   #plt.show()   

   #r.plot_baseline(1,2,D,"g",real=False)
   #r.plot_baseline(1,2,Y_o*G_o,"m",real=False)
   #r.plot_baseline(1,2,G_s*Y_s,"k",real=False)
   #plt.show()
   #r.plot_baseline(0,1,Y_o,"k",real=True)

   #r.plot_error2(0,1,D,G_o,Y_o)
   
   #r.plot_redundant_visibilities(M,b_m,antennas=5,time_slot=200) 

   #r.plot_redundant_visibilities(D,b_m,antennas=10,time_slot=200)
   #plt.show()

   #r.plot_redundant_visibilities(Y_o,b_m,antennas=5,time_slot=200)
   #r.plot_redundant_visibilities(D*r.invert_G(G_o),b_m,antennas=10,time_slot=200)
   #plt.show()      
   
   """
   r = test_redundant_calibration()
   u_m, v_m, b_m = r.create_uv_regular_line()
   point_sources = np.array([(10,0,0),(25,(1*np.pi)/180,(0*np.pi)/180)]) #l and m are measured in radians
   g_un = np.array([1.,1.,1.,1.,1.])
   M = r.create_vis_mat(point_sources,u_m,v_m,g_un,0) 
   y = np.zeros((4,),dtype=complex)
   for k in xrange(len(y)):
       y[k] = M[0,k+1,10]
   
   sig_t = 0.5
   
   y_0 = y + sig_t*np.random.randn() + sig_t*np.random.randn()*1j
   g = np.array([2.+0.1j,1.8-0.01j,1.05,3.-0.03j,1-4j])
   g_0 = np.array([1.,1.,1.,1.,1.])
   #y_0 = np.array([5.+4.1j,2+1.j,1+1.1j,1+4.3j])
   #y = np.array([5.+4j,2+1j,1+1j,1+4j])
   z = np.hstack([g,y])
   z_0 = np.hstack([g_0,y_0])

   #print "z = ",z
   d = r.construct_fake_d(z)
   z_new = r.solve(d,z_0,iter_num=10)
   z_new_slow = r.solve(d,z_0,iter_num=10,avg=False,sparse=False)
   
   print "z0 = ",z_0
   print "z = ",z
   print "z_new = ",z_new
   print "z_new_slow = ",z_new_slow 

   v = r.computev(z_new)
   v_s = r.computev(z_new_slow)
   v_0 = r.computev(z_0)  

   plt.plot(v_0.real,v_0.imag,"gx")
   plt.plot(d.real,d.imag,"rx")
   plt.plot(v.real,v.imag,"bx")
   plt.plot(v_s.real,v_s.imag,"kx")
   
   plt.show()

   #plt.plot(d.real[0:4],d.imag[0:4],"bx")
   #plt.plot(d.real[4:7],d.imag[4:7],"bo")
   #plt.plot(d.real[7:9],d.imag[7:9],"b^")
   #plt.plot(d.real[9:],d.imag[9:],"bs")
   #plt.plot(y.real[0],y.imag[0],"rx")
   #plt.plot(y.real[1],y.imag[1],"ro")
   #plt.plot(y.real[2],y.imag[2],"r^")
   #plt.plot(y.real[3],y.imag[3],"rs")
   #plt.show() 
   """
   """
   v = r.computev(z_0)
   
   #plt.plot(v.real,v.imag,"bx")
   #plt.plot(d.real,d.imag,"rx")
   #plt.show()

   r_var = r.computer(d,z_0)
   #plt.plot(r_var.real,r_var.imag,"gx")
   #plt.show()

   J = r.computeJ(z_0,plot=True)
   print "J.shape = ",J.shape
   JH = r.computeJH(z_0,plot=True)
   print "JH.shape = ",JH.shape
   H = r.computeH(z_0,plot=True)
   print "H.shape = ",H
   x = r.computeJHr(d,z_0)
   print "x = ",x
   print "x.shape = ",x.shape
   z_0_breve = np.hstack([z_0,z_0.conj()])
   dz = r.update_step(d,z_0_breve)
   print "dz = ",dz
   print "dz.shape = ",dz.shape
   """
   
   """
   H_int = np.absolute(H.copy())
   H_int[H>100] = 1
   H_int[H<100] = 0 
   plt.imshow(H_int,interpolation='nearest')
   plt.colorbar()
   plt.show()
   G = nx.Graph(H_int)
   #rcm = list(reverse_cuthill_mckee_ordering(G))
   rcm = list(reverse_cuthill_mckee_ordering(G))
   print "rcm = ",rcm
   A1 = H[rcm, :][:, rcm]
   plt.imshow(np.absolute(A1),interpolation='nearest')
   plt.colorbar()
   plt.show()
   """
   """
   #u_m,v_m, b_m = r.create_uv_regular_line()

   #point_sources = np.array([(30,0,0),(15,(1*np.pi)/180,(0*np.pi)/180)]) #l and m are measured in radians
   #g = np.array([3.+2.1j,1.-2.5j,4.1,1.-3.15j,20.,3.+2.1j,1.-1.5j,4.1,1.-6.15j,20.])
   #g = np.array([1.,1.,1.,1.,1.])
   #sig = 5
   #D = r.create_vis_mat(point_sources,u_m,v_m,g,sig) #we corrupt our data and we add noise
   g_un = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
   M = r.create_vis_mat(point_sources,u_m,v_m,g_un,0) 
   print "b_m = ",b_m
   g_o,G_o,y_o,Y_o,r_out,r_in = r.create_G_LM(D,M,use_M=True)
      
   t = np.ones((len(r_out),))
   t = np.cumsum(t)
   plt.plot(t,r_out)
   plt.show()
   plt.plot(t,r_in)
   plt.show()    

   r.plot_baseline(0,1,M,"r",real=True)
   #r.plot_baseline(1,2,M,"b",real=True)

   r.plot_baseline(0,1,D,"g",real=True)
   #r.plot_baseline(1,2,D,"c",real=True)

   r.plot_baseline(0,1,Y_o,"k",real=True)
   #r.plot_baseline(1,2,Y_o,"y",real=True)
   
   r.plot_baseline(0,1,Y_o*G_o,"m",real=True)
   #r.plot_baseline(1,2,D*G_o[1,2]**(-1),"y",real=True)

   plt.show()   

   #r.plot_baseline(0,1,Y_o,"k",real=True)

   #r.plot_error2(0,1,D,G_o,Y_o)
   
   #r.plot_redundant_visibilities(M,b_m,antennas=5,time_slot=200) 

   r.plot_redundant_visibilities(D,b_m,antennas=10,time_slot=200)
   plt.show()

   #r.plot_redundant_visibilities(Y_o,b_m,antennas=5,time_slot=200)
   r.plot_redundant_visibilities(D*r.invert_G(G_o),b_m,antennas=10,time_slot=200)
   plt.show()      
   """
