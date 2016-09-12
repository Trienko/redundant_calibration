import numpy as np
import scipy as sp
import pylab as plt
import simulator
import analytic
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.linalg import pinv
from scipy.sparse import dia_matrix
from scipy.stats import mode
import time
import pickle
import os
import sys, getopt
from scipy.stats import gaussian_kde
#from statistics import mode

class SPARC():

      def __init__(self,N,L,phi,zeta,PQ):

          self.N = N
          self.L = L
          self.B = (N**2-N)/2
          self.phi = phi
          self.zeta = zeta
          self.PQ = PQ
          self.itr = 0
          self.itr_vec = np.array([],dtype=int)
          self.kappa_vec = np.array([],dtype=float)

      def compute_v(self,z):
          g = z[:self.N] 
          y = z[self.N:]
           
         
          v = np.zeros((self.B,),dtype=complex)

          counter = 0
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  v[counter] = g[p-1]*np.conjugate(g[q-1])*y[self.phi[p-1,q-1]-1]
                  counter = counter + 1
          v = np.hstack([v,v.conjugate()])
        
          return v

      def vectorize_D(self,D):
          d = np.zeros((self.B,),dtype=complex)  
          
          counter = 0
          for p in xrange(1,self.N):
              for q in xrange(p+1,self.N+1):
                  d[counter] = D[p-1,q-1]
                  counter = counter + 1
          d = np.hstack([d,d.conjugate()])
        
          return d

      def compute_r(self,d,v):
          r = d-v
          return r

      def construct_R(self,r):
          r_new = r[:self.B]
          counter = 0
          R = np.zeros((self.N,self.N),dtype=complex)
          for k in xrange(self.N):
              for j in xrange(k+1,self.N):
                  R[k,j] = r_new[counter]
                  R[j,k] = np.conjugate(r_new[counter])
                  counter = counter + 1
          return R 

      def compute_J(self,z):
          g = z[:self.N]
          y = z[self.N:]

          rows = (self.N**2 - self.N)/2
          columns = self.N + self.L

          J1 = np.zeros((rows,columns),dtype=complex)
          J2 = np.zeros((rows,columns),dtype=complex)

          J = np.zeros((2*rows,2*columns),dtype=complex)
          
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
              phi_v = phi[p,q]
              for c in xrange(columns):
                  if c == p:
                     J1[r,c] = np.conjugate(g[q])*y[phi_v-1]            
                  if c == (self.N + phi_v - 1):
                     J1[r,c] = g[p]*np.conjugate(g[q])

          for r in xrange(rows):
              p = p_v[r]
              q = q_v[r]
              phi_v = phi[p,q]
              for c in xrange(columns):
                  if c == q:
                     J2[r,c] = g[p]*y[phi_v-1] 

          J[:rows,:columns] = J1
          J[:rows,columns:] = J2
          J[rows:,:columns] = J2.conjugate()
          J[rows:,columns:] = J1.conjugate()
          return J

      def compute_JHJ(self,J):
          return np.dot(J.transpose().conjugate(),J)

      def compute_JHr(self,J,r):
          return np.dot(J.transpose().conjugate(),r)

      def compute_JHr_formula(self,z,R):
          
          g = z[:self.N]
          y = z[self.N:]
          
          vec1 = np.zeros((self.N,),dtype=complex)
          for i in xrange(len(vec1)):
              sum_v = 0j
              for k in xrange(len(vec1)):
                  if k <> i:
                     if k > i:
                        x_ki = np.conjugate(y[self.zeta[i,k]-1])  
                     else:
                        x_ki = y[self.zeta[i,k]-1]
                     sum_v = sum_v + g[k]*x_ki*R[i,k]
              vec1[i] = sum_v

          vec2 = np.zeros((self.L,),dtype=complex)
          
          for i in xrange(len(vec2)):
              sum_v = 0j
              pq = self.PQ[str(i)]
              for k in xrange(len(pq)):
                  p = pq[k][0]
                  q = pq[k][1]
                  
                  sum_v = sum_v + np.conjugate(g[p])*g[q]*R[p,q]
              vec2[i] = sum_v

          return np.hstack([vec1,vec2,np.conjugate(vec1),np.conjugate(vec2)]) 

      def compute_inverse_cg(self,A,b,M=None,tol=1e-6):
          self.itr = 0
          x,info = cg(A=A,b=b,tol=tol,M=M,callback=self.report)
          self.itr_vec = np.append(self.itr_vec,np.array([self.itr]))
          return x,info

      def report(self,xk):
          self.itr = self.itr + 1

      def compute_kappa(self,A):
          w = np.linalg.eigvals(A)
          wb0 = w[np.nonzero(w)].real
          lam_max = np.amax(wb0)
          lam_min = np.amin(wb0)
          kappa = lam_max/lam_min
          return kappa

      def compute_update_step_cg(self,r,z,psi_func_eval,xi_func_eval,lam=0,method="PCG",tol=1e-6,kappa=True):
          R = self.construct_R(r)

          H,H_sparse,D_l,D_sparse = self.generate_H_formula(z,psi_func_eval,xi_func_eval,lam=lam)
          JHr_fun = self.compute_JHr_formula(z,R) #CAN STILL INCREASE THE SPEED BY DIRECTLY COMPUTING R form D and V
          
          if method == "PCG":
             dz,info = self.compute_inverse_cg(A=H_sparse,b=JHr_fun,M=D_sparse,tol=tol)
             if kappa:
                self.kappa_vec = np.append(self.kappa_vec,np.array([self.compute_kappa(np.dot(D_l,H))])) 
          elif method == "CG":
             dz,info = self.compute_inverse_cg(A=H,b=JHr_fun,M=None,tol=tol)
             if kappa:
                self.kappa_vec = np.append(self.kappa_vec,np.array([self.compute_kappa(H)])) 
          elif method == "SVD":
               H_inv = pinv(H)
               dz = np.dot(H_inv,JHr_fun)
               info = None
            
          return dz[:len(dz)/2],info


      def levenberg_marquardt(self,D,psi_func_eval,xi_func_eval,convert_y_to_M,tol1=1e-6,tol2=1e-6,tol3=1e-6,lam=2,max_itr=2000,method="PCG",time_var=True):

          temp = np.ones((D.shape[0],D.shape[1]) ,dtype=complex)
          z = np.ones((self.N+self.L,),dtype=complex)
 
          counter = 0

          converged = False

          d = self.vectorize_D(D)
          
          start = time.time()

          v = self.compute_v(z)
          r = self.compute_r(d,v)      
          
          while (True):
                
                dz,info = self.compute_update_step_cg(r,z,psi_func_eval,xi_func_eval,lam=lam,method=method,tol=tol3,kappa= not time_var)
                
                z = z + dz

                v = self.compute_v(z)
                r = self.compute_r(d,v)
               
                if not time_var:
                    
                   new_chi = np.linalg.norm(r)
                
                   #print "new_chi = ",new_chi
                   if new_chi < tol1:
                      converged = True 
                      break

                if np.linalg.norm(dz)/np.linalg.norm(z) < tol2:
                   converged = True
                   break 
                
                if counter > max_itr:
                   break
          
                counter = counter + 1

          stop = time.time()

          G = np.dot(np.diag(z[:self.N]),temp)
          G = np.dot(G,np.diag(z[:self.N].conj()))  
          M = convert_y_to_M(self.PQ,z[self.N:],self.N)  
                     
          return z,converged,G,M,start,stop,counter

      def levenberg_marquardt_time(self,D,psi_func_eval,xi_func_eval,convert_y_to_M,tol1=1e-6,tol2=1e-6,tol3=1e-6,lam=2,max_itr=2000,method="PCG",time_var=False):
          z_temp = np.zeros((self.N+self.L,D.shape[2]),dtype=complex)
          M = np.zeros((self.N,self.N,D.shape[2]),dtype=complex)
          G = np.zeros((self.N,self.N,D.shape[2]),dtype=complex)
          c_temp = np.zeros((D.shape[2],),dtype=bool)
          t_temp = np.zeros((2,D.shape[2]))
          count_temp = np.zeros((D.shape[2],),dtype=int)
          for t in xrange(D.shape[2]):
              print "###########"
              print "t= ",t
              print "###########"
              z_temp[:,t],c_temp[t],G[:,:,t],M[:,:,t],t_temp[0,t],t_temp[1,t],count_temp[t] = self.levenberg_marquardt(D[:,:,t],psi_func_eval,xi_func_eval,convert_y_to_M,tol1=tol1,tol2=tol2,tol3=tol3,lam=2,max_itr=max_itr,method=method,time_var=time_var)
              print "c_temp = ",c_temp[t]  
          return z_temp,c_temp,G,M,t_temp,count_temp    
           
      '''
      INPUTS:
      z - Input vector.
      '''
      def generate_H_formula(self,z,psi_func_eval,xi_func_eval,lam=0):
          g = z[:self.N]
          y = z[self.N:]
                    
          #CONSTRUCTING C
          #**************
          C = np.zeros((self.N,self.N),dtype=complex)
          for i in xrange(self.N):
              sum_v = 0j
              for k in xrange(self.N):
                  if k != i:
                     sum_v = np.absolute(g[k])**2*np.absolute(y[self.zeta[k,i]-1])**2 + sum_v
              C[i,i] = sum_v

          #CONSTRUCTING E
          #**************
          E = np.zeros((self.L,self.L),dtype=complex)
          for i in xrange(self.L):
              #pq = zip(*np.where(phi == (i+1))) #[(0, 0), (0, 1), (1, 2), (1, 3)]
              pq = self.PQ[str(i)]
              sum_v = 0j
              for k in xrange(len(pq)):
                  sum_v = np.absolute(g[pq[k][0]])**2*np.absolute(g[pq[k][1]])**2 + sum_v
              E[i,i] = sum_v

          #CONSTRUCTING D
          #**************
          D = np.zeros((self.N,self.L),dtype=complex)
          for i in xrange(self.N):
              for j in xrange(self.L):
                  psi,found = psi_func_eval(i,j+1,self.phi)
                  if found:
                     D[i,j] = g[i]*y[j].conj()*np.absolute(g[psi])**2 

          DH = D.transpose().conj()  

          #CONSTRUCTING F
          #**************
          F = np.zeros((self.N,self.N),dtype=complex)
          for i in xrange(self.N):
              for j in xrange(i,self.N):
                  if i <> j:
                     F[i,j] = g[i]*g[j]*np.absolute(y[self.phi[i,j]-1])**2
                     F[j,i] = F[i,j] 

          #CONSTRUCTING G
          #**************
          G = np.zeros((self.N,self.L),dtype=complex)
          for i in xrange(self.N):
              for j in xrange(self.L):
                  xi,found = xi_func_eval(i,j+1,self.phi)
                  
                  if found:
                     #print "i","j","xi",i+1,j+1,xi+1
                     #print "found",found
                     G[i,j] = g[i]*y[j]*np.absolute(g[xi])**2
                  #else:
                  #   print "i","j","xi",i+1,j+1,xi
                  #   print "found",found  

          GT = G.transpose() 

          #CONSTRUCTING Z
          #**************
          Z = np.zeros((self.L,self.L),dtype=complex)

          #CONSTRUCTING A
          #**************1
          A = np.zeros((self.N+self.L,self.N+self.L),dtype=complex)
          A[:self.N,:self.N] = C
          A[:self.N,self.N:] = D
          A[self.N:,self.N:] = E
          A[self.N:,:self.N] = DH

          #CONSTRUCTING B
          #**************
          B = np.zeros((self.N+self.L,self.N+self.L),dtype=complex)
          B[:self.N,:self.N] = F
          B[:self.N,self.N:] = G
          B[self.N:,self.N:] = Z
          B[self.N:,:self.N] = GT

          #CONSTRUCTING H
          #**************
          H = np.zeros((2*(self.N+self.L),2*(self.N+self.L)),dtype=complex)
          H[:self.N+self.L,:self.N+self.L] = A
          H[:self.N+self.L,self.N+self.L:] = B
          H[self.N+self.L:,:self.N+self.L] = B.conj()
          H[self.N+self.L:,self.N+self.L:] = A.conj()

          H_lam = H + lam*np.diag(np.diag(H))

          H_D = np.diag(np.diag(H_lam)**(-1))

          H_sparse = csr_matrix(H_lam,dtype=complex) #THIS MIGHT NOT BE THE FASTEST WAY - MAY NEED TO TWEAK THIS 
          H_D_sparse = dia_matrix(H_D,dtype=complex)

          return H_lam,H_sparse,H_D,H_D_sparse
           
def do_sparc_experiment(SNR=5,method="PCG",min_order=1,max_order=3,layout="HEX",time_var=False,exp_num=5):
    for e in xrange(exp_num):
 
    	order_vec = np.arange(min_order,max_order+1)
        dir_name = layout+"_"+method+"_"+str(SNR)+"_"+str(time_var)+"_"+str(e)
    
        if not os.path.isdir("./"+dir_name): 
           os.system("mkdir "+dir_name)

        for k in xrange(len(order_vec)):
            print "*********"
            print "e = ",e
            print "k = ",k
            print "*********"
            s = simulator.sim(nsteps=50,layout=layout,order=order_vec[k],seed=e) #INSTANTIATE OBJECT
            s.generate_antenna_layout()
            phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
            PQ = s.create_PQ(phi,s.L)
            s.uv_tracks()
            point_sources = s.create_point_sources(100,fov=3,a=2)
            g=s.create_antenna_gains(s.N,0.9,0.8,10,1,5,s.nsteps,plot = False)
            D,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=SNR,w_m=None)
            M,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=None,w_m=None) #PREDICTED VIS
            sparc_object = SPARC(s.N,s.L,phi,zeta,PQ)
            z_cal,c_cal,G_cal,M_cal,time_mat,outer_loop=sparc_object.levenberg_marquardt_time  (D,s.psi_func_eval,s.xi_func_eval,s.convert_y_to_M,tol1=1e-6,tol2=1e-6,tol3=1e-15,lam=2,max_itr=10000,method=method,time_var=time_var)
       
            file_name = "./"+dir_name+"/"+str(order_vec[k])+"_"+str(s.N)+"_"+str(s.L)+"_"+dir_name+".p"

            output = open(file_name, 'wb')
            pickle.dump(order_vec[k], output) 
            pickle.dump(s.N, output) 
            pickle.dump(s.L, output) 
            pickle.dump(zeta, output)
            pickle.dump(PQ,output)
            pickle.dump(z_cal,output)        
            pickle.dump(c_cal,output)
            pickle.dump(time_mat,output)
            pickle.dump(outer_loop,output)
            pickle.dump(sparc_object.itr_vec,output)
            pickle.dump(sparc_object.kappa_vec,output)
            pickle.dump(G_cal,output)
            pickle.dump(M_cal,output)
            pickle.dump(D,output)
            pickle.dump(M,output)
            output.close()

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    #kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    kde = gaussian_kde(x)
    return kde.evaluate(x_grid)

def plot_kappa_itr():
    #file_name = "./HEX_PCG_1000_False/5_91_165_HEX_PCG_1000_False.p"
    #file_name = "./HEX_PCG_1000_False/1_7_9_HEX_PCG_1000_False.p"
    
    #file_name = "./HEX_CG_1000_False/5_91_165_HEX_CG_1000_False.p"
    #file_name = "./HEX_CG_1000_False/1_7_9_HEX_CG_1000_False.p"
    file_name = "./HEX_PCG_1000_False/2_19_30_HEX_PCG_1000_False.p"
    
    file_p = open(file_name, 'rb')
    order = pickle.load(file_p)
    N = pickle.load(file_p)
    L = pickle.load(file_p)
    zeta = pickle.load(file_p)
    PQ = pickle.load(file_p)
    z_cal = pickle.load(file_p)
    c_cal = pickle.load(file_p)
    time_mat = pickle.load(file_p)
    outer_loop = pickle.load(file_p)
    itr_vec = pickle.load(file_p)
    kappa_vec = pickle.load(file_p)
    
    print "len(itr_vec) = ",len(itr_vec)
    print "len(kappa_vec) = ",len(kappa_vec)
    print "mean(itr_vec) = ",np.mean(itr_vec)
    print "std(itr_vec) = ",np.std(itr_vec)
    print "mean(kappa_vec) = ",np.mean(kappa_vec)
    print "std(kappa_vec) = ",np.std(kappa_vec)
    print "amax(kappa_vec) = ",np.amax(kappa_vec)
    print "amin(kappa_vec) = ",np.amin(kappa_vec)
    print "median(kappa_vec) = ",np.median(kappa_vec)
    print "mad = ",np.median(np.absolute(kappa_vec - np.median(kappa_vec)))
    print "c_cal = ",c_cal

    x_grid = np.linspace(0,200000,1000)
 
    y = kde_scipy(kappa_vec[kappa_vec<200000], x_grid)
    
    plt.plot(x_grid,y)
    plt.show()

    #n, bins, patches = plt.hist(kappa_vec[kappa_vec<100000], 200, normed=1, facecolor='green', alpha=0.75)
    #plt.show()

    G_cal = pickle.load(file_p)
    M_cal = pickle.load(file_p)
    D = pickle.load(file_p)
    M = pickle.load(file_p)
    file_p.close()


def main(argv):
    snr = 1000
    l = "HEX"
    min_order = 1 
    max_order = 2 
    m = "PCG"
    time_var = False
    exp_num = 5

    try:
       opts, args = getopt.getopt(argv,"htl:s:m:e:",["minorder=","maxorder="])
    except getopt.GetoptError:
       print 'sparc.py -t -l <layout> -s <SNR> -m <method> -e <exp_number> --minorder <minorder> --maxorder <maxorder>'
       print 'Does a sparc experiment'
       print '-t : set timing to True. False (default)'
       print '-l <layout> : i.e. HEX, REG or SQR. HEX (default)'
       print '-m <method> : i.e. PCG or CG. PCG (default)'
       print '-s <SNR> : signal-to-noise ratio. 1000 (default).'
       print '-e <exp_num> : number of experiments to perform . 5 (default)'
       print '-- min_order : minimum order of redundant array. 1 default.'
       print '-- max_order : maximum order of redundant array. 3 default.'
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print 'sparc.py -l <layout> -s <SNR> -m <method> -e <exp_number> --minorder <minorder> --maxorder <maxorder>'
          print 'Does a sparc experiment'
          print '-t : set timing to True. False (default)'
          print '-l <layout> : i.e. HEX, REG or SQR. HEX (default)'
          print '-m <method> : i.e. PCG or CG. PCG (default)'
          print '-s <SNR> : signal-to-noise ratio. 1000 (default).'
          print '-e <exp_num> : number of experiments to perform . 5 (default)'
          print '-- min_order : minimum order of redundant array. 1 default.'
          print '-- max_order : maximum order of redundant array. 3 default.'
          sys.exit()
       elif opt in ("-l"):
          l = arg
       elif opt in ("-t"):
          time_var = True
       elif opt in ("-s"):
          snr = int(arg)
       elif opt in ("-m"):
          m = arg
       elif opt in ("-m"):
          exp_num = int(arg)
       elif opt in ("--minorder"):
          min_order = int(arg)
       elif opt in ("--maxorder"):
          max_order = int(arg)
  
    print "OPTIONS SELECTED:"
    print "*****************"
    print 'layout: ', l
    print 'SNR: ', snr
    print 'method: ',m
    print 'maxorder: ',max_order
    print 'minorder: ',min_order
    print 't: ',time_var
    print 'exp_num: ', exp_num
    print "*****************"

    return snr,l,m,max_order,min_order,time_var,exp_num

if __name__ == "__main__":
   snr,l,m,max_order,min_order,time_var,exp_num = main(sys.argv[1:])
   do_sparc_experiment(SNR=snr,method=m,min_order=min_order,max_order=max_order,layout=l,time_var=time_var,exp_num=exp_num)
   #plot_kappa_itr()
   '''
   s = simulator.sim(nsteps=100,layout="HEX",order=1) #INSTANTIATE OBJECT
   #s.read_antenna_layout()
   s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
   s.plot_ant(title="HEX") #PLOT THE LAYOUT
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   PQ = s.create_PQ(phi,s.L)
   s.uv_tracks() #GENERATE UV TRACKS
   #s.plot_uv_coverage(title="HEX") #PLOT THE UV TRACKS
   point_sources = s.create_point_sources(100,fov=3,a=2) #GENERATE RANDOM SKYMODEL
   g=s.create_antenna_gains(s.N,0.9,0.1,50,1,5,s.nsteps,plot = True) #GENERATE GAINS
   D,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=5,w_m=None) #CREATE VIS MATRIX
   M,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=None,w_m=None) #PREDICTED VIS
   s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   s.plot_visibilities([0,1],M,"r",s=True)    
   
   sparc_object = SPARC(s.N,s.L,phi,zeta,PQ)
   z_cal,c_cal,G_cal,M_cal=sparc_object.levenberg_marquardt_time(D,s.psi_func_eval,s.xi_func_eval,s.convert_y_to_M,tol1=1e-6,tol2=1e-6,tol3=1e-15,lam=2,max_itr=5000,method="CG")
   s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   s.plot_visibilities([0,1],M,"r",s=False)    
   s.plot_visibilities([0,1],G_cal*M_cal,"g",s=True)
   #s.plot_visibilities([0,1],M_cal,"c",s=True)
   print "sparc_object.itr_vec = ",sparc_object.itr_vec
   print "sparc_object.kappa_vec = ",sparc_object.kappa_vec   
   #import IPython
   #IPython.embed()
   '''
   '''
   g = np.random.randn(s.N)+1j*np.random.randn(s.N)
   y = np.random.randn(s.L)+1j*np.random.randn(s.L)

   z = np.hstack([g,y])

   print "z = ",z

   sparc_object = SPARC(s.N,s.L,phi,zeta,PQ)
   v = sparc_object.compute_v(z)
   J = sparc_object.compute_J(z)
   H_comp = sparc_object.compute_JHJ(J)
   H,H_sparse,D_l,D_sparse = sparc_object.generate_H_formula(z,s.psi_func_eval,s.xi_func_eval,lam=0)
   
   d = sparc_object.vectorize_D(D[:,:,0])
   
   r = sparc_object.compute_r(d,v)
   
   JHr_com = sparc_object.compute_JHr(J,r)
  
   R = sparc_object.construct_R(r)
   
   JHr_fun = sparc_object.compute_JHr_formula(z,R) #CAN STILL INCREASE THE SPEED BY DIRECTLY COMPUTING R form D and V

   dz,info = sparc_object.compute_inverse_cg(A=H_sparse,b=JHr_fun,M=D_sparse,tol=1e-6)
   H_inv = pinv(H)
   dz_p = np.dot(H_inv,JHr_fun)

   

   p1 = np.dot(H,dz)
   p2 = np.dot(H,dz_p)
   

   print "JHr_com = ",JHr_com
   print "len(JHr_com) = ", len(JHr_com)
   print "JHr_fun = ",JHr_fun
   print "len(JHr_fun) = ", len(JHr_fun)
   
         
   print "v = ",v
   plt.imshow(np.absolute(J),interpolation="nearest")
   plt.show()   

   plt.imshow(np.absolute(H_comp),interpolation="nearest")
   plt.show()

   plt.imshow(np.absolute(H),interpolation="nearest")
   plt.show()

   plt.imshow(np.absolute(H-H_comp),interpolation="nearest")
   plt.colorbar()
   plt.show()

   print "dz = ",dz
   print "info = ",info
   print "dz_p = ",dz_p

   print "itr = ",sparc_object.itr_vec   

   print "p1 = ",p1
   print "p2 = ",p2

   print "p1 - p2 = ",p1-p2

   print "dz - dz_p = ",dz-dz_p 
   '''   

