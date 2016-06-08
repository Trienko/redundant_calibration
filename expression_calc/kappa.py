import numpy as np
import pylab as plt
import scipy as sp
import H_gen

class redundant_calibration():
      def __init__(self):
          self.H_symbols = np.array([])
          pass

      def compute_kappa(self,H_num):
          #ones = np.ones((H.shape[0],),dtype=float)
          #I = np.diag(ones)
          d = np.diag(H_num)
          d = d**(-1)
          D = np.diag(d)
          w = np.linalg.eigvals(np.dot(D,H_num))
          lam_max = np.amax(w.real[w.real>1e-6])
          lam_min = np.min(w.real[w.real>1e-6])
          print "lam_max = ",lam_max
          print "lam_min = ",lam_min
          print "w = ",w
          return lam_max/lam_min,lam_min,lam_max
 
      def computeH(self,z,N):
          g = z[:N]
          y = z[N:] 
          print "len(g) = ",len(g)
          print "len(y) = ",len(y)      
          r = H_gen.redundant(N)
          r.create_regular()
          r.create_J1()
          r.create_J2()
          r.conjugate_J1_J2()
          r.create_J()
          r.hermitian_transpose_J()
          r.compute_H()
          H_int = r.to_int_H()
          #plt.imshow(H_int,interpolation="nearest")
          #plt.show()
          H_num = r.substitute_H(g,y)
          #plt.imshow(np.real(H_num),interpolation="nearest")
          #plt.show()
          return H_num,H_int


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

      def create_uv_regular_line(self,spacing_in_meters=20,antennas=5,lam=0.21428571428571427,dec=60,H_min=-6,H_max=6,time_steps=600,plot=False):
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

def kappa_experiment(sources_k,N,sig):
          
          point_sources = np.zeros((sources_k,3),dtype=float)
          g = np.zeros((N,),dtype=complex)
          g_un = np.ones((N,))
          for k in xrange(sources_k):
              point_sources[k,0] = np.random.uniform(low=0.001,high=10)
              point_sources[k,1] = np.random.uniform(low=-5,high=5)*(np.pi/180)
              point_sources[k,2] = np.random.uniform(low=-5,high=5)*(np.pi/180)
           
          red_object = redundant_calibration()
          u_m,v_m, b_m = red_object.create_uv_regular_line(antennas=N)
          
          for k in xrange(len(g)):
              g[k] = (sig*np.random.randn()+1) + sig*np.random.randn()*1j

          D = red_object.create_vis_mat(point_sources,u_m,v_m,g,sig)
          M = red_object.create_vis_mat(point_sources,u_m,v_m,g_un,0)

          M_temp = (M[:,:,0] + sig*np.random.randn(M.shape[0],M.shape[1]) + sig*np.random.randn(M.shape[0],M.shape[1])*1j) 
          y_0 = np.zeros((N-1,),dtype=complex)
          for k in xrange(1,N):
              y_0[k-1] = np.mean(np.diag(M_temp,k))
              
          g_0 = np.ones((N,),dtype=complex)  
          z_0 = np.hstack([g_0,y_0])

          H_num,H_int = red_object.computeH(z_0,N)

          kappa,lam_min,lam_max = red_object.compute_kappa(H_num)  
          
          print "kappa = ",kappa 
          return kappa,lam_min,lam_max


if __name__ == "__main__":
   N_v = np.arange(5,20)
   kappa_v = np.zeros(N_v.shape)
   lam_min_v = np.zeros(N_v.shape)
   lam_max_v = np.zeros(N_v.shape)

   for j in xrange(len(N_v)):          
       print "j = ",j
       kappa_v[j],lam_min_v[j],lam_max_v[j] = kappa_experiment(200,N_v[j],1)

   plt.plot(N_v,kappa_v)
   plt.show()
   plt.plot(N_v,lam_min_v)
   plt.plot(N_v,lam_max_v)
   plt.show()

   #N=12
   #red_object = redundant_calibration()
   #u_m,v_m, b_m = red_object.create_uv_regular_line(antennas=12)
   #point_sources = np.array([(3,0,0),(15,(1*np.pi)/180,(0*np.pi)/180),(1.5,(1*np.pi)/180,(-1*np.pi)/180),(1.5,(-1*np.pi)/180,(0*np.pi)/180),(2.5,(-2*np.pi)/180,(0*np.pi)/180),(10.5,(-1*np.pi)/180,(-2*np.pi)/180),(10.5,(-4*np.pi)/180,(-2*np.pi)/180),(20.5,(-5*np.pi)/180,(-2*np.pi)/180),(2.5,(-np.sqrt(5)*np.pi)/180,(-2*np.pi)/180),(0.5,(-5*np.pi)/180,(-np.sqrt(2)*np.pi)/180),(5,(-7*np.pi)/180,(-2.5*np.pi)/180),(20.5,(-np.sqrt(5,889)*np.pi)/180,(-3*np.pi)/180),(2.5,(-1*np.pi)/180,(-10*np.pi)/180),(2.5,(-5.7*np.pi)/180,(-4*np.pi)/180),(2.5,(-5.9*np.pi)/180,(-7*np.pi)/180),(2.5,(-5*np.pi)/180,(-3*np.pi)/180),(20.5,(-7*np.pi)/180,(-9*np.pi)/180),(20.5,(-1*np.pi)/180,(-7*np.pi)/180)]) #l and m are measured in radians
   #point_sources = np.array([(1,0,0),(1.5,(1*np.pi)/180,(0*np.pi)/180)])
   #g = np.array([2.+0.03j,1.007-0.8j,1.009,3.-0.8004j,1.1,1.3,1.4,4j,1.+3j,5.3,1j,10j])
   #sig = 0.1
   #D = red_object.create_vis_mat(point_sources,u_m,v_m,g,sig) #we corrupt our data and we add noise
   #g_un = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
   #M = red_object.create_vis_mat(point_sources,u_m,v_m,g_un,0)
   #M_temp = (M[:,:,0] + sig*np.random.randn(M.shape[0],M.shape[1]) + sig*np.random.randn(M.shape[0],M.shape[1])*1j) 
   #y_0 = np.zeros((N-1,),dtype=complex)
   #for k in xrange(1,N):
   #    y_0[k-1] = np.mean(np.diag(M_temp,k))
              
   #g_0 = np.ones((N,),dtype=complex)  
   #z_0 = np.hstack([g_0,y_0])

   #H_num,H_int = red_object.computeH(z_0,N)

   #kappa = red_object.compute_kappa(H_num)           
   #print "kappa = ",kappa
