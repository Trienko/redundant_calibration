import numpy as np
import scipy as sp
import pylab as plt
import simulator
import analytic

def redundant_StEFCal(D,phi,tau=1e-3,alpha=0.3,max_itr=1000,PQ=None):
    converged = False
    N = D.shape[0]
    D = D - D*np.eye(N)
    L = np.amax(phi)
    temp =np.ones((D.shape[0],D.shape[1]) ,dtype=complex)
    
    g_temp = np.ones((N,),dtype=complex)
    y_temp = np.ones((L,),dtype=complex)
    z_temp = np.hstack([g_temp,y_temp])

    #EXTRACT BASELINE INDICES FOR EACH REDUNDANT SPACING
    if PQ is not None:
       PQ = create_PQ(phi,L)
    
    for i in xrange(max_itr): #MAX NUMBER OF ITRS
        g_old = np.copy(g_temp)
        y_old = np.copy(y_temp)
        z_old = np.copy(z_temp)
 
        M = convert_y_to_M(PQ,y_old,N) #CONVERT y VECTOR TO M matrix
    
        #plt.imshow(np.absolute(M))
        #plt.show()
        #from IPython import embed; embed()        
        
        for p in xrange(N): #STEFCAL - update antenna gains
            z = g_old*M[:,p]
            g_temp[p] = np.sum(np.conj(D[:,p])*z)/(np.sum(np.conj(z)*z))
        
        for l in xrange(L): #UPDATE y
            pq = PQ[str(l)]
            num = 0
            den = 0
            for k in xrange(len(pq)): #loop through all baselines for each redundant spacing
                p = pq[k][0]
                q = pq[k][1]

                num = num + np.conjugate(g_old[p])*g_old[q]*D[p,q]
                den = den + np.absolute(g_old[p])**2*np.absolute(g_old[q])**2
            y_temp[l] = num/den
         
        g_temp = alpha*g_temp + (1-alpha)*g_old
        y_temp = alpha*y_temp + (1-alpha)*y_old
        z_temp = np.hstack([g_temp,y_temp]) #final update
        #print "g_temp =",g_temp
        #print "y_temp =",y_temp        

        #e = np.sqrt(np.sum(np.absolute(z_temp-z_old)**2))/np.sqrt(np.sum(np.absolute(z_temp)**2))
        #print "e = ",e
      
        if (np.sqrt(np.sum(np.absolute(z_temp-z_old)**2))/np.sqrt(np.sum(np.absolute(z_temp)**2)) <= tau):
           converged = True 
           break

    G = np.dot(np.diag(g_temp),temp)
    G = np.dot(G,np.diag(g_temp.conj()))  
    M = convert_y_to_M(PQ,y_temp,N)         

    return z_temp,converged,G,M

def redundant_StEFCal_time(D,phi,tau=1e-3,alpha=0.3,max_itr=3000):
    N = D.shape[0]
    L = np.amax(phi)
    PQ = create_PQ(phi,L) 
    z_temp = np.zeros((N+L,D.shape[2]),dtype=complex)
    M = np.zeros((N,N,D.shape[2]),dtype=complex)
    G = np.zeros((N,N,D.shape[2]),dtype=complex)
    c_temp = np.zeros((D.shape[2],),dtype=bool)
    for t in xrange(D.shape[2]):
        print "t= ",t
        z_temp[:,t],c_temp[t],G[:,:,t],M[:,:,t] = redundant_StEFCal(D=D[:,:,t],phi=phi,tau=tau,alpha=alpha,max_itr=max_itr,PQ=PQ)
        print "c_temp = ",c_temp[t]  
    return z_temp,c_temp,G,M
             
def create_PQ(phi,L):

    PQ = {}
    for i in xrange(L):
        pq = zip(*np.where(phi == (i+1)))
        #print "pq2 = ",pq        
        PQ[str(i)]=pq
    #print "PQ = ",PQ    
    return PQ 

def convert_y_to_M(PQ,y,N):
    
    M = np.zeros((N,N),dtype=complex)
    for i in xrange(len(y)):
        pq = PQ[str(i)]
        #print "pq = ",pq
                          
        for k in xrange(len(pq)):
            p = pq[k][0]
            q = pq[k][1]
             
            M[p,q] = y[i]
            M[q,p] = np.conjugate(y[i])  
    #from IPython import embed; embed() 
    return M  
   
if __name__ == "__main__":
   s = simulator.sim() #INSTANTIATE OBJECT
   #s.read_antenna_layout()
   s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
   s.plot_ant(title="HEX") #PLOT THE LAYOUT
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   s.uv_tracks() #GENERATE UV TRACKS
   s.plot_uv_coverage(title="HEX") #PLOT THE UV TRACKS
   point_sources = s.create_point_sources(100,fov=3,a=2) #GENERATE RANDOM SKYMODEL
   g=s.create_antenna_gains(s.N,0.9,0.1,50,1,5,s.nsteps,plot = True) #GENERATE GAINS
   D,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=g,SNR=10,w_m=None) #CREATE VIS MATRIX
   M,sig = s.create_vis_mat(point_sources,s.u_m,s.v_m,g=None,SNR=None,w_m=None) #PREDICTED VIS
   s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   s.plot_visibilities([0,1],M,"r",s=True)    
   z_cal,c_cal,G_cal,M_cal = redundant_StEFCal_time(D,phi)
   s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   s.plot_visibilities([0,1],M,"r",s=False)    
   s.plot_visibilities([0,1],G_cal*M_cal,"g",s=True)
   
    
   

