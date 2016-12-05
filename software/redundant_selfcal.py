import numpy as np
import scipy as sp
import pylab as plt
import simulator
import analytic
import time
import sys, getopt
import os
import pickle
import matplotlib as mpl

def redundant_StEFCal(D,phi,tau=1e-3,alpha=0.3,max_itr=1000,PQ=None):
    converged = False
    N = D.shape[0]
    D = D - D*np.eye(N)
    L = np.amax(phi)
    temp =np.ones((D.shape[0],D.shape[1]) ,dtype=complex)
    
    g_temp = np.ones((N,),dtype=complex)
    y_temp = np.ones((L,),dtype=complex)
    z_temp = np.hstack([g_temp,y_temp])

    error_vector = np.array([])

    #EXTRACT BASELINE INDICES FOR EACH REDUNDANT SPACING
    if PQ is not None:
       PQ = create_PQ(phi,L)

    start = time.time() 
    
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
      
        

        err = np.sqrt(np.sum(np.absolute(z_temp-z_old)**2))/np.sqrt(np.sum(np.absolute(z_temp)**2))

        error_vector = np.append(error_vector,np.array([err]))

        if (err <= tau):
           converged = True 
           break

    #print "i = ",i
    #print "norm = ",np.sqrt(np.sum(np.absolute(z_temp-z_old)**2))/np.sqrt(np.sum(np.absolute(z_temp)**2))
    stop = time.time()
    G = np.dot(np.diag(g_temp),temp)
    G = np.dot(G,np.diag(g_temp.conj()))  
    M = convert_y_to_M(PQ,y_temp,N)         

    return z_temp,converged,G,M,start,stop,i,error_vector

def redundant_StEFCal_time(D,phi,tau=1e-6,alpha=1./3,max_itr=10000):
    N = D.shape[0]
    L = np.amax(phi)
    PQ = create_PQ(phi,L) 
    z_temp = np.zeros((N+L,D.shape[2]),dtype=complex)
    M = np.zeros((N,N,D.shape[2]),dtype=complex)
    G = np.zeros((N,N,D.shape[2]),dtype=complex)
    c_temp = np.zeros((D.shape[2],),dtype=bool)
    t_temp = np.zeros((2,D.shape[2]))
    count_temp = np.zeros((D.shape[2],),dtype=int)
    error = {}
    for t in xrange(D.shape[2]):
        print "*********"
        print "t = ",t
        print "*********"
        z_temp[:,t],c_temp[t],G[:,:,t],M[:,:,t],t_temp[0,t],t_temp[1,t],count_temp[t],error[str(t)] = redundant_StEFCal(D=D[:,:,t],phi=phi,tau=tau,alpha=alpha,max_itr=max_itr,PQ=PQ)
        #print "c_temp = ",c_temp[t]  
    return z_temp,c_temp,G,M,t_temp,count_temp,error
             
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

#type_exp: G_OLD - old gain with real and imag sinusoids
#type_exp: G - ant gain with amp and phase sinusoids
#type_exp: F - phase slope

def do_red_cal_experiment(SNR=5,min_order=1,max_order=2,layout="HEX",exp_number=5,type_exp="G_OLD",freq_enabled=False):
    order_vec = np.arange(min_order,max_order+1)
    for e in xrange(exp_number):
        
        method = "R_StEFCal"
        if freq_enabled:
           dir_name = layout+"_"+method+"_"+str(SNR)+"_"+str(e)+"_"+type_exp+"_f"
        else:
           dir_name = layout+"_"+method+"_"+str(SNR)+"_"+str(e)+"_"+type_exp
    
        if not os.path.isdir("./"+dir_name): 
           os.system("mkdir "+dir_name)

        for k in xrange(len(order_vec)):
            print "*********"
            print "e = ",e
            print "k = ",k
            print "*********"
            nsteps = 50
            s = simulator.sim(nsteps=nsteps,layout=layout,order=order_vec[k],seed=e) #INSTANTIATE OBJECT
            s.generate_antenna_layout()
            phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
            s.uv_tracks()
            #PQ = s.create_PQ(phi,s.L)
            point_sources = s.create_point_sources(100,fov=3,a=2)
            
            if freq_enabled: #DO A FREQ_DOMAIN EXPERIMENT 
               nsteps = 1024
               s.create_uv_f(num_channels = nsteps)
               u_temp = s.u_f
               v_temp = s.v_f
               w_temp = s.w_f
            else: #DO A TIME_DOMAIN EXPERIMENT
               u_temp = s.u_m
               v_temp = s.v_m
               w_temp = s.w_m
                        
            #CHOOSE THE TYPE OF GAIN ERROR TO ADD
            if type_exp == "F":
               g = s.generate_phase_slope_gains(num_channels=nsteps)
            elif type_exp == "G":
               g = s.create_antenna_gains(s.N,0.9,0.8,10,1,5,nsteps,plot=False)
            else:
               g = s.create_antenna_gains_old(s.N,10,0.5,5,nsteps,plot = False)

            D,sig = s.create_vis_mat(point_sources,u_temp,v_temp,g=g,SNR=SNR,w_m=None)
            M,sig = s.create_vis_mat(point_sources,u_temp,v_temp,g=g,SNR=None,w_m=None) #PREDICTED VIS
            z_cal,c_cal,G_cal,M_cal,t,outer_loop,error = redundant_StEFCal_time(D,phi,tau=1e-6,alpha=1./3,max_itr=10000)
        
            #z_cal,c_cal,G_cal,M_cal,t,outer_loop=sparc_object.levenberg_marquardt_time(D,s.psi_func_eval,s.xi_func_eval,s.convert_y_to_M,tol1=1e-6,tol2=1e-6,tol3=1e-15,lam=2,max_itr=5000,method=method)
       
            file_name = "./"+dir_name+"/"+str(order_vec[k])+"_"+str(s.N)+"_"+str(s.L)+"_"+dir_name+".p"

            output = open(file_name, 'wb')
            pickle.dump(order_vec[k], output) 
            pickle.dump(s.N, output) 
            pickle.dump(s.L, output) 
            pickle.dump(zeta, output)
            #pickle.dump(PQ,output)
            pickle.dump(z_cal,output)        
            pickle.dump(c_cal,output)
            pickle.dump(t,output)
            pickle.dump(outer_loop,output)
            #pickle.dump(sparc_object.itr_vec,output)
            #pickle.dump(sparc_object.kappa_vec,output)
            pickle.dump(G_cal,output)
            pickle.dump(M_cal,output)
            pickle.dump(D,output)
            pickle.dump(M,output)
            pickle.dump(error,output)
            output.close()

def find_color_marker(index_value):
    color = np.array(['b','g','r','c','m','y','k'])
    marker = np.array(['o','v','^','<','>','8','s','p','*','H','+','x','D'])
    index_value = index_value-1
    counter = 0
    c_final = None
    m_final = None
    for k in xrange(len(marker)):
        for l in xrange(len(color)):
            if counter == index_value:
               c_final = color[l]
               m_final = marker[k]
               break
            counter = counter+1
        else:
            continue  # executed if the loop ended normally (no break)
        break  # executed if loop ended through break
    #print "c_final = ",c_final
    #print "m_final = ",m_final

    if c_final is None:
       i_c = np.random.randint(low=0,high=len(color))
       i_m = np.random.randint(low=0,high=len(marker)) 
       c_final = color[i_c]
       m_final = color[i_m]
    return c_final,m_final 

def plot_before_after_cal_per_t_f(D,G,phi,k):
        mpl.rcParams['xtick.labelsize'] = 20 
        mpl.rcParams['ytick.labelsize'] = 20 
        #mpl.rcParams['ylabel.labelsize'] = 22  
        #mpl.rcParams['xlabel.labelsize'] = 22 
        #mpl.rcParams['title.labelsize'] = 22
        D_new = D[:,:,k]
        G_new = G[:,:,k]
        D_cal = D_new*G_new**(-1)
        for i in xrange(D_new.shape[0]):
            for j in xrange(i+1,D_new.shape[1]):
                ind = phi[i,j]
                #print "ind = ",ind
                c_final,m_final = find_color_marker(ind)
                plt.plot(D_new[i,j].real,D_new[i,j].imag,c_final+m_final,ms=10)
        plt.axis("equal")
        plt.xlabel("real",fontsize=15)
        plt.ylabel("imag",fontsize=15)
        plt.title("Before Calibration",fontsize=15)
        plt.show()        

        for i in xrange(D_cal.shape[0]):
            for j in xrange(i+1,D_cal.shape[1]):
                ind = phi[i,j]
                c_final,m_final = find_color_marker(ind)
                plt.plot(D_cal[i,j].real,D_cal[i,j].imag,c_final+m_final,ms=10)
        plt.axis("equal")
        plt.ylabel("imag",fontsize=15)
        plt.xlabel("real",fontsize=15)
        plt.title("After Calibration",fontsize=15)
        plt.show()
                  

def main(argv):
    snr = 1000
    l = "HEX"
    min_order = 1 
    max_order = 2 
    exp_number = 5
    freq_enabled = False
    type_exp = "G_OLD"

    try:
       opts, args = getopt.getopt(argv,"hfl:s:e:",["minorder=","maxorder=","type_exp="])
    except getopt.GetoptError:
       print 'redundant_selfcal.py -l <layout> -s <SNR> -e <exp_number> --minorder <minorder> --maxorder <maxorder> --type_exp <typeexp>'
       print 'Does a redundant stefcal experiment'
       print '-f : Enable frequency simulation'
       print '-l <layout> : i.e. HEX (default), REG or SQR. HEX (default)'
       print '-s <SNR> : signal-to-noise ratio. 1000 (default).'
       print '-e <exp_number> : the experiment number. 5 (default).'
       print '--min_order : minimum order of redundant array. 1 default.'
       print '--max_order : maximum order of redundant array. 3 default.'
       print '--type_exp : G (phase, ampl sinusoids), G_OLD (real and imag sinusoids), F (phase_slope)'
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print 'redundant_selfcal.py -l <layout> -s <SNR> -e <exp_num> --minorder <minorder> --maxorder <maxorder> --type_exp <typeexp>'
          print 'Does a redundant stefcal experiment'
          print '-f : Enable frequency simulation'
          print '-l <layout> : i.e. HEX (default), REG or SQR. HEX (default)'
          print '-s <SNR> : signal-to-noise ratio. 1000 (default).'
          print '-e <exp_number> : the experiment number. 5 (default).'
          print '--min_order : minimum order of redundant array. 1 default.'
          print '--max_order : maximum order of redundant array. 3 default.'
          print '--type_exp : G (phase, ampl sinusoids), G_OLD (real and imag sinusoids), F (phase_slope)' 
          sys.exit()
       elif opt in ("-f"):
          freq_enabled = True     
       elif opt in ("-l"):
          l = arg
       elif opt in ("-s"):
          snr = int(arg)
       elif opt in ("-e"):
          exp_number = int(arg)
       elif opt in ("--minorder"):
          min_order = int(arg)
       elif opt in ("--maxorder"):
          max_order = int(arg)
       elif opt in ("--type_exp"):
          type_exp = arg 
  
    print "freq_enable: ", freq_enabled
    print 'layout: ', l
    print 'SNR: ', snr
    print 'maxorder: ',max_order
    print 'minorder: ',min_order
    print 'exp_number: ',exp_number
    print 'type_exp: ',type_exp    

    return freq_enabled,snr,l,max_order,min_order,exp_number,type_exp

if __name__ == "__main__":
   
   freq_enabled,snr,l,max_order,min_order,exp_number,type_exp = main(sys.argv[1:])
   do_red_cal_experiment(SNR=snr,min_order=min_order,max_order=max_order,layout=l,exp_number=exp_number,type_exp=type_exp,freq_enabled=freq_enabled)
   
   '''
   s = simulator.sim(nsteps=100,layout="HEX",order=2) #INSTANTIATE OBJECT
   #s.read_antenna_layout()
   s.generate_antenna_layout() #CREATE ANTENNA LAYOUT - DEFAULT IS HEXAGONAL
   s.plot_ant(title="HEX") #PLOT THE LAYOUT
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   
   s.plot_zeta(zeta,1,1,10,'cubehelix')
   s.uv_tracks() #GENERATE UV TRACKS
   #s.plot_uv_coverage(title="HEX") #PLOT THE UV TRACKS
   point_sources = s.create_point_sources(100,fov=3,a=2) #GENERATE RANDOM SKYMODEL
   #g=s.create_antenna_gains(s.N,0.9,0.1,50,1,5,s.nsteps,plot = True) #GENERATE GAINS
   s.create_uv_f() #GENERATE UV POINTS f (for one time-slot)
   s.plot_uv_f()
   g = s.generate_phase_slope_gains()
   #point_sources = np.array([(1,0,0)])
   D,sig = s.create_vis_mat(point_sources,s.u_f,s.v_f,g=g,SNR=15,w_m=None) #CREATE VIS MATRIX
   M,sig = s.create_vis_mat(point_sources,s.u_f,s.v_f,g=g,SNR=None,w_m=None) #PREDICTED VIS
   
   #s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   #s.plot_visibilities([0,1],M,"r",s=True)    
   z_cal,c_cal,G_cal,M_cal,t,count_temp,_ = redundant_StEFCal_time(D,phi)

   plot_before_after_cal_per_t_f(D,G_cal,phi,0)
   '''
   '''
   s.plot_visibilities([0,1],D,"b",s=False) #PLOT VIS
   s.plot_visibilities([0,1],M,"r",s=False)    
   s.plot_visibilities([0,1],G_cal*M_cal,"g",s=True)

   s.plot_visibilities([0,1],M_cal,"r",s=True)

   prec_error = np.zeros((G_cal.shape[2],),dtype=float)

   for k in xrange(M_cal.shape[2]):
         mask = np.ones(M[:,:,k].shape,dtype=float)-np.diag(np.ones((M[:,:,k].shape[0],),dtype=float))
         prec_error[k] = np.linalg.norm((M[:,:,k] - G_cal[:,:,k]*M_cal[:,:,k])*mask)**2/np.linalg.norm(M[:,:,k]*mask)**2 

   plt.plot(prec_error)

   plt.show()
   '''
   '''
   for n in xrange(s.N):
       plt.plot(np.angle(z_cal[n]))
   plt.show() 

   for n in xrange(s.N):
       plt.plot(np.absolute(z_cal[n]))
   plt.show() 

   yv = z_cal[s.N:]

   for n in xrange(len(yv)):
       plt.plot(np.angle(yv[n]))
   plt.show() 

   for n in xrange(len(yv)):
       plt.plot(np.absolute(yv[n]))
   plt.show() 

   avg_m = np.mean(np.absolute(yv),axis=0)

   plt.plot(avg_m)
   plt.show()

   for n in xrange(s.N):
       plt.plot(np.absolute(z_cal[n]/np.sqrt(avg_m)))
   plt.show()
   '''
   
   
    
   

