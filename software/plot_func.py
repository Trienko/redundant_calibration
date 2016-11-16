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
import matplotlib

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    #kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    kde = gaussian_kde(x)
    return kde.evaluate(x_grid)

def plot_kappa_itr(SNR=1000,k_upper1=4,k_upper2=4):
    
    N = np.array([7,19,37,61,91,108,169,217])

    kappa_pcg_mean = np.zeros((len(N),),dtype=float)
    itr_pcg_mean = np.zeros((len(N),),dtype=float)
    kappa_cg_median = np.zeros((len(N),),dtype=float)
    itr_cg_median = np.zeros((len(N),),dtype=float)

    kappa_pcg_std = np.zeros((len(N),),dtype=float)
    itr_pcg_std = np.zeros((len(N),),dtype=float)
    kappa_cg_mad = np.zeros((len(N),),dtype=float)
    itr_cg_mad = np.zeros((len(N),),dtype=float)

    kappa_pcg_dic = {}
    kappa_cg_dic = {}
    itr_pcg_dic = {}
    itr_cg_dic = {}

    for j in xrange(len(N)):
        kappa_pcg_dic[str(N[j])]=np.array([],dtype=float)
        kappa_cg_dic[str(N[j])]=np.array([],dtype=float)
        itr_pcg_dic[str(N[j])]=np.array([],dtype=float) 
        itr_cg_dic[str(N[j])]=np.array([],dtype=float)     

    print "k = ",kappa_pcg_dic[str(N[0])]    

    for k in xrange(k_upper1):
        PCG_DIR = "./HEX_PCG_"+str(SNR)+"_False_"+str(k)
        PCG_FILE_LIST = np.array([PCG_DIR+"/1_7_9_"+PCG_DIR[2:]+".p",PCG_DIR+"/2_19_30_"+PCG_DIR[2:]+".p",PCG_DIR+"/3_37_63_"+PCG_DIR[2:]+".p",PCG_DIR+"/4_61_108_"+PCG_DIR[2:]+".p",PCG_DIR+"/5_91_165_"+PCG_DIR[2:]+".p",PCG_DIR+"/6_127_234_"+PCG_DIR[2:]+".p",PCG_DIR+"/7_169_315_"+PCG_DIR[2:]+".p",PCG_DIR+"/8_217_408_"+PCG_DIR[2:]+".p"]) 
        
        for i in xrange(len(PCG_FILE_LIST)):
            #LOAD PCG
            print "PCG_FILE_LIST = ",PCG_FILE_LIST[i]
            if not os.path.isfile(PCG_FILE_LIST[i]):
               break
            file_p = open(PCG_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            PQ = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            #print "outer_loop.shape = ",outer_loop.shape
            itr_vec = pickle.load(file_p)
            kappa_vec = pickle.load(file_p)
            kappa_pcg_dic[str(N[i])] = np.append(kappa_pcg_dic[str(N[i])],kappa_vec)
            itr_pcg_dic[str(N[i])] = np.append(itr_pcg_dic[str(N[i])],itr_vec)
            file_p.close()
        

    for k in xrange(k_upper2):
        CG_DIR = "./HEX_CG_"+str(SNR)+"_False_"+str(k)
        CG_FILE_LIST = np.array([CG_DIR+"/1_7_9_"+CG_DIR[2:]+".p",CG_DIR+"/2_19_30_"+CG_DIR[2:]+".p",CG_DIR+"/3_37_63_"+CG_DIR[2:]+".p",CG_DIR+"/4_61_108_"+CG_DIR[2:]+".p",CG_DIR+"/5_91_165_"+CG_DIR[2:]+".p",CG_DIR+"/6_127_234_"+CG_DIR[2:]+".p",CG_DIR+"/7_169_315_"+CG_DIR[2:]+".p",CG_DIR+"/8_217_408_"+CG_DIR[2:]+".p"])
            #LOAD CG

        for i in xrange(len(CG_FILE_LIST)):
              print "CG_FILE_LIST = ",CG_FILE_LIST[i]
              if not os.path.isfile(CG_FILE_LIST[i]):
                 break
              file_p = open(CG_FILE_LIST[i], 'rb')
              order = pickle.load(file_p)
              N_v = pickle.load(file_p)
              L = pickle.load(file_p)
              zeta = pickle.load(file_p)
              PQ = pickle.load(file_p)
              z_cal = pickle.load(file_p)
              c_cal = pickle.load(file_p)
              time_mat = pickle.load(file_p)
              outer_loop = pickle.load(file_p)
              itr_vec = pickle.load(file_p)
              kappa_vec = pickle.load(file_p)
              print "s = ",str(N[i])
              t_kappa = kappa_cg_dic[str(N[i])]
              t_itr = itr_cg_dic[str(N[i])]
              kappa_cg_dic[str(N[i])] = np.append(t_kappa,kappa_vec)
              itr_cg_dic[str(N[i])] = np.append(t_itr,itr_vec)
              file_p.close()
    
    for n in xrange(len(N)):
        if len(kappa_pcg_dic[str(N[n])]) == 0:
           break 
        kappa_pcg_mean[n] = np.mean(kappa_pcg_dic[str(N[n])])
        itr_pcg_mean[n] = np.mean(itr_pcg_dic[str(N[n])])
        if len(kappa_cg_dic[str(N[n])]) == 0:
           break
        kappa_cg_median[n] = np.median(kappa_cg_dic[str(N[n])])
        itr_cg_median[n] = np.median(itr_cg_dic[str(N[n])])

        kappa_pcg_std[n] = np.std(kappa_pcg_dic[str(N[n])])
        itr_pcg_std[n] = np.std(itr_pcg_dic[str(N[n])])
        kappa_cg_mad[n] = np.median(np.absolute(kappa_cg_dic[str(N[n])] - np.median(kappa_cg_dic[str(N[n])])))
        itr_cg_mad[n] = np.median(np.absolute(itr_cg_dic[str(N[n])] - np.median(itr_cg_dic[str(N[n])])))
    

    #print "kappa_cg_mad = ",kappa_cg_mad
    #print "itr_cg_mad = ",itr_cg_mad

    #print "N = ",N
    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(N,itr_cg_median,"r",lw=2,label="CG")
    ax.fill_between(N, itr_cg_median-itr_cg_mad, itr_cg_median+itr_cg_mad,alpha=0.2, edgecolor='k', facecolor='r')
    ax.plot(N,itr_pcg_mean,"b",lw=2,label="PCG")
    ax.fill_between(N, itr_pcg_mean-itr_pcg_std, itr_pcg_mean+itr_pcg_std,alpha=0.2, edgecolor='k', facecolor='b')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Iterations required by CG/PCG')
    ax.legend(loc=5)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(N,kappa_cg_median,"r",lw=2,label="CG")
    ax.fill_between(N, kappa_cg_median-kappa_cg_mad, kappa_cg_median+kappa_cg_mad,alpha=0.2, edgecolor='k', facecolor='r')
    ax.plot(N,kappa_pcg_mean,"b",lw=2,label="PCG")
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$\kappa$')
    ax.legend(loc=5)
    plt.show()
    '''
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
    '''

def plot_outer_loop(SNR=10,k_upper1=5,k_upper2=5):
    
    N = np.array([7,19,37,61,91,108,169,217])

    outerloop_pcg_mean = np.zeros((len(N),),dtype=float)
    outerloop_stef_mean = np.zeros((len(N),),dtype=float)

    outerloop_pcg_std = np.zeros((len(N),),dtype=float)
    outerloop_stef_std = np.zeros((len(N),),dtype=float)
    
    outerloop_pcg_dic = {}
    outerloop_stef_dic = {}
    
    for j in xrange(len(N)):
        outerloop_pcg_dic[str(N[j])]=np.array([],dtype=float)
        outerloop_stef_dic[str(N[j])]=np.array([],dtype=float)

    for k in xrange(k_upper1):
        PCG_DIR = "./HEX_PCG_"+str(SNR)+"_False_"+str(k)
        PCG_FILE_LIST = np.array([PCG_DIR+"/1_7_9_"+PCG_DIR[2:]+".p",PCG_DIR+"/2_19_30_"+PCG_DIR[2:]+".p",PCG_DIR+"/3_37_63_"+PCG_DIR[2:]+".p",PCG_DIR+"/4_61_108_"+PCG_DIR[2:]+".p",PCG_DIR+"/5_91_165_"+PCG_DIR[2:]+".p",PCG_DIR+"/6_127_234_"+PCG_DIR[2:]+".p",PCG_DIR+"/7_169_315_"+PCG_DIR[2:]+".p",PCG_DIR+"/8_217_408_"+PCG_DIR[2:]+".p"]) 
        
        for i in xrange(len(PCG_FILE_LIST)):
            #LOAD PCG
            print "PCG_FILE_LIST = ",PCG_FILE_LIST[i]
            if not os.path.isfile(PCG_FILE_LIST[i]):
               break
            file_p = open(PCG_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            PQ = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            print "len(outer_loop) = ",len(outer_loop)
            outerloop_pcg_dic[str(N[i])] = np.append(outerloop_pcg_dic[str(N[i])],outer_loop)
            print "len(outerloop_pcg_dic[str(N[i])]) = ",len(outerloop_pcg_dic[str(N[i])])
            file_p.close()
    
    for k in xrange(k_upper2):
        STEF_DIR = "./HEX_R_StEFCal_"+str(SNR)+"_"+str(k)
        STEF_FILE_LIST = np.array([STEF_DIR+"/1_7_9_"+STEF_DIR[2:]+".p",STEF_DIR+"/2_19_30_"+STEF_DIR[2:]+".p",STEF_DIR+"/3_37_63_"+STEF_DIR[2:]+".p",STEF_DIR+"/4_61_108_"+STEF_DIR[2:]+".p",STEF_DIR+"/5_91_165_"+STEF_DIR[2:]+".p",STEF_DIR+"/6_127_234_"+STEF_DIR[2:]+".p",STEF_DIR+"/7_169_315_"+STEF_DIR[2:]+".p",STEF_DIR+"/8_217_408_"+STEF_DIR[2:]+".p"])
        for i in xrange(len(STEF_FILE_LIST)):
            #LOAD STEF
            print "STEF_FILE_LIST = ",STEF_FILE_LIST[i]
            if not os.path.isfile(STEF_FILE_LIST[i]):
               break
            file_p = open(STEF_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            print "len(outer_loop) = ",len(outer_loop) 
            outerloop_stef_dic[str(N[i])] = np.append(outerloop_stef_dic[str(N[i])],outer_loop)
            file_p.close()
    
    for n in xrange(len(N)):
        outerloop_pcg_vec = outerloop_pcg_dic[str(N[n])]
        
        print "N = ",N[n]
        print "len(outerloop_pcg_vec) = ",len(outerloop_pcg_vec)
                  
        if len(outerloop_pcg_vec) == 0:
           break

        outerloop_pcg_mean[n] = np.median(outerloop_pcg_vec[outerloop_pcg_vec<=9998])
        outerloop_stef_vec = outerloop_stef_dic[str(N[n])]

        print "len(outerloop_stef_vec) = ",len(outerloop_stef_vec)
        temp_v = outerloop_stef_vec[outerloop_stef_vec<=9998]
        temp_v = outerloop_stef_vec[temp_v>9000]
        print "len(temp_v) = ",len(temp_v)              
   

        if len(outerloop_stef_vec) == 0:
           break
        
        outerloop_stef_mean[n] = np.median(outerloop_stef_vec[outerloop_stef_vec<=9998])

        print "outerloop_stef_mean = ",outerloop_stef_mean
                
        outerloop_pcg_std[n] = np.median(np.absolute(outerloop_pcg_vec[outerloop_pcg_vec<=9998] - np.median(outerloop_pcg_vec[outerloop_pcg_vec<=9998])))
        outerloop_stef_std[n] = np.median(np.absolute(outerloop_stef_vec[outerloop_stef_vec<=9998] - np.median(outerloop_stef_vec[outerloop_stef_vec<=9998])))
        print "outerloop_stef_std = ",outerloop_stef_std 
    

    #print "kappa_cg_mad = ",kappa_cg_mad
    #print "itr_cg_mad = ",itr_cg_mad

    #print "N = ",N
    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(N,outerloop_pcg_mean,"r",lw=2,label="SPARC")
    ax.fill_between(N,outerloop_pcg_mean-outerloop_pcg_std, outerloop_pcg_mean+outerloop_pcg_std,alpha=0.2, edgecolor='k', facecolor='r')
    ax.plot(N,outerloop_stef_mean,"b",lw=2,label="R-StEFCal")
    ax.fill_between(N, outerloop_stef_mean-outerloop_stef_std, outerloop_stef_mean+outerloop_stef_std,alpha=0.2, edgecolor='k', facecolor='b')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Iterations required by R-StEFCal/SPARC')
    ax.legend(loc=5)
    plt.show()

def plot_err_itr(SNR=1000,num=4,e_upper1=5,e_upper2=5):
    N = np.array([7,19,37,61,91,127,169,217])
    L = np.array([9,30,63,108,165,234,315,408])

    N_num = N[num]
    L_num = L[num]

    stefcal_err_dic = {}
    pcg_err_dic = {}

    for e in xrange(e_upper1):

    	PCG_DIR = "./HEX_PCG_"+str(SNR)+"_False_"+str(e)
    	PCG_FILE = PCG_DIR+"/"+str(num+1)+"_"+str(N_num)+"_"+str(L_num)+"_"+PCG_DIR[2:]+".p"

        file_p = open(PCG_FILE, 'rb')
    	order = pickle.load(file_p)
    	N_v = pickle.load(file_p)
    	L_v = pickle.load(file_p)
    	zeta = pickle.load(file_p)
    	PQ = pickle.load(file_p)
    	z_cal = pickle.load(file_p)
    	c_cal = pickle.load(file_p)
    	time_mat = pickle.load(file_p)
    	outer_loop = pickle.load(file_p)
    	itr_vec = pickle.load(file_p)
    	kappa_vec = pickle.load(file_p)
    	G_cal = pickle.load(file_p)
    	M_cal = pickle.load(file_p)
    	D = pickle.load(file_p)
    	M = pickle.load(file_p)
    	error_pcg = pickle.load(file_p)
        for t in xrange(len(error_pcg)):
            e_pcg = error_pcg[str(t)]
            for i in xrange(len(e_pcg)):
                if str(i) not in pcg_err_dic.keys():
                   pcg_err_dic[str(i)] = np.array([e_pcg[i]])
                else:
		   pcg_err_dic[str(i)] = np.append(pcg_err_dic[str(i)],np.array([e_pcg[i]]))

    	file_p.close()

    for e in xrange(e_upper2):

        STEF_DIR = "./HEX_R_StEFCal_"+str(SNR)+"_"+str(e)
        STEF_FILE = STEF_DIR+"/"+str(num+1)+"_"+str(N_num)+"_"+str(L_num)+"_"+STEF_DIR[2:]+".p"
        
        file_p = open(STEF_FILE, 'rb')
        order = pickle.load(file_p)
        N_v = pickle.load(file_p)
        L_v = pickle.load(file_p)
        zeta = pickle.load(file_p)
        z_cal = pickle.load(file_p)
        c_cal = pickle.load(file_p)
        time_mat = pickle.load(file_p)
        outer_loop = pickle.load(file_p)
        G_cal = pickle.load(file_p)
        M_cal = pickle.load(file_p)
        D = pickle.load(file_p)
        M = pickle.load(file_p)
        error_stef = pickle.load(file_p) 
        
        for t in xrange(len(error_stef)):
            e_stef = error_stef[str(t)]
            for i in xrange(len(e_stef)):
                if str(i) not in stefcal_err_dic.keys():
                   stefcal_err_dic[str(i)] = np.array([e_stef[i]])
                else:
		   stefcal_err_dic[str(i)] = np.append(stefcal_err_dic[str(i)],np.array([e_stef[i]]))
        file_p.close()

    mean_pcg_error = np.zeros((len(pcg_err_dic),))
    std_pcg_error = np.zeros((len(pcg_err_dic),))

    for i in xrange(len(pcg_err_dic)):
        mean_pcg_error[i] = np.median(pcg_err_dic[str(i)])
        std_pcg_error[i] = mad(pcg_err_dic[str(i)])  

    mean_stefcal_error = np.zeros((len(stefcal_err_dic),))
    std_stefcal_error = np.zeros((len(stefcal_err_dic),))

    for i in xrange(len(stefcal_err_dic)):
        mean_stefcal_error[i] = np.median(stefcal_err_dic[str(i)])
        std_stefcal_error[i] = mad(stefcal_err_dic[str(i)])  

    itr_pcg = np.cumsum(np.ones(mean_pcg_error.shape))
    itr_stef = np.cumsum(np.ones(mean_stefcal_error.shape))
 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.semilogy(itr_pcg,mean_pcg_error,'r')
    ax.fill_between(itr_pcg,mean_pcg_error-std_pcg_error, mean_pcg_error + std_pcg_error, alpha=0.2, edgecolor='k', facecolor='r')
    

    ax.semilogy(itr_stef,mean_stefcal_error,'b')
    ax.fill_between(itr_stef,mean_stefcal_error-std_stefcal_error, mean_stefcal_error + std_stefcal_error, alpha=0.2, edgecolor='k', facecolor='b')

    #print "std_stefcal_error = ",std_stefcal_error
    #print "mean_stefcal_error = ",mean_stefcal_error

    #print "mean_stefcal_error+std = ",mean_stefcal_error+std_stefcal_error
    #print "std_stefcal_error-std = ",mean_stefcal_error-std_stefcal_error

    plt.show()

def mad(x):
    return np.median(np.absolute(x - np.median(x)))

def plot_precentage_error(SNR=10,k_upper1=5,k_upper2=5):
    
    N = np.array([7,19,37,61,91,108,169,217])

    precentage_pcg_mean = np.zeros((len(N),),dtype=float)
    precentage_stef_mean = np.zeros((len(N),),dtype=float)

    precentage_pcg_std = np.zeros((len(N),),dtype=float)
    precentage_stef_std = np.zeros((len(N),),dtype=float)
    
    precentage_pcg_dic = {}
    precentage_stef_dic = {}
    
    for j in xrange(len(N)):
        precentage_pcg_dic[str(N[j])]=np.array([],dtype=float)
        precentage_stef_dic[str(N[j])]=np.array([],dtype=float)

    for k in xrange(k_upper1):
        PCG_DIR = "./HEX_PCG_"+str(SNR)+"_False_"+str(k)
        PCG_FILE_LIST = np.array([PCG_DIR+"/1_7_9_"+PCG_DIR[2:]+".p",PCG_DIR+"/2_19_30_"+PCG_DIR[2:]+".p",PCG_DIR+"/3_37_63_"+PCG_DIR[2:]+".p",PCG_DIR+"/4_61_108_"+PCG_DIR[2:]+".p",PCG_DIR+"/5_91_165_"+PCG_DIR[2:]+".p",PCG_DIR+"/6_127_234_"+PCG_DIR[2:]+".p",PCG_DIR+"/7_169_315_"+PCG_DIR[2:]+".p",PCG_DIR+"/8_217_408_"+PCG_DIR[2:]+".p"]) 
        
        for i in xrange(len(PCG_FILE_LIST)):
            #LOAD PCG
            print "PCG_FILE_LIST = ",PCG_FILE_LIST[i]
            if not os.path.isfile(PCG_FILE_LIST[i]):
               break
            file_p = open(PCG_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            PQ = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            itr_vec = pickle.load(file_p)
    	    kappa_vec = pickle.load(file_p)
    	    G_cal = pickle.load(file_p)
    	    M_cal = pickle.load(file_p)
    	    D = pickle.load(file_p)
    	    M = pickle.load(file_p)

            pcg_prec_error_t = np.zeros((M.shape[2],))

            #print "c_cal = ",c_cal
 
            for t in xrange(M.shape[2]):
                mask = np.ones(M[:,:,t].shape,dtype=float)-np.diag(np.ones((M[:,:,t].shape[0],),dtype=float))
                pcg_prec_error_t[t] = np.linalg.norm((M[:,:,t] - G_cal[:,:,t]*M_cal[:,:,t])*mask)**2/np.linalg.norm(M[:,:,t]*mask)**2
                #print "pcg_prec_error_t = ",pcg_prec_error_t[t]
            #from IPython import embed
            #embed() 

            precentage_pcg_dic[str(N[i])] = np.append(precentage_pcg_dic[str(N[i])],pcg_prec_error_t)
            file_p.close()
    
    for k in xrange(k_upper2):
        STEF_DIR = "./HEX_R_StEFCal_"+str(SNR)+"_"+str(k)
        STEF_FILE_LIST = np.array([STEF_DIR+"/1_7_9_"+STEF_DIR[2:]+".p",STEF_DIR+"/2_19_30_"+STEF_DIR[2:]+".p",STEF_DIR+"/3_37_63_"+STEF_DIR[2:]+".p",STEF_DIR+"/4_61_108_"+STEF_DIR[2:]+".p",STEF_DIR+"/5_91_165_"+STEF_DIR[2:]+".p",STEF_DIR+"/6_127_234_"+STEF_DIR[2:]+".p",STEF_DIR+"/7_169_315_"+STEF_DIR[2:]+".p",STEF_DIR+"/8_217_408_"+STEF_DIR[2:]+".p"])
        for i in xrange(len(STEF_FILE_LIST)):
            #LOAD STEF
            print "STEF_FILE_LIST = ",STEF_FILE_LIST[i]
            if not os.path.isfile(STEF_FILE_LIST[i]):
               break
            file_p = open(STEF_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            G_cal = pickle.load(file_p)
            M_cal = pickle.load(file_p)
            D = pickle.load(file_p)
            M = pickle.load(file_p)
            
            stef_prec_error_t = np.zeros((M.shape[2],))

            for t in xrange(M.shape[2]):
                mask = np.ones(M[:,:,t].shape,dtype=float)-np.diag(np.ones((M[:,:,t].shape[0],),dtype=float))
                stef_prec_error_t[t] = np.linalg.norm((M[:,:,t] - G_cal[:,:,t]*M_cal[:,:,t])*mask)**2/np.linalg.norm(M[:,:,t]*mask)**2 

            precentage_stef_dic[str(N[i])] = np.append(precentage_stef_dic[str(N[i])],stef_prec_error_t)

            file_p.close()
    
    for n in xrange(len(N)):
        precentage_pcg_vec = precentage_pcg_dic[str(N[n])]
        
        if len(precentage_pcg_vec) == 0:
           break

        precentage_pcg_mean[n] = np.median(precentage_pcg_vec)
        precentage_pcg_std[n] = mad(precentage_pcg_vec)
        
        precentage_stef_vec = precentage_stef_dic[str(N[n])]

        precentage_stef_mean[n] = np.median(precentage_stef_vec)
        precentage_stef_std[n] = mad(precentage_stef_vec)
                     
   
    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.plot(N,precentage_pcg_mean,"r",lw=2,label="SPARC")
    #ax.fill_between(N,precentage_pcg_mean-precentage_pcg_std, precentage_pcg_mean+precentage_pcg_std,alpha=0.2, edgecolor='k', facecolor='r')
    ax.plot(N,precentage_stef_mean,"bo",lw=2,label="R-StEFCal")
    ax.errorbar(N, precentage_stef_mean,yerr=precentage_stef_std)
    #ax.fill_between(N, precentage_stef_mean-precentage_stef_std, precentage_stef_mean+precentage_stef_std,alpha=0.2, edgecolor='k', facecolor='b')
    #ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Precentage Error')
    ax.legend(loc=5)
    plt.show()


def plot_time(SNR=1000,k_upper1=5,k_upper2=5,k_upper3=5):
    
    N = np.array([7,19,37,61,91,108,169,217])
    L = np.array([9,30,63,108,165,234,315,408])
    
    P = 2*(N+L)
 
     

    time_pcg_mean = np.zeros((len(N),),dtype=float)
    time_stef_mean = np.zeros((len(N),),dtype=float)
    time_svd_mean = np.zeros((len(N),),dtype=float)

    time_pcg_std = np.zeros((len(N),),dtype=float)
    time_stef_std = np.zeros((len(N),),dtype=float)
    time_svd_std = np.zeros((len(N),),dtype=float)    

    time_pcg_dic = {}
    time_stef_dic = {}
    time_svd_dic = {}    

    for j in xrange(len(N)):
        time_pcg_dic[str(N[j])]=np.array([],dtype=float)
        time_stef_dic[str(N[j])]=np.array([],dtype=float)
        time_svd_dic[str(N[j])]=np.array([],dtype=float)

    for k in xrange(k_upper1):
        PCG_DIR = "./HEX_PCG_"+str(SNR)+"_False_"+str(k)
    
        PCG_FILE_LIST = np.array([PCG_DIR+"/1_7_9_"+PCG_DIR[2:]+".p",PCG_DIR+"/2_19_30_"+PCG_DIR[2:]+".p",PCG_DIR+"/3_37_63_"+PCG_DIR[2:]+".p",PCG_DIR+"/4_61_108_"+PCG_DIR[2:]+".p",PCG_DIR+"/5_91_165_"+PCG_DIR[2:]+".p",PCG_DIR+"/6_127_234_"+PCG_DIR[2:]+".p",PCG_DIR+"/7_169_315_"+PCG_DIR[2:]+".p",PCG_DIR+"/8_217_408_"+PCG_DIR[2:]+".p"]) 

        #SVD_FILE_LIST = np.array([SVD_DIR+"/1_7_9_"+SVD_DIR[2:]+".p",SVD_DIR+"/2_19_30_"+SVD_DIR[2:]+".p"])
        for i in xrange(len(PCG_FILE_LIST)):
            #LOAD PCG
            print "PCG_FILE_LIST = ",PCG_FILE_LIST[i]
            if not os.path.isfile(PCG_FILE_LIST[i]):
               break
            
            file_p = open(PCG_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            PQ = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            t = (time_mat[1,:] - time_mat[0,:])/(1.0*outer_loop)
            time_pcg_dic[str(N[i])] = np.append(time_pcg_dic[str(N[i])],t)
            file_p.close()
  
    for k in xrange(k_upper2):

        STEF_DIR = "./HEX_R_StEFCal_"+str(SNR)+"_"+str(k)
        STEF_FILE_LIST = np.array([STEF_DIR+"/1_7_9_"+STEF_DIR[2:]+".p",STEF_DIR+"/2_19_30_"+STEF_DIR[2:]+".p",STEF_DIR+"/3_37_63_"+STEF_DIR[2:]+".p",STEF_DIR+"/4_61_108_"+STEF_DIR[2:]+".p",STEF_DIR+"/5_91_165_"+STEF_DIR[2:]+".p",STEF_DIR+"/6_127_234_"+STEF_DIR[2:]+".p",STEF_DIR+"/7_169_315_"+STEF_DIR[2:]+".p",STEF_DIR+"/8_217_408_"+STEF_DIR[2:]+".p"])
        
        #LOAD STEF
        for i in xrange(len(STEF_FILE_LIST)):
            print "STEF_FILE_LIST = ",STEF_FILE_LIST[i]   
            if not os.path.isfile(STEF_FILE_LIST[i]):
               break
            
            file_p = open(STEF_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            t = (time_mat[1,:] - time_mat[0,:])/(1.0*outer_loop)
            time_stef_dic[str(N[i])] = np.append(time_stef_dic[str(N[i])],t)
            file_p.close()

    for k in xrange(k_upper3):
        SVD_DIR = "./HEX_SVD_"+str(SNR)+"_True_"+str(k)
        SVD_FILE_LIST = np.array([SVD_DIR+"/1_7_9_"+SVD_DIR[2:]+".p",SVD_DIR+"/2_19_30_"+SVD_DIR[2:]+".p",SVD_DIR+"/3_37_63_"+SVD_DIR[2:]+".p",SVD_DIR+"/4_61_108_"+SVD_DIR[2:]+".p",SVD_DIR+"/5_91_165_"+SVD_DIR[2:]+".p",SVD_DIR+"/6_127_234_"+SVD_DIR[2:]+".p",SVD_DIR+"/7_169_315_"+SVD_DIR[2:]+".p",SVD_DIR+"/8_217_408_"+SVD_DIR[2:]+".p"])

        #LOAD SVD
        for i in xrange(len(SVD_FILE_LIST)):    
            print "SVD_FILE_LIST = ",SVD_FILE_LIST[i]
            if not os.path.isfile(SVD_FILE_LIST[i]):
               break
            file_p = open(SVD_FILE_LIST[i], 'rb')
            order = pickle.load(file_p)
            N_v = pickle.load(file_p)
            L = pickle.load(file_p)
            zeta = pickle.load(file_p)
            PQ = pickle.load(file_p)
            z_cal = pickle.load(file_p)
            c_cal = pickle.load(file_p)
            time_mat = pickle.load(file_p)
            outer_loop = pickle.load(file_p)
            t = (time_mat[1,:] - time_mat[0,:])/(1.0*outer_loop)
            time_svd_dic[str(N[i])] = np.append(time_svd_dic[str(N[i])],t)
            file_p.close()
    
    for n in xrange(len(N)):
        time_pcg_mean[n] = np.mean(time_pcg_dic[str(N[n])])
        time_stef_mean[n] = np.mean(time_stef_dic[str(N[n])])
        time_svd_mean[n] = np.mean(time_svd_dic[str(N[n])])

        time_pcg_std[n] = np.std(time_pcg_dic[str(N[n])])
        time_stef_std[n] = np.std(time_stef_dic[str(N[n])])
        time_svd_std[n] = np.std(time_svd_dic[str(N[n])])
        
        
    #print "kappa_cg_mad = ",kappa_cg_mad
    #print "itr_cg_mad = ",itr_cg_mad

    #print "N = ",N
    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(N,time_pcg_mean,"r",lw=2,label="PCG")
    ax.fill_between(N,time_pcg_mean-time_pcg_std, time_pcg_mean+time_pcg_std,alpha=0.2, edgecolor='k', facecolor='r')
    #z = np.polyfit(N, time_pcg_mean, 2)
    #ax.plot(N,z[0]*N**2+z[1]*N+z[0],"r--")
    #print "z = ",z
    ax.plot(N,time_stef_mean,"b",lw=2,label="R-StEFCal")
    ax.fill_between(N, time_stef_mean-time_stef_std, time_stef_mean+time_stef_std,alpha=0.2, edgecolor='k', facecolor='b')
    #z = np.polyfit(N, time_stef_mean, 2)
    #ax.plot(N,z[0]*N**2,"b--")
    #print "z = ",z
    ax.plot(N,time_svd_mean,"g",lw=2,label="SVD")
    ax.fill_between(N, time_svd_mean-time_svd_std, time_svd_mean+time_svd_std,alpha=0.2, edgecolor='k', facecolor='g')
    #z = np.polyfit(N, time_svd_mean, 3)
    #ax.plot(N,z[0]*N**3,"g--")
    #print "z = ",z
    #ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Average Execution Time per outer loop [$s$]')
    ax.legend(loc=2)
    plt.show()

    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(P,time_pcg_mean,"r",lw=2,label="PCG")
    ax.fill_between(P,time_pcg_mean-time_pcg_std, time_pcg_mean+time_pcg_std,alpha=0.2, edgecolor='k', facecolor='r')
    #z = np.polyfit(P, time_pcg_mean, 2)
    #ax.plot(P,z[0]*P**2,"r--")
    #print "z = ",z
    ax.plot(P,time_stef_mean,"b",lw=2,label="R-StEFCal")
    ax.fill_between(P, time_stef_mean-time_stef_std, time_stef_mean+time_stef_std,alpha=0.2, edgecolor='k', facecolor='b')
    #z = np.polyfit(P, time_stef_mean, 2)
    #ax.plot(P,z[0]*P**2,"b--")
    #print "z = ",z
    ax.plot(P,time_svd_mean,"g",lw=2,label="SVD")
    ax.fill_between(P, time_svd_mean-time_svd_std, time_svd_mean+time_svd_std,alpha=0.2, edgecolor='k', facecolor='g')
    #z = np.polyfit(P, time_svd_mean, 3)
    #ax.plot(P,z[0]*P**3,"g--")
    #print "z = ",z
    #ax.set_yscale('log')
    ax.set_xlabel(r'$P$')
    ax.set_ylabel('Average Execution Time per outer loop [$s$]')
    ax.legend(loc=2)
    plt.show()

def plot_sparsity():
    output = open('sparsity.p', 'rb')
    N = pickle.load(output)
    gamma = pickle.load(output) 
    N_REG = pickle.load(output) 
    gamma_REG = pickle.load(output)
    S_SQR = pickle.load(output)
    N_SQR = pickle.load(output)
    gamma_SQR = pickle.load(output)
    R_HEX = pickle.load(output)
    N_HEX = pickle.load(output)
    gamma_HEX = pickle.load(output)
    output.close()   

    plt.plot(N,gamma,'r--',lw=1.5)
    plt.plot(N,5.0/8.0*np.ones(N.shape),'k:',lw=1.5)
    plt.plot(N_REG,gamma_REG,'ro',ms=8.0,mfc="w",mec='r')
    plt.plot(N_SQR,gamma_SQR,'gx',ms=8.0)
    plt.plot(N_HEX,gamma_HEX,'bo',ms=8.0,mfc="w",mec='b')
    plt.show()

    P = 4*N-2
    P_REG = 4*N_REG-2

    P_SQR = 2*(N_SQR + 2*S_SQR**2 - 2*S_SQR)

    P_HEX = 2*(N_HEX+6*R_HEX**2 + 3*R_HEX)
  
    plt.plot(N,np.log(1-gamma)/np.log(P) + 2,'r--',lw=1.5)
    plt.plot(N_REG,np.log(1-gamma_REG)/np.log(P_REG)+2,'ro',ms=8.0)
    plt.plot(N_SQR,np.log(1-gamma_SQR)/np.log(P_SQR)+2,'gx',ms=8.0)
    plt.plot(N_HEX,np.log(1-gamma_HEX)/np.log(P_HEX)+2,'bo',ms=8.0)
    plt.show()

if __name__ == "__main__":
   #plot_kappa_itr(SNR=5)
   #plot_outer_loop(SNR=5)
   #plot_time()
   #plot_sparsity()
   plot_precentage_error(SNR=5,k_upper1=5,k_upper2=5)
   #plot_err_itr(num=4)
