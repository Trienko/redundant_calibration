import numpy as np
import pylab as plt
import scipy as sp

import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering

ITR = 0
compute_error = False
error_list = []
x_c = 0
M = np.array([])

def report(xk):
    #print "IN"
    global ITR
    global compute_error
    global error_list
    global M

    ITR = ITR + 1
    if compute_error:
       #print "len xk = ",len(xk)
       e = x_c - xk
       e_temp = np.dot(M,e)
       #print "e_temp = ",len(e_temp)
       e_v = np.sqrt(np.sum(e_temp*e))
       #print "e_v = ",len(e_v)
       e2 = x_c - np.ones(x_c.shape)
       e_temp2 = np.dot(M,e2)   
       e_v2 = np.sqrt(np.sum(e_temp2*e2))       

       error_list.append(e_v.real/e_v2.real) 
       
    

def create_matrix():
    A = np.array([[3,2],[2,6]])
    b = np.array([2,-8])
    return A,b

def tridiag(a, b, c, N, k1=-1, k2=0, k3=1):
    t_1 = np.ones((N,))
    return np.diag(a*t_1[1:], k1) + np.diag(b*t_1, k2) + np.diag(c*t_1[1:], k3)

def quaddiag_p1(a,b,c,d,e,N,k1=-2,k2=-1,k3=0,k4=1,k5=2):
    t_1 = np.ones((N,))
    D = np.diag(a*t_1[2:], k1) + np.diag(b*t_1[1:], k2) + np.diag(c*t_1, k3) + np.diag(d*t_1[1:], k4) + np.diag(e*t_1[2:], k5)
    D[0,0] = D[0,0]-1
    D[N-1,N-1] = D[N-1,N-1]-1
    return D
    
if  __name__=="__main__":
    #global ITR
    #global compute_error
    #global error_list
    #global M
    '''
    #a = 1./9; b = 5./18; c = 1./9; m = 50; N = m**2
    a = -1; b = 2; c = -1; m = 50; N = m**2
    T1 = tridiag(a, b, c, m)

    a = 0; b = 1; c = 0
    I = tridiag(a, b, c, m)

    A = np.kron(T1,I) + np.kron(I,T1)

    A_s = sp.sparse.bsr_matrix(A)

    h = 1./(m+1)

    f = (h**2)*np.ones((N,))

    x,info = sp.sparse.linalg.cg(A=A_s,b=f,x0=np.ones((N,)),tol=1e-12,callback=report)
    
    print "N = ",N
    print "x = ",x
    print "info = ",info
    print "ITR = ",ITR

    compute_error = True
    M = A
    x_c = np.copy(x)
    ITR = 0

    lam = np.linalg.eigvals(A)

    print "lam = ",lam
    
    lam_max = np.amax(lam)
    lam_min = np.amin(lam)
    kappa = lam_max.real/lam_min.real

    print "kappa = ",kappa
     

    x,info = sp.sparse.linalg.cg(A=A_s,b=f,x0=np.ones((N,)),tol=1e-5,callback=report)
    print "error_list = ",error_list
    error_list_numpy = np.array(error_list)
    K = np.cumsum(np.ones((ITR,)))
    plt.semilogy(K,error_list_numpy)
    plt.semilogy(K,2*np.exp(-2./np.sqrt(kappa)*K))
    plt.show()
  
    '''
    #A,b = create_matrix()
    #sp.sparse.linalg.cg(A=A,b=JHr,M=np.linalg.pinv(H*I),callback=report)
    #x,info = sp.sparse.linalg.cg(A=A,b=b,callback=report) 
    #w = np.linalg.eigvals(A)
    #print "max_w = ",np.amax(w)
    #print "min_w = ",np.amin(w)


    #a = 1./9; b = 5./18; c = 1./9; m = 50; N = m**2
    a = 1; b = -4; c = 6; d = -4; e = 1; m = 40; N = m**2
    A = quaddiag_p1(a, b, c, d, e, m)

    print "A = ",A.shape
    print "A = ",A

    A_s = sp.sparse.bsr_matrix(A)

    f = np.zeros((m,))

    f[0] = 1

    print "f = ",f.shape

    x,info = sp.sparse.linalg.cg(A=A_s,b=f,x0=np.ones((m,)),tol=1e-12,callback=report)
    
    print "N = ",N
    print "x = ",x
    print "info = ",info
    print "ITR = ",ITR

    compute_error = True
    M = A
    x_c = np.copy(x)
    ITR = 0

    lam = np.linalg.eigvals(A)

    print "lam = ",lam
    
    lam_max = np.amax(lam)
    lam_min = np.amin(lam)
    kappa = lam_max.real/lam_min.real

    print "kappa = ",kappa
     

    x,info = sp.sparse.linalg.cg(A=A_s,b=f,x0=np.ones((m,)),tol=1e-5,callback=report)
    print "x = ",x
    print "info =",info
    print "error_list = ",error_list
    error_list_numpy = np.array(error_list)
    K = np.cumsum(np.ones((ITR,)))
    plt.semilogy(K,error_list_numpy)
    plt.semilogy(K,2*np.exp(-2./np.sqrt(kappa)*K))
    plt.show()
  
    
    #A,b = create_matrix()
    #sp.sparse.linalg.cg(A=A,b=JHr,M=np.linalg.pinv(H*I),callback=report)
    #x,info = sp.sparse.linalg.cg(A=A,b=b,callback=report) 
    #w = np.linalg.eigvals(A)
    #print "max_w = ",np.amax(w)
    #print "min_w = ",np.amin(w)
    



