import numpy as np
import scipy as sp
import pylab as plt
import simulator
import analytic




def LINCAL_J(z,N,L,phi):
    g = z[:N]
    y = z[N:]

    rows = (N**2 - N)/2
    columns = 2*N + 2*L

    J = np.zeros((rows,columns),dtype=complex)

    p_v = np.zeros((rows,),dtype=int)
    q_v = np.zeros((rows,),dtype=int)
          
    counter = 0
    for k in xrange(N):
        for j in xrange(k+1,N):
              p_v[counter] = k
              q_v[counter] = j
              counter = counter + 1

    for r in xrange(rows):
          p = p_v[r]
          q = q_v[r]
          phi_v = phi[p,q]
          for c in xrange(columns):
              if c == p:
                 J[r,c] = g[p]*np.conj(g[q])*y[phi_v-1]  
              if c == q:
                 J[r,c] = g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (N + p)):
                 J[r,c] = 1j*g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (N + q)):
                 J[r,c] = -1j*g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (2*(N) - 1 + phi_v)):
                 J[r,c] = g[p]*np.conj(g[q])*y[phi_v-1]
              elif (c == (2*(N) + L - 1 + phi_v)):
                 J[r,c] = 1j*g[p]*np.conj(g[q])*y[phi_v-1]  

    J = np.vstack([J,np.conj(J)])     

    return J

def compute_v(z,N,phi):
    g = z[:N]
    y = z[N:]

    rows = (N**2 - N)
              
    v = np.zeros((rows,),dtype=complex)

    counter = 0
    for k in xrange(N):
        for j in xrange(k+1,N):
              phi_v = phi[k,j]
              v[counter] = g[k]*np.conjugate(g[j])*y[phi_v-1]
              counter = counter + 1
    v[counter:] = np.conjugate(v[:counter]) 
    return v

def LINCAL_H_analytic(z,N,L,zeta):
    g = z[:N]
    y = z[N:] 

    
    #COMPUTING MATRIX A
    ####################
 
    A = np.zeros((2*N,2*N),dtype=float)

    #COMPUTING MATRIX D
    #*******************     
    D = np.zeros((N,N),dtype=float)

    T = 2*np.diag(np.ones((N,),dtype=float)) - np.ones((N,N),dtype=float)

    for i in xrange(D.shape[0]):
        for j in xrange(D.shape[1]):
            #print "i = ",i
            #print "j = ",j
            if i <> j:
               D[i,j] = 2*np.absolute(g[i])**2*np.absolute(g[j])**2*np.absolute(y[zeta[i,j]-1])**2   
            else:
               for k in xrange(D.shape[0]):
                   print "k = ",k
                   if k <> i:
                      print "zeta[i,k] = ",zeta[i,k]
                      D[i,i] = D[i,i] + 2*np.absolute(g[i])**2*np.absolute(g[k])**2*np.absolute(y[zeta[i,k]-1])**2  
                   
                       
    A[:N,:N] = D
    A[N:,N:] = T*D

    #print "A = ",A
    #plt.imshow(np.absolute(A),interpolation="nearest")
    #plt.show()

    #COMPUTING MATRIX B
    ###################

    B = np.zeros((2*N,2*L),dtype=float) 

    #COMPUTING MATRIX E,F
    #********************

    E = np.zeros((N,L),dtype=float)
    F = np.zeros((N,L),dtype=float)

    for i in xrange(E.shape[0]):
        for j in xrange(E.shape[1]):
            p,q = create_index_ij(zeta,N,i,j)
            #print "i = ",i
            #print "j = ",j
            #print "p = ",p
            #print "q = ",q
            for k in xrange(len(p)):
                temp = 2*np.absolute(g[p[k]])**2*np.absolute(g[q[k]])**2*np.absolute(y[zeta[p[k],q[k]]-1])**2
                E[i,j] = E[i,j] + temp 
                F[i,j] = F[i,j] + delta_i_pq(i,p[k],q[k])*temp

    B[:N,:L] = E
    B[N:,L:] = F 

    #print "B = ",B
    #plt.imshow(np.absolute(B),interpolation="nearest")
    #plt.show() 

    #COMPUTING MATRIX C
    ###################

    C = np.zeros(((2*L,2*L)),dtype=float)
    
    #COMPUTING MATRIX G
    #******************

    G = np.zeros((L,L),dtype=float)

    for i in xrange(G.shape[0]):
        p,q = create_index_i(zeta,N,i)
        for k in xrange(len(p)):
            G[i,i] = G[i,i]+2*np.absolute(g[p[k]])**2*np.absolute(g[q[k]])**2*np.absolute(y[zeta[p[k],q[k]]-1])**2

    C[:L,:L] = G
    C[L:,L:] = G
    
    #print "C = ",C
    #plt.imshow(np.absolute(C),interpolation="nearest")
    #plt.show() 

    #COMPUTING H
    ############

    H = np.zeros((2*N+2*L,2*N+2*L),dtype=float)
    H[:2*N,:2*N] = A
    H[:2*N,2*N:] = B
    H[2*N:,:2*N] = B.transpose()
    H[2*N:,2*N:] = C
    
    #print "H = ",H
    #plt.imshow(np.absolute(H),interpolation="nearest")
    #plt.show()

    return H 
      

def delta_i_pq(i,p,q):
    if i == p:
       return 1
    elif i ==q:
       return -1
    else:
       return 0

def create_index_i(zeta,N,i):

    p = np.array([])
    q = np.array([])

    for l in xrange(N):
        for m in xrange(l+1,N):
            if (zeta[l,m] == i+1):
               p = np.append(p,np.array([l]))
               q = np.append(q,np.array([m])) 
    return p,q

def create_index_ij(zeta,N,i,j):

    p = np.array([])
    q = np.array([])

    for l in xrange(N):
        if i<l:
           if (zeta[i,l] == j+1):
              p = np.append(p,np.array([i]))
              q = np.append(q,np.array([l]))

    for l in xrange(N):
        if l<i:
           if (zeta[l,i] == j+1):
              p = np.append(p,np.array([l]))
              q = np.append(q,np.array([i]))

    return p,q 
  
'''
Creates the redundant complex Jacobian analytically. Returns the result
      
RETURNS:
J - analytic Jacobian 
 
INPUTS:
dim - dimesnion of array
layout - which layout ot choose     
'''
def construct_sym_J_layout(dim=5,layout="REG"):
    r = analytic.redundant()
    r.create_redundant(layout=layout,order=dim,print_pq=False)
    r.create_J1()
    r.create_J2()
    r.conjugate_J1_J2()
    r.create_J()
    #print "self.J_sym.size = ",r.J.shape
    return r.J,r.N,r.L
          
'''
Substitutes the vector z into the symbolic redundant Jacobian. Stores the result in self.J
      
RETURNS:
J - the numerical Jacobian
 
INPUTS:
z - Input vector.
N - Number of elements in array.
plot - Plot the resultant matrices if True.     
'''
def substitute_sym_J(z,J_sym,dim=5,layout="REG",plot=True):
    r = analytic.redundant()
    r.create_redundant(layout=layout,order=dim,print_pq=False)
    g = z[:r.N]
    y = z[r.N:]
    r.J = J_sym
    J_num = r.substitute_J(g,y)
    if plot:
       plt.imshow(np.absolute(J_num),interpolation="nearest")
       plt.show()
    return J_num


   
if __name__ == "__main__":
   s = simulator.sim(layout="REG",order=5)
   s.generate_antenna_layout()
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
 
   J_sym,N,L = construct_sym_J_layout()
   g = np.random.randn(N)+1j*np.random.randn(N)
   y = np.random.randn(L)+1j*np.random.randn(L)
   #g = np.ones((N,))
   #y = np.ones((L,))
   z = np.hstack([g,y])
   
   J_num = substitute_sym_J(z,J_sym)

   JH_num = J_num.transpose().conjugate()
   v_num = compute_v(z,N,phi)

   H_num = np.dot(JH_num,J_num)*np.eye(JH_num.shape[0])
   
   z2 = np.hstack([z,np.conjugate(z)])

   Hz_num = np.dot(H_num,z2)

   JHv_num = np.dot(JH_num,v_num)

   print "v_num = ",v_num 
   print "JHv_num = ",JHv_num
   print "Hz_num = ",Hz_num
   print "d = ",JHv_num - Hz_num
   

   
   '''
   s = simulator.sim(layout="SQR",order=5)
   s.generate_antenna_layout()
   phi,zeta = s.calculate_phi(s.ant[:,0],s.ant[:,1])
   s.plot_ant(title="SQR")
   s.plot_zeta(data=zeta,step1=1,step2=1,label_size=14,cs="cubehelix")
   
   #g = np.random.randn(s.N)+1j*np.random.randn(s.N)
   #y = np.random.randn(s.L)+1j*np.random.randn(s.L)
   g = np.ones((s.N,))
   y = np.ones((s.L,))
   z = np.hstack([g,y])

   J = LINCAL_J(z,s.N,s.L,phi)

   H = np.dot(J.conj().transpose(),J)

   #plt.imshow(np.absolute(J),interpolation='nearest')

   plt.imshow(H.real,interpolation='nearest')
   plt.colorbar()
   plt.show()

   H2 = LINCAL_H_analytic(z,s.N,s.L,zeta)

   plt.imshow(H2,interpolation='nearest')
   plt.colorbar()
   plt.show()

   plt.imshow(H2-H.real,interpolation='nearest')
   plt.colorbar()
   plt.show() 

   print "H2 - H.real = ",H2-H.real
   '''
