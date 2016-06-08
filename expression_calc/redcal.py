import numpy as np
import pylab as plt

import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering

def testH(g,y):

    gc = g.conjugate() 
    J_1 = np.array([[y[0]*gc[1],0,0,0,0,g[0]*gc[1],0,0,0],[0,y[0]*gc[2],0,0,0,g[1]*gc[2],0,0,0],[0,0,y[0]*gc[3],0,0,g[2]*gc[3],0,0,0],[0,0,0,y[0]*gc[4],0,g[3]*gc[4],0,0,0],[y[1]*gc[2],0,0,0,0,0,g[0]*gc[2],0,0],[0,y[1]*gc[3],0,0,0,0,g[1]*gc[3],0,0],[0,0,y[1]*gc[4],0,0,0,g[2]*gc[4],0,0],[y[2]*gc[3],0,0,0,0,0,0,g[0]*gc[3],0],[0,y[2]*gc[4],0,0,0,0,0,g[1]*gc[4],0],[y[3]*gc[4],0,0,0,0,0,0,0,g[0]*gc[4]]])
    print "J_1 = ",J_1

    J_2 = np.array([[0,g[0]*y[0],0,0,0,0,0,0,0],[0,0,g[1]*y[0],0,0,0,0,0,0],[0,0,0,g[2]*y[0],0,0,0,0,0],[0,0,0,0,g[3]*y[0],0,0,0,0],[0,0,g[0]*y[1],0,0,0,0,0,0],[0,0,0,g[1]*y[1],0,0,0,0,0],[0,0,0,0,g[2]*y[1],0,0,0,0],[0,0,0,g[0]*y[2],0,0,0,0,0],[0,0,0,0,g[1]*y[2],0,0,0,0],[0,0,0,0,g[0]*y[3],0,0,0,0]])
    print "J_2 = ",J_2

    J = np.zeros((20,18),dtype=complex)
  
    J[0:10,0:9] = J_1
    J[0:10:,9:] = J_2
    J[10:,0:9] = J_2.conjugate()
    J[10:,9:] = J_1.conjugate()

    print "J = ",J
    plt.imshow(np.absolute(J),interpolation="nearest")
    plt.colorbar()
    plt.show()
    J_H = J.transpose()
    J_H = J_H.conjugate()

    H = np.dot(J_H,J)

    plt.imshow(np.absolute(H),interpolation="nearest") 
    plt.colorbar()
    plt.show()

    H_mask = np.copy(H)

    H_mask[np.absolute(H) > 6] = 1
    H_mask[np.absolute(H) <= 6] = 0

    plt.imshow(np.absolute(H)*np.absolute(H_mask),interpolation="nearest") 
    plt.colorbar()
    plt.show()
    return np.real(H_mask),H   

if  __name__=="__main__":
    A,H = testH(np.array([1+2j,0.2-0.7j,1-0.9j,1+1.3j,1-0.9j]),np.array([1+1.1j,0.2-0.3j,2-0.9j,2+1.3j]))

    G = nx.Graph(A)
    #rcm = list(reverse_cuthill_mckee_ordering(G))
    rcm = list(reverse_cuthill_mckee_ordering(G))
    print "rcm = ",rcm
    A1 = A[rcm, :][:, rcm]
    H = H[rcm,:][:,rcm]
    plt.imshow(A1,interpolation='nearest')
    plt.show()
    plt.imshow(np.absolute(H),interpolation='nearest')
    plt.colorbar()
    plt.show()
     
    

