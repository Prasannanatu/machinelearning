import math
from tkinter import PhotoImage
import numpy as np



def compute_Phi(x,p):
    
    # Phi = np.empty(p)
    # for i in range(p): 
    #     Phi[i] = np.power(x,i)
    #     print(Phi)
    # Phi = [[1,1,1]]
    # for i in range(1,p):
    #     M1= np.power(x,p)

    #     Phi = np.concatenate((Phi, M1),axis = 1)
    

    # return Phi
    Phi =np.power(x,0)
    for i in range(1,p):
        M1= np.power(x,i)
        # Phi = np.append(Phi,np.power(x,i), axis = 0)
        Phi = np.concatenate((Phi,np.power(x,i)),axis = 1)
        print("New value of phi with i:", i, Phi)
    Phi = np.mat(Phi)
    Phi = Phi.astype('int64')
    # Phi = np.power(x,0)
    # for i in range(1,p):
    #     Phi = np.concatenate((Phi,np.power(x,i)),axis=1)
        
      
        
        
        
    Phi = np.transpose(Phi)   
    print(Phi)
    return Phi



x = np.mat('1.;2.;3')
print(compute_Phi(x,4))





yhat = np.dot(X,w)
        dL_dw = compute_dL_dw(Y,yhat,X)
        w = update_w(w,dL_dw,alpha)