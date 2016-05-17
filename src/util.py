import numpy as np


def prob_clust_Indiv(p, xhat, cX, yhat, cY):
    return np.sum( p[np.array(cX==xhat).ravel(),:][:,np.array(cY==yhat).ravel()] )


def prob_clust(p, Xhat, cX, Yhat, cY):
    output = np.empty([len(Xhat), len(Yhat)],dtype=float)  
    for xhat in Xhat:
        for yhat in Yhat:
            output[xhat, yhat] = prob_clust_Indiv(p, xhat,cX, yhat,cY)
    return output


def prob_x_given_xhat(p, x, xhat, cX):
    return  np.sum(p[x,:]) / np.sum(p[np.array(cX==xhat).ravel(),:])   #

def prob_y_given_yhat(p ,y, yhat, cY):
    return  np.sum(p[:,y]) / np.sum(p[:,np.array(cY==yhat).ravel()])   #  

def prob_Y_given_x(p, x):
    return p[x,:] / np.sum(p[x,:])

def prob_X_given_y(p, y):
    return p[:,y] / np.sum(p[:,y])



def calc_q_Indiv(p, x,cX, y,cY):
    return prob_clust_Indiv(p, cX[0,x],cX, cY[0,y], cY) * prob_x_given_xhat(p, x, cX[0,x],cX) * prob_y_given_yhat(p, y, cY[0,y],cY)


def calc_q(p, X, cX, Y, cY):
    output = np.empty([len(X), len(Y)],dtype=float)
    for x in X:
        for y in Y:
            output[x,y] = calc_q_Indiv(p, x,cX, y,cY)
    return output



def prob_Y_given_xhat(p, xhat,cX):
    return  np.sum(  p[np.array(cX==xhat).ravel(),:]  /  np.sum(p[np.array(cX==xhat).ravel(),:])  , axis=0 ) 
         
def prob_X_given_yhat(p, yhat,cY):
    return   np.sum (p[:,np.array(cY==yhat).ravel()]  /  np.sum(p[:,np.array(cY==yhat).ravel()])  ,axis=1 ) 


                  
#def kl_divergence(p,q):
    #p = np.asmatrix(p, dtype=np.float)
    #q = np.asmatrix(q, dtype=np.float)
    #S=0
    #m = np.shape(p)[0]
    #n = np.shape(p)[1]
    #for i in range(0,m):
         #for j in range(0,n):
             #kl=p[i,j]*np.log2(p[i,j]) / (q[i,j])
             #S=S+ kl       
    #return S
    
    #import scipy.stats.distributions
    #sum(scipy.stats.entropy(p,q))
    #return np.sum(np.where(p != 0, p * np.log2(p / q), 0))



def kl_divergence(p,q):
    TOLERANCE = 0.00000000000000000001
    #Big=1000000
    p = np.asmatrix(p, dtype=np.float)
    q = np.asmatrix(q, dtype=np.float)
    S=0
    m = np.shape(p)[0]
    n = np.shape(p)[1]
    for i in range(0,m):
         for j in range(0,n):
             kl=(p[i,j]+TOLERANCE) / (q[i,j]+TOLERANCE)  #+TOLERANCE
             #if (kl)>0 :  #*q[i,j]
             S=S+ (p[i,j]*np.log2(kl))
             #if q[i,j]<TOLERANCE:
                #print(p[i,j])
                #print(q[i,j])
                #print("salam")          
    return S

    
    
    
    
    


def next_cx(p,q, x, cX, k):
    q_dist_xhat = np.empty(k)
    p_dist_x = prob_Y_given_x(p,x)

    for xhat in range(0,k):
        q_dist_xhat[xhat] = kl_divergence( p_dist_x.ravel(), prob_Y_given_xhat(q, xhat,cX).ravel()  )
        #print(q_dist_xhat)
    return np.argmin(q_dist_xhat)   #

def next_cX(p,q, cX, k):
    output = np.empty(np.shape(cX)[1])
    for x in range(0,np.shape(cX)[1]):
        #print(x)
        output[x] = next_cx(p,q, x,cX, k)
    return output

def next_cy(p,q, y,cY, l):
    q_dist_yhat = np.empty(l)
    p_dist_y = prob_X_given_y(p,y)
    for yhat in range(0,l):
        q_dist_yhat[yhat] = kl_divergence( p_dist_y.ravel(), prob_X_given_yhat(q, yhat, cY).ravel()  )  
    return np.argmin(q_dist_yhat)


def next_cY(p,q, cY, l):
    output = np.empty(np.shape(cY)[1])
    for y in range(0,np.shape(cY)[1]):
        output[y] = next_cy(p,q, y, cY, l)
    return output
    
    
    
def sorting(p,k,l,cX,cY):
     m = np.shape(p)[0]
     n = np.shape(p)[1]
     M=np.empty((1,n))
     M=np.delete(M, (0), axis=0)

     for i in range(0,k):
    #ind=cX[np.array(cX==i)]
        indexes=[]
        indexes =np.where(cX==i)
        a=p[indexes[1],:]
        M=np.vstack([M,a])

     M1=np.empty((m,1))
     M1=np.delete(M1, (0), axis=1) 
        
     for j in range(0,l):
    #ind=cX[np.array(cX==i)]
        indexes=[]
        indexes =np.where(cY==j)
        a=M[:,indexes[1]]
        M1=np.hstack([M1,a]) 
        
     return M1 