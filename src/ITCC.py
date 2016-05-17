
def ITCC(p, k, l, n_iters, convergeThresh, cX, cY):


    import numpy as np
    from util import prob_clust_Indiv,prob_clust,prob_x_given_xhat,prob_y_given_yhat,prob_Y_given_x,prob_X_given_y,calc_q_Indiv,calc_q,prob_Y_given_xhat,prob_X_given_yhat,kl_divergence,next_cx,next_cX,next_cy,next_cY,sorting
     #import calc_q #,kl_divergence,next_cX,next_cY

    m = np.shape(p)[0]
    n = np.shape(p)[1]
    
    converged = False

    kl_curr = 0.0
    kl_prev = 0.0
    

    
    q = calc_q(p, range(0,m), cX, range(0,n), cY)
    kl_curr = kl_divergence(p.ravel(), q.ravel())
    Error=[kl_curr] 

    for i in range(0,n_iters):
        
        kl_prev = kl_curr
    # Update cX, q
        cX = np.matrix (next_cX(p,q, cX, k) )
        q = calc_q(p, range(0,m), cX, range(0,n), cY)

    # Update cY, q
        cY = np.matrix (next_cY(p,q, cY, l) )
        q = calc_q(p, range(0,m), cX, range(0,n), cY)
    
        kl_curr = kl_divergence(p.ravel(), q.ravel())
        Error.append(kl_curr)
    
        
        #print(1)
        if (kl_prev - kl_curr) < convergeThresh:
            converged = True
            break
            
    M1=sorting(p,k,l,cX,cY)      
    clustered=prob_clust(M1, range(0,k),cX, range(0,l),cY)     
    return(M1,q,cX,cY,clustered,Error)