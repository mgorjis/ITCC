# ITCC
Information theoretic co-clustering
based on julia implementation of https://github.com/slongwell/ITCC


##Input  
`itcc(p, k, l, n_iters, convergeThresh, cX, cY)` 

`p`: A joint probability matrix  
`k`: Number of row-clusters  
`l`: Number of column-clusters  
`n_iters`: Maximum number of iterations  
`convergeThresh`: Threshold at which algroithm has is said to have converged, i.e. KL-D between p and q has not decreased significantly between iterations  
`cX`: Initial row-cluster assignments (optional; default to random assignments)  
`cY`: Initial column-cluster assignments (optional; default to random assignments)  

##Output  
An instance of the `ITCC_Result` type, which has the following attributes:  
'M1' : sorted p
`cX`: Final row-cluster assignments  
`cY`: Final column-cluster assignmnents  
`q`: Final q  
`clsutered`: View of p given final cX and cY  
Error: error of clustering through iterations