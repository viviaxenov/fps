import numpy as np
import emcee

H5_FILE = "./sample_restart_1769524224.h5"

discard_frac = 0.9
thin = 1

r = emcee.backends.HDFBackend(H5_FILE, read_only=True)

T = r.iteration
chain = r.get_chain(discard=int(discard_frac * T), thin=thin, flat=False) 

L, J, D = chain.shape

xbar_j = chain.mean(axis=0)                
xbar_star = xbar_j.mean(axis=0)            

B = (L / (J - 1.0)) * ((xbar_j - xbar_star) ** 2).sum(axis=0)  
W = chain.var(axis=0, ddof=1).mean(axis=0)                     

R = (((L - 1.0) / L) * W + (1.0 / L) * B) / W

print("T =", T, "J =", J, "L =", L, "D =", D)
print("R per dim:", R)
print("R max:", float(np.max(R)))
