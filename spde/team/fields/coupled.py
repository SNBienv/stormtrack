import numpy as np,sys,time; sys.path.insert(0,'.'); from l96 import *
c=float(sys.argv[1]); J=int(sys.argv[2]); T=float(sys.argv[3]); K=8; F=20.;h=1.;b=10.
dt=0.001*10/max(c,10); every=int(round(0.005/dt))
p=(F,h,b,c,J); rng=np.random.default_rng(1)
B=4  # 4 independent coupled trajectories
X0=rng.normal(5,3,(B,K)); Y0=rng.normal(0,0.1,(B,K*J))
t0=time.time()
Xs,Us,_,_=run(p,T,dt,every,X0,Y0,spin=10.)
np.savez(f'coupled_c{c:g}_J{J}.npz',X=Xs,U=Us,dt=0.005)
print('done',time.time()-t0,Xs.shape)
