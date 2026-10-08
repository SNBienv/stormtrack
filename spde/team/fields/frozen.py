import numpy as np,sys,time; sys.path.insert(0,'.'); from l96 import *
c=float(sys.argv[1]); J=int(sys.argv[2]); b=float(sys.argv[3]) if len(sys.argv)>3 else 10.
T=float(sys.argv[4]) if len(sys.argv)>4 else 40.
K=8; F=20.;h=1.
xs=np.arange(-6,16.1,2.0); B=len(xs)
dt=0.001*10/max(c,10); every=max(1,int(round(0.002/dt)))
p=(F,h,b,c,J); rng=np.random.default_rng(2)
X0=np.repeat(xs[:,None],K,1); Y0=rng.normal(0,0.1,(B,K*J))
t0=time.time()
_,Us,_,_=run(p,T,dt,every,X0,Y0,freezeX=True,spin=3.)
np.savez(f'frozen_c{c:g}_J{J}_b{b:g}.npz',U=Us,xs=xs,dt=every*dt)
print('done',time.time()-t0)
