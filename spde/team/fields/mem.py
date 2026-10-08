import numpy as np
fz=np.load('frozenfine_c10_J32_b10.npz'); xs=fz['xs']; Ubar=fz['U'].mean((0,2))
d=np.load('coupled_c10_J32.npz'); X=d['X'];U=d['U'];dt=0.005
r=U-np.interp(X,xs,Ubar)
for lag in (0,0.02,0.05,0.1,0.15,0.2):
    L=int(round(lag/dt)); 
    Ul=np.interp(X[:X.shape[0]-L],xs,Ubar); rl=U[L:]-Ul
    print("lag %.2f: Var(U - Ubar(X(t-lag))) = %.3f"%(lag,rl.var()))
Xd=(X[2:]-X[:-2])/(2*dt); dU=np.gradient(Ubar,xs); g=np.interp(X[1:-1],xs,dU)*Xd
a=np.sum(r[1:-1]*g)/np.sum(g*g); print("regress r on Ubar'(X)*Xdot: coef (=-lag) %.4f, R2 %.3f"%(a,1-((r[1:-1]-a*g)**2).mean()/r[1:-1].var()))
