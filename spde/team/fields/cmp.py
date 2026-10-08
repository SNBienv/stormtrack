import numpy as np
from analyse import winvar, acf_time
fz=np.load('frozenfine_c10_J32_b10.npz'); Uf=fz['U']; xs=fz['xs']; dtf=float(fz['dt'])
Ubar_f=Uf.mean((0,2))
s2f={D:np.array([winvar(Uf[:,i]-Uf[:,i].mean(),dtf,int(round(D/dtf))).mean() for i in range(len(xs))]) for D in (0.05,0.2,0.5)}
d=np.load('coupled_c10_J32.npz'); X=d['X'];U=d['U'];dt=0.005
print("coupled X: mean %.2f sd %.2f, 1/5/50/95/99 pct"%(X.mean(),X.std()),np.percentile(X,[1,5,50,95,99]).round(2))
# conditional mean
bins=np.arange(-9,20.5,1.0)-0.5; idx=np.digitize(X,bins)-1
print("\n x   frac   E[U|X]coupled  Ubar_frozen   Var(U|X)c  VarU_frozen  s2_frozen(D=.2)  s2_coupled(D=.2)")
r=U-np.interp(X,xs,Ubar_f)
m=int(round(0.2/dt)); n=(r.shape[0]//m)*m
I=r[:n].reshape(n//m,m,4,8).sum(1)*dt; X0=X[:n:m]; id0=np.digitize(X0,bins)-1
for i,x in enumerate(xs):
    sel=idx==i
    if sel.mean()<0.003: continue
    s0=id0==i
    print("%5.1f %.3f  %8.3f  %8.3f   %7.3f  %7.3f   %7.3f   %7.3f"%(x,sel.mean(),U[sel].mean(),Ubar_f[i],U[sel].var(),Uf[:,i].var(),s2f[0.2][i],(I[s0]**2).mean()/0.2))
# pooled
print("\npooled window variance / Delta of residual r=U-Ubar_frozen(X):")
for D in (0.005,0.02,0.05,0.1,0.2,0.5,1.0,2.0):
    v=winvar(r,dt,int(round(D/dt))).mean()
    # frozen prediction averaged over coupled X climatology
    pred=np.interp(X,xs,s2f[0.2] if D>=0.2 else s2f[0.05]).mean()
    print("  D=%5.3f  coupled %.3f   frozen-GK prediction (clim-avg) %.3f"%(D,v,pred))
rc=r.reshape(r.shape[0],-1)
C=acf_time(rc,dt,int(3/dt))
print("\ncoupled residual ACF at lags 0.01,0.05,0.1,0.5,1,2:",[round(C[int(l/dt)],3) for l in (0.01,0.05,0.1,0.5,1,2)])
# cubic fit residual
co=np.polyfit(X.ravel(),U.ravel(),3); rr=U-np.polyval(co,X)
print("cubic fit coeffs",co.round(4),"resid var",rr.var().round(3), " r var",r.var().round(3), "mean r",r.mean().round(3))
C2=acf_time(rr.reshape(rr.shape[0],-1),dt,int(3/dt))
print("cubic-resid ACF at lags 0.005,0.01,0.05,0.1,0.5,1:",[round(C2[max(1,int(round(l/dt)))],3) for l in (0.005,0.01,0.05,0.1,0.5,1)])
