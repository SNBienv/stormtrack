import numpy as np, os
def winvar(r,dt,m):
    n=(r.shape[0]//m)*m; I=r[:n].reshape(n//m,m,*r.shape[1:]).sum(1)*dt
    return I.var(0)/(m*dt)
fz=np.load('frozenfine_c10_J32_b10.npz'); Uf=fz['U']; xs=fz['xs']; dtf=float(fz['dt'])
Ubar=Uf.mean((0,2))
gk10=np.array([winvar(Uf[:,i]-Uf[:,i].mean(),dtf,int(round(0.5/dtf))).mean() for i in range(len(xs))])
Ds=(0.01,0.05,0.2,0.5,1.0,2.0)
print("c   | E[(E[U|X]-Ubar_fr)^2]  | Var(U)  | window var/Delta for Delta="+",".join(map(str,Ds))+" | fit s_inf | GK pred (10/c)*<s_GK,10>")
for c in (5,10,20,40):
    f='coupled_c10_J32.npz' if c==10 else f'coupledM_c{c}_J32.npz'
    if not os.path.exists(f): continue
    d=np.load(f); X=d['X'];U=d['U'];dt=0.005
    r=U-np.interp(X,xs,Ubar)
    bins=np.arange(-9,20.5,1.0)-0.5; idx=np.digitize(X,bins)-1
    cm=np.array([U[idx==i].mean() if (idx==i).sum()>200 else np.nan for i in range(len(xs))])
    fr=np.array([(idx==i).mean() for i in range(len(xs))])
    bias=np.nansum(fr*(cm-Ubar)**2)
    v=[winvar(r,dt,int(round(D/dt))).mean() for D in Ds]
    A=np.vstack([np.ones(3),1/np.array(Ds[3:])]).T; s,a=np.linalg.lstsq(A,np.array(v[3:]),rcond=None)[0]
    pred=(10/c)*np.interp(X,xs,gk10).mean()
    print("%3d | %.3f | %.3f | "%(c,bias,r.var())+" ".join("%.4f"%x for x in v)+" | %.4f | %.4f  ratio %.2f  Xmean %.2f Xsd %.2f"%(s,pred,s/pred,X.mean(),X.std()))
