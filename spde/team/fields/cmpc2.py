import numpy as np
def winI(r,dt,m):
    n=(r.shape[0]//m)*m; return r[:n].reshape(n//m,m,*r.shape[1:]).sum(1)*dt
fz=np.load('frozenfine_c10_J32_b10.npz'); Uf=fz['U']; xs=fz['xs']; dtf=float(fz['dt'])
Ubar=Uf.mean((0,2))
gk10=np.array([(winI(Uf[:,i]-Uf[:,i].mean(),dtf,int(round(0.5/dtf)))**2).mean()/0.5 for i in range(len(xs))])
dt=0.005; bins=np.arange(-9,20.5,1.0)-0.5
groups={'X<0':lambda x:x<0,'0<=X<5':lambda x:(x>=0)&(x<5),'X>=6':lambda x:x>=6}
print("c-scaled diffusion c*s2 (window D) of r=U-E_coupled[U|X]; frozen GK: 10*s2_GK,10 (avg over same X)")
for c in (5,10,20,40):
    f='coupled_c10_J32.npz' if c==10 else f'coupledM_c{c}_J32.npz'
    d=np.load(f); X=d['X'];U=d['U']
    idx=np.digitize(X,bins)-1
    cm=np.array([U[idx==i].mean() if (idx==i).sum()>50 else np.nan for i in range(len(xs))])
    ok=~np.isnan(cm); r=U-np.interp(X,xs[ok],cm[ok])
    line=f"c={c:2d} rmsbias(Ecoupled-Ufrozen)={np.sqrt(np.nanmean((cm-Ubar)[xs>=6]**2)):.3f}(X>=6) "
    for D in (0.1,0.5,2.0):
        m=int(round(D/dt)); I=winI(r,dt,m); X0=X[:I.shape[0]*m:m]
        line+=f"| D={D}: "
        for g,fn in groups.items():
            s=fn(X0); line+=f"{g} {c*(I[s]**2).mean()/D:.2f}/{10*np.interp(X0[s],xs,gk10).mean():.2f} "
    print(line)
