import numpy as np,sys,glob
def winvar(r,dt,m):
    # r: (n, ...) ; variance of non-overlapping window integrals of length m*dt, divided by window length
    n=(r.shape[0]//m)*m; I=r[:n].reshape(n//m,m,*r.shape[1:]).sum(1)*dt
    return I.var(0)/(m*dt)
def acf_time(r,dt,maxlag):
    r=r-r.mean(0); v=(r*r).mean()
    return np.array([ (r[:r.shape[0]-l]*r[l:]).mean()/v for l in range(maxlag)])
out={}
for f in sorted(glob.glob('frozen_*.npz')):
    d=np.load(f); U=d['U']; xs=d['xs']; dt=float(d['dt'])
    Ubar=U.mean((0,2)); res=[]
    for ib,x in enumerate(xs):
        r=U[:,ib,:]-U[:,ib,:].mean()
        C=acf_time(r,dt,int(1.0/dt))
        tau=np.trapezoid(C,dx=dt); 
        s2=[winvar(r,dt,int(round(D/dt))).mean() for D in (0.1,0.5,1.0)]
        res.append((x,Ubar[ib],r.var(),tau,*s2))
    out[f]=np.array(res)
    print(f); print("   x     Ubar    Var(U)  tau_int  s2(D=.1) s2(.5) s2(1.)")
    for row in res: print("  %5.1f %7.3f %8.3f %7.4f  %7.3f %7.3f %7.3f"%row)
np.save('frozen_summary.npy',out,allow_pickle=True)
