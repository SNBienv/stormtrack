import sys, numpy as np
sys.dont_write_bytecode=True
from scipy import special
import os; SPDE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."); sys.path.insert(0,SPDE)
exec(open(sys.argv[1] if len(sys.argv)>1 else os.path.join(SPDE,"mixing_closure.py")).read().split("FAMILIES = {")[0])
N,nu=200,0.25
x=np.arange(N)/N
k0=2.0+0.3*np.sin(2*np.pi*x); lam0=7.0+2.0*np.cos(2*np.pi*x); a_phys=1.0+0.004*np.sin(4*np.pi*x)
R=np.roll(np.eye(N),1,axis=0); Lap=np.roll(np.eye(N),1,0)+np.roll(np.eye(N),-1,0)-2*np.eye(N)
K=np.diag(a_phys)@(np.eye(N)+nu*Lap)@R
raw0=[weibull_raw(k0,lam0,r) for r in (1,2,3,4)]; c0=cumulants_from_raw(*raw0)
g10=c0[2]/c0[1]**1.5; g20=c0[3]/c0[1]**2
print("initial median skew %.3f exkurt %.3f ratio g2/g1^2 %.3f"%(np.median(g10),np.median(g20),np.median(g20/g10**2)))
P=np.eye(N)
for t in range(1,81):
    P=K@P
    if t in (1,2,5,10,20,40,80):
        m=P@c0[0]; k2=(P**2)@c0[1]; k3=(P**3)@c0[2]; k4=(P**4)@c0[3]
        rows=[]
        for i in range(0,N,20):
            g1=k3[i]/k2[i]**1.5; g2=k4[i]/k2[i]**2
            _,ew,_=match_expweib(m[i],k2[i],k3[i]); _,w3,_=match_weibull3(m[i],k2[i],k3[i])
            w=P[i]; CS=(w**4).sum()*(w**2).sum()/((w**3).sum())**2
            Neff=1/(w**2).sum()*(w.sum())**2
            rows.append((g1,g2,ew-g2,w3-g2,g2/g1**2,CS,Neff))
        r=np.median(np.array(rows),0)
        print("t=%2d skew %.3f exk %.4f  EWgap %+.4f W3gap %+.4f  g2/g1^2 %.3f  CSfac %.3f Neff %.1f"%((t,)+tuple(r)))
# EW shape map: (skew, exkurt) for grid of k,a, and slope g2/g1^2 near small skew
print("\nEW family: g2/g1^2 along its curve at fixed CV")
for k in (2,3,4,6,10):
  for a in (0.3,1,3,10):
    m1,c2,c3,c4=cumulants_from_raw(*quantile_raw(ew_quantile(k,a)))
    print("k=%4.1f a=%5.1f CV %.3f skew %+.3f exk %+.3f"%(k,a,np.sqrt(c2)/m1,c3/c2**1.5,c4/c2**2))
