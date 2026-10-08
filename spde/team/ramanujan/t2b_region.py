"""Which (skewness, excess kurtosis) pairs can the truncated Champernowne reach? (alpha=1 wlog)"""
import numpy as np
from scipy.integrate import quad
def sk(lam, c):
    f = lambda v: 1/(np.cosh(v-c)+lam)
    I = [quad(lambda v: v**r*f(v), 0, np.inf, points=None, limit=200, epsabs=0, epsrel=1e-12)[0] for r in range(5)]
    m = [I[r]/I[0] for r in range(5)]
    k2 = m[2]-m[1]**2; k3 = m[3]-3*m[2]*m[1]+2*m[1]**3
    k4 = m[4]-4*m[3]*m[1]-3*m[2]**2+12*m[2]*m[1]**2-6*m[1]**4
    return k3/k2**1.5, k4/k2**2, f(0)/I[0]*np.sqrt(k2)
from scipy.special import gamma as G
for k in [1.7, 2.0, 2.3]:
    g = [G(1+i/k) for i in range(5)]
    m=g; k2=m[2]-m[1]**2; k3=m[3]-3*m[2]*m[1]+2*m[1]**3; k4=m[4]-4*m[3]*m[1]-3*m[2]**2+12*m[2]*m[1]**2-6*m[1]**4
    print(f"Weibull k={k}: skew {k3/k2**1.5:.3f} exkurt {k4/k2**2:.3f}")
print("lam  alpha*v0  skew  exkurt  f(0)*sd")
best=[]
for lam in [-0.99,-0.9,-0.7,-0.4,0,0.5,1,3,10,100,1e3]:
    for c in np.linspace(0,8,33):
        s,kk,f0=sk(lam,c); best.append((lam,c,s,kk,f0))
best=np.array(best)
# closest to Weibull k=2 (0.631, 0.245)
d=(best[:,2]-0.631)**2+(best[:,3]-0.245)**2
for i in np.argsort(d)[:5]: print(" ".join(f"{x:8.3f}" for x in best[i]))
