import sys, numpy as np
sys.dont_write_bytecode=True; import os; SPDE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."); sys.path.insert(0,SPDE)
exec(open(os.path.join(SPDE,"mixing_closure.py")).read().split("FAMILIES = {")[0])
from scipy import stats
N,nu=200,0.25
x=np.arange(N)/N
R=np.roll(np.eye(N),1,axis=0); Lap=np.roll(np.eye(N),1,0)+np.roll(np.eye(N),-1,0)-2*np.eye(N)
K=np.diag(a_phys)@(np.eye(N)+nu*Lap)@R
raw0=[weibull_raw(k0,lam0,r) for r in (1,2,3,4)]; c0=cumulants_from_raw(*raw0)
def cf_q(p,m,k2,k3,k4,use4=True):
    z=stats.norm.ppf(p); s=np.sqrt(k2); g1=k3/s**3; g2=k4/k2**2 if use4 else 0
    w=z+g1/6*(z**2-1)+g2/24*(z**3-3*z)-g1**2/36*(2*z**3-5*z)
    return m+s*w
L,M=40.0,2**17; dx=L/M; edges=np.arange(M+1)*dx
for T in (10,40):
  P=np.linalg.matrix_power(K,T)
  m=P@c0[0]; k2=(P**2)@c0[1]; k3=(P**3)@c0[2]; k4=(P**4)@c0[3]
  print(f"--- t={T}: rel. error of q99.9 in % (exact = FFT convolution of the {N} scaled Weibull pmfs)")
  for i in (0,50,100,150):
    w=P[i]; idx=np.where(w>1e-9)[0]
    Fh=np.ones(M,complex)
    for j in idx:
        cdf=stats.weibull_min.cdf(edges/w[j]+0*edges,k0[j],scale=lam0[j])
        pm=np.diff(cdf); pm[-1]+=1-cdf[-1]
        # bin-centre placement: shift handled by subtracting dx/2 per term after
        Fh*=np.fft.fft(pm)
    pmf=np.real(np.fft.ifft(Fh)); pmf=np.clip(pmf,0,None); pmf/=pmf.sum()
    cdf=np.cumsum(pmf); grid=edges[1:]-dx/2*0  # mass of bin b sits at its midpoint; sum of len(idx) midpoints offset
    off=dx/2*len(idx)  # each term's mass placed at bin left edge -> add dx/2 per term
    def q(p): return np.interp(p,cdf,edges[:-1])+off
    qe=q(0.999)
    ew=match_expweib(m[i],k2[i],k3[i])[0](0.999)
    print(f"pt {i:3d} q999={qe:6.3f} (check mean {np.dot(pmf,edges[:-1])+off:.4f} vs {m[i]:.4f}) "
          f"EW3 {100*(ew/qe-1):+.3f}  CF(k2,k3) {100*(cf_q(.999,m[i],k2[i],k3[i],k4[i],False)/qe-1):+.3f} "
          f"CF(k2,k3,k4) {100*(cf_q(.999,m[i],k2[i],k3[i],k4[i])/qe-1):+.3f}  N_eff {w.sum()**2/(w**2).sum():.1f}")
