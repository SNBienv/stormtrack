# Single-reporter decomposition: S = sqrt(2 beta G), G~Gamma(m, theta/m) (intrinsic bursts), ln beta ~ N(-s^2/2, s^2) (extrinsic)
import numpy as np, pyreadr
from scipy import stats, optimize, special
D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data") + "/"
irl = pyreadr.read_r(D+"gstat_wind.rda")["wind"]; st=[c for c in irl.columns if c not in ("year","month","day")]
lon = pyreadr.read_r(D+"openair_mydata.rda")["mydata"]
xg,wg=np.polynomial.hermite_e.hermegauss(40); wg=wg/wg.sum()
def nll(t,s,w):
    m,sig,th=np.exp(t)
    b=np.exp(sig*xg-sig**2/2)                # quadrature nodes for beta
    # density of S: E=S^2/2 ~ Gamma(m, th*b/m); f_S = f_E * S
    E=(s**2/2)[:,None]; sc=th*b[None,:]/m
    lf=stats.gamma.logpdf(E,m,scale=sc)+np.log(s)[:,None]
    return -(w*special.logsumexp(lf,b=wg[None,:],axis=1)).sum()
def fit(s):
    s=s[s>0]; s0_=s; s,w=np.unique(np.round(s,3),return_counts=True); best=None
    for m0 in (0.8,2.5):
        for s0 in (0.2,0.6):
            r=optimize.minimize(nll,[np.log(m0),np.log(s0),np.log((s**2/2).mean())],args=(s,w),method="Nelder-Mead",options=dict(maxiter=3000))
            if best is None or r.fun<best.fun: best=r
    m,sig,th=np.exp(best.x)
    llnak=(w*stats.nakagami.logpdf(s,*stats.nakagami.fit(s0_,floc=0))).sum()
    return m,sig,1/m,np.exp(sig**2)-1,2*(-best.fun-llnak)
ws=lon[["date","ws"]].dropna(); h=ws.ws[ws.ws>0].to_numpy()
d=ws.set_index("date").ws.resample("D").mean().dropna().to_numpy()
sets=[("London hourly",h),("London daily",d)]+[("IRL "+c,irl[c].to_numpy()*0.514444) for c in st]
for n,s in sets:
    m,sig,ei,ee,lr=fit(s)
    print(f"{n:15s} m={m:5.2f} sigma={sig:4.2f}  eta2_int(E)=1/m={ei:5.3f}  eta2_ext(E)=e^s2-1={ee:5.3f}  LR vs Nakagami={lr:7.1f}")
