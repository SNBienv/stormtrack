import numpy as np, pyreadr
from scipy import stats, optimize
D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data") + "/"
irl = pyreadr.read_r(D+"gstat_wind.rda")["wind"]; st=[c for c in irl.columns if c not in ("year","month","day")]
lon = pyreadr.read_r(D+"openair_mydata.rda")["mydata"]
def ew(s):
    k0,_,l0=stats.weibull_min.fit(s,floc=0)
    nll=lambda t:-stats.exponweib.logpdf(s,np.exp(t[0]),np.exp(t[1]),scale=np.exp(t[2])).sum()
    r=min((optimize.minimize(nll,[np.log(a0),np.log(k0),np.log(l0)],method="Nelder-Mead",options=dict(maxiter=4000,xatol=1e-6,fatol=1e-6)) for a0 in (0.5,1,2,4)),key=lambda r:r.fun)
    a,k,l=np.exp(r.x); return k0,a,k,l
def nak(s):
    m,_,O=stats.nakagami.fit(s,floc=0); return m
ws=lon[["date","ws"]].dropna(); h=ws.ws[ws.ws>0].to_numpy()
d=ws.set_index("date").ws.resample("D").mean().dropna().to_numpy(); d=d[d>0]
for name,s in [("London hourly",h),("London daily",d)]+[("IRL "+c, irl[c].to_numpy()*0.514444) for c in st]+[("IRL 12-stn mean",irl[st].mean(1).to_numpy()*0.514444)]:
    s=s[s>0]; kw,a,k,l=ew(s); print(f"{name:18s} Weibull k={kw:4.2f} | EW a={a:5.2f} k={k:4.2f} lam={l:5.2f} | a*k/2={a*k/2:4.2f} | Nakagami m={nak(s):4.2f} | CV2={s.var()/s.mean()**2:.3f}")
