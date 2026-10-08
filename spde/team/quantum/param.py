import numpy as np, sys
sys.path.insert(0,'.')
from l96 import K,F
D=np.load('truth.npz'); Xt=D['X']; U=D['U']
fit=np.load('fit.npz'); p=fit['p']; phi=float(fit['phi']); sig_r=float(fit['sig_r'])
dts=0.005
r=U-np.polyval(p,Xt)
def f(X, extra=0.0):
    return -np.roll(X,1,-1)*(np.roll(X,2,-1)-np.roll(X,-1,-1)) - X + F + np.polyval(p,X) + extra
def step(X, extra=0.0, dt=dts):
    k1=f(X,extra); k2=f(X+.5*dt*k1,extra); k3=f(X+.5*dt*k2,extra); k4=f(X+dt*k3,extra)
    return X+dt/6*(k1+2*k2+2*k3+k4)
# --- 1. short-range error growth of deterministic model vs Green-Kubo prediction
N,M,_=Xt.shape; L=200
starts=np.arange(1000, N-L-1, 100)
X0=Xt[starts].reshape(-1,K)
Xd=X0.copy(); mse=[]; bias=[]
for l in range(1,L+1):
    Xd=step(Xd)
    err=Xt[starts+l].reshape(-1,K)-Xd
    mse.append((err**2).mean()); bias.append(err.mean())
cs=np.cumsum(r,0)*dts
print('lead   MSE_det   Var(int r)+mean^2 (GK, no Jacobian)   mean err')
for l in [2,5,10,20,40,100,200]:
    I=(cs[l:]-cs[:-l]); gk=(I**2).mean()
    print('%.3f  %.4f   %.4f   %+.4f'%(l*dts, mse[l-1], gk, bias[l-1]))
# --- 2. climatology: deterministic vs AR(1) vs white(GK) vs truth
rng=np.random.default_rng(5)
Tlong=2000; n=int(Tlong/dts); Mc=8
def clim(kind):
    X=Xt[-1].copy(); e=np.zeros_like(X); out=[]
    for i in range(n):
        if kind=='ar1':
            e=phi*e+sig_r*np.sqrt(1-phi**2)*rng.standard_normal(X.shape)
            X=step(X,e)
        elif kind=='white':
            X=step(X)+np.sqrt(2*Dgk*dts)*rng.standard_normal(X.shape)
        else:
            X=step(X)
        if i>2000 and i%4==0: out.append(X.copy())
    return np.array(out).ravel()
Dgk=float(sys.argv[1]) if len(sys.argv)>1 else 0.08
tr=Xt.ravel()
def summ(z): return 'mean %.3f  sd %.3f  p1 %.2f  p99 %.2f  p99.9 %.2f'%(z.mean(),z.std(),*np.percentile(z,[1,99,99.9]))
print('truth  ', summ(tr))
for kind in ['det','ar1','white']:
    print('%-6s '%kind, summ(clim(kind)))
