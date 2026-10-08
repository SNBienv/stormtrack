import numpy as np, sys
from scipy import stats
sys.path.insert(0,'.')
from l96 import *
D = np.load('truth.npz'); X, U, B, varY = D['X'], D['U'], D['B'], D['varY']
dts=0.005
x=X.ravel(); bb=B.ravel()
pb=np.polyfit(x,bb,3); Bp=B-np.polyval(pb,X)
print('<B|X> cubic coeffs', np.round(pb,4), ' R2=%.3f'%(1-Bp.var()/B.var()))
print("B' sd %.2f skew %.3f exkurt %.3f"%(Bp.std(),stats.skew(Bp.ravel()),stats.kurtosis(Bp.ravel())))
# correlation of B with sector TKE
print('corr(B, varY) = %.3f   corr(B,X)=%.3f'%(np.corrcoef(bb,varY.ravel())[0,1],np.corrcoef(bb,x)[0,1]))
# Green-Kubo: Var of time-integral of cubic residual over windows
p=np.load('fit.npz')['p']; r=U-np.polyval(p,X)
cs=np.cumsum(r,0)*dts
print('window T, Var(int r)/(2T) [-> D=sig^2 tau_int]')
for L in [2,5,10,20,40,100,200,400]:
    I=cs[L:]-cs[:-L]; T=L*dts
    print('  T=%.3f  %.4f'%(T, I.var()/(2*T)))
# full acf integral
z=r-r.mean()
ac=np.array([(z[l:]*z[:len(z)-l]).mean() for l in range(400)])/z.var()
for Lmax in [20,40,100,200,400]:
    print(' acf integral to %.2f: %.4f'%(Lmax*dts,(ac[:Lmax].sum()-0.5)*dts))
