import numpy as np, sys
sys.path.insert(0, '.')
from l96 import *
from scipy import stats
D = np.load('truth.npz')
X, U, B, varY = D['X'], D['U'], D['B'], D['varY']   # (N,M,K)
dts = 0.005
N = X.shape[0]
print('samples', X.size, ' <X>=%.3f sd=%.3f  <U>=%.3f sd=%.3f' % (X.mean(), X.std(), U.mean(), U.std()))

# 1. exact sector budget: dU/dt = -c U - (h^2 c J/b^2)*c... check with centered FD
alpha = hcb * h * c * J / b   # = coefficient of X in dU/dt (with sign -)
dUdt = (U[2:] - U[:-2]) / (2*dts)
rhs = -c*U[1:-1] - alpha*X[1:-1] - hcb*B[1:-1]
print('budget: coeff X =', alpha, ' FD residual rms / rms(dUdt) = %.3e' % (np.sqrt(((dUdt-rhs)**2).mean())/dUdt.std()))
print('edge-flux B: mean %.3f sd %.3f skew %.3f exkurt %.3f' % (B.mean(), B.std(), stats.skew(B.ravel()), stats.kurtosis(B.ravel())))
print('static equilibrium slope -h^2 c J / b^2 /... = U* = -%.3f X' % (alpha/c))

# 2. Arnold et al cubic fit and AR(1) residual
x = X.ravel(); u = U.ravel()
p = np.polyfit(x, u, 3)
print('cubic U_det coeffs (b3,b2,b1,b0):', np.round(p, 5))
r = U - np.polyval(p, X)
print('R^2 of cubic: %.3f' % (1 - r.var()/U.var()))
phi = (r[1:]*r[:-1]).mean() / r.var()
sig_r = r.std(); tau_r = -dts/np.log(phi); sig_e = sig_r*np.sqrt(1-phi**2)
print('residual: sd %.3f  phi(0.005) %.4f  tau %.4f  sig_e %.4f  skew %.3f  exkurt %.3f' %
      (sig_r, phi, tau_r, sig_e, stats.skew(r.ravel()), stats.kurtosis(r.ravel())))
# integral time scale
def acf(z, nlag):
    z = z - z.mean(0)
    return np.array([(z[l:]*z[:len(z)-l]).mean() for l in range(nlag)]) / z.var()
ac = acf(r, 200)
print('residual acf at lags 0.05,0.1,0.25,0.5,1.0:', np.round(ac[[10,20,50,100,199]], 3))
iz = np.argmax(ac < 0) if np.any(ac < 0) else len(ac)
print('integral time of residual (to first zero) %.4f' % (ac[:iz].sum()*dts - 0.5*dts))
# state dependence
bins = np.percentile(x, [0, 10, 30, 50, 70, 90, 100])
idx = np.digitize(x, bins[1:-1])
rr = r.ravel()
for i in range(6):
    s = idx == i
    print('  X bin [%.1f,%.1f]: sd(r)=%.3f  mean varY=%.3f  skew r=%.2f' % (bins[i], bins[i+1], rr[s].std(), varY.ravel()[s].mean(), stats.skew(rr[s])))

# 3. Mori-Zwanzig split: U = memory term + noise, kernel K(t)=alpha e^{-ct}
a = np.exp(-c*dts)
Umem = np.zeros_like(U); Umem[0] = U[0]
# exact discretisation of dU/dt = -cU - alpha X with X piecewise linear (trapezoid)
for i in range(1, N):
    Umem[i] = a*Umem[i-1] - alpha*(1-a)/c*0.5*(X[i]+X[i-1])
zeta = (U - Umem)[200:]
print('MZ: Var(U)=%.3f  Var(zeta)=%.3f (noise part)  Var(r cubic)=%.3f' % (U.var(), zeta.var(), r.var()))
print('    R^2 of memory term alone: %.3f' % (1 - zeta.var()/U[200:].var()))
acz = acf(zeta, 120)
print('    zeta acf at 0.05,0.1,0.2,0.4:', np.round(acz[[10,20,40,80]], 3), ' vs kernel shape e^{-ct}:', np.round(np.exp(-c*np.array([0.05,0.1,0.2,0.4])),3))
print('    Zwanzig "temperature" T_eff = Var(zeta)/K(0) = %.4f ; mean sector varY = %.3f' % (zeta.var()/alpha, varY.mean()))
print('    zeta skew %.3f exkurt %.3f' % (stats.skew(zeta.ravel()), stats.kurtosis(zeta.ravel())))
np.savez('fit.npz', p=p, phi=phi, sig_r=sig_r, tau_r=tau_r)
