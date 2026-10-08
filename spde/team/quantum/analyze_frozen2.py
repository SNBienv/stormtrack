import numpy as np, sys
from scipy import stats
sys.path.insert(0,'.')
from l96 import *
D=np.load('frozen2.npz'); U=D['U'].astype(float); Yun=D['Yun'].astype(float); x0s=D['x0s']; d=float(D['d'])
dts=0.005; Lmax=100  # integrate correlations to 0.5 MTU
def block_se(z, nb=20):
    m=np.array([b.mean() for b in np.array_split(z,nb)]); return m.std()/np.sqrt(nb)
for ix,x0 in enumerate(x0s):
    U0=U[:,3*ix,0]; Up=U[:,3*ix+1,0]; Um=U[:,3*ix+2,0]
    Uun=U[:,3*ix,:]                       # unperturbed, all sectors equivalent
    direct=(Up.mean()-Um.mean())/(2*d)
    se=np.hypot(block_se(Up),block_se(Um))/(2*d)
    # neighbours: leak
    lk=(U[:,3*ix+1,1].mean()-U[:,3*ix+2,1].mean())/(2*d)
    Y=Yun[:,ix,:]; mu=Y.mean(); Yp=Y-mu; n=Y.shape[1]
    # circulant covariance row
    Fy=np.fft.rfft(Yp,axis=1); crow=np.fft.irfft((np.abs(Fy)**2).mean(0),n=n)/n
    sigY2=crow[0]
    lam=np.fft.rfft(crow)
    resp=[]; resp_diag=[]
    Uc=Uun-Uun.mean(0)
    for k in range(K):
        e=np.zeros(n); e[k*J:(k+1)*J]=hcb
        w=np.fft.irfft(np.fft.rfft(e)/lam,n=n)
        z=Yp@w
        Ck=np.array([(Uc[l:,k]*z[:len(z)-l]).mean() for l in range(Lmax)])
        resp.append((Ck.sum()-0.5*Ck[0])*dts)
        Cd=np.array([(Uc[l:,k]*Uc[:len(z)-l,k]).mean() for l in range(Lmax)])
        resp_diag.append(-(Cd.sum()-0.5*Cd[0])*dts/sigY2)
    varU=Uun.var(); 
    acU=np.array([(Uc[l:]*Uc[:len(Uc)-l]).mean() for l in range(Lmax)])/Uc.var()
    tauU=(acU.sum()-0.5)*dts
    print('x0=%4.1f  sigma_Y^2=%.4f  <U>=%.3f  Var(U|x)=%.3f tau_U=%.4f skewU=%.2f exkurtU=%.2f'%(x0,sigY2,Uun.mean(),varU,tauU,stats.skew(Uc.ravel()),stats.kurtosis(Uc.ravel())))
    print('     direct dU0/dX0 = %.3f +- %.3f (neighbour dU1/dX0=%.3f); quasi-Gaussian FDT = %.3f ; Einstein (T=sigY^2) = %.3f ; no-leak bound -h^2cJ/b^2/c=-3.2'%(direct,se,lk,np.mean(resp),np.mean(resp_diag)))
