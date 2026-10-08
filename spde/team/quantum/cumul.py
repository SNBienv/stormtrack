import numpy as np
from scipy import stats
D=np.load('frozen2.npz'); Y=D['Yun'][:,1,:].astype(float)  # x0=10
Dt=np.load('truth.npz')
for m in [1,2,4,8,16,32]:
    S=Y.reshape(Y.shape[0],-1,m).sum(-1).ravel()
    print('frozen x0=10  m=%2d  skew %+.3f  exkurt %+.3f  var/m %.4f'%(m,stats.skew(S),stats.kurtosis(S),S.var()/m))
