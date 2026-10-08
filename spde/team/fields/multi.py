import numpy as np,sys; sys.path.insert(0,'.'); from l96 import *
p=(20.,1.,10.,10.,32); K=8; rng=np.random.default_rng(5)
# chaotic fast state at x=8
_,_,_,Yc=run(p,0.0,0.001,1,np.full((1,K),8.),rng.normal(0,0.1,(1,256)),freezeX=True,spin=3.)
xs=np.array([-2.,0.,1.,2.,3.,4.,5.,6.])
for name,Y0 in (('from small random Y',rng.normal(0,0.1,(len(xs),256))),('from chaotic Y (x=8)',np.repeat(Yc,len(xs),0))):
    _,U,_,_=run(p,20.,0.001,2,np.repeat(xs[:,None],K,1),Y0,freezeX=True,spin=0.)
    print(name); 
    for i,x in enumerate(xs):
        print("  x=%4.1f  Ubar first5 %.2f last10 %.2f  VarU first5 %.3f last10 %.3f"%(x,U[:2500,i].mean(),U[-5000:,i].mean(),U[:2500,i].var(),U[-5000:,i].var()))
