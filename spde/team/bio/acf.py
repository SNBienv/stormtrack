import numpy as np, l96_dual as L
J=[8,16,32,64,128]; tot=np.array([6.728,4.114,3.459,3.373,3.573]); intr=np.array([7.442,3.296,1.460,0.722,0.331])
A=np.vstack([np.ones(5),1/np.array(J)]).T; print("tot = A + B/J fit:",np.linalg.lstsq(A,tot,rcond=None)[0])
print("int slope:",np.polyfit(np.log(J),np.log(intr),1)[0], " J*int:",np.round(intr*np.array(J),1))
for Jv in (32,128):
    X,U1,U2=L.run(Jv,30.0,seed=3)
    Ad,Lg=L.design(X,1)
    e1=(U1[Lg:].ravel()-Ad@np.linalg.lstsq(Ad,U1[Lg:].ravel(),rcond=None)[0]).reshape(-1,8)
    e2=(U2[Lg:].ravel()-Ad@np.linalg.lstsq(Ad,U2[Lg:].ravel(),rcond=None)[0]).reshape(-1,8)
    def acf(z,lags):
        z=z-z.mean(0); v=(z*z).mean(); return [round(float((z[l:]*z[:-l]).mean()/v),2) for l in lags]
    lags=[1,2,5,10,20,50,100]
    print(f"J={Jv} lags(x0.01)={lags}\n  ACF intrinsic (e1-e2): {acf(e1-e2,lags)}\n  ACF total     (e1+e2): {acf(e1+e2,lags)}")
