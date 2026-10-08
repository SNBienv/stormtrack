import numpy as np, sys, time
def tend(X,Y,F,h,b,c,J):
    # X: (B,K), Y: (B,K*J) ring
    B,K=X.shape
    sY=Y.reshape(B,K,J).sum(2)
    dX=-np.roll(X,1,1)*(np.roll(X,2,1)-np.roll(X,-1,1))-X+F-(h*10.0/b)*sY
    dY=-c*b*np.roll(Y,-1,1)*(np.roll(Y,-2,1)-np.roll(Y,1,1))-c*Y+(h*c/b)*np.repeat(X,J,1)
    return dX,dY,-(h*10.0/b)*sY
def rk4(X,Y,dt,p,freezeX=False):
    k1=tend(X,Y,*p); 
    if freezeX: z=0
    a=lambda k,s: (0 if freezeX else s*k)
    X2,Y2=X+a(k1[0],dt/2),Y+dt/2*k1[1]; k2=tend(X2,Y2,*p)
    X3,Y3=X+a(k2[0],dt/2),Y+dt/2*k2[1]; k3=tend(X3,Y3,*p)
    X4,Y4=X+a(k3[0],dt),Y+dt*k3[1]; k4=tend(X4,Y4,*p)
    Xn=X if freezeX else X+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    return Xn,Y+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]),k1[2]
def run(p,T,dt,every,X0,Y0,freezeX=False,spin=0.0):
    X,Y=X0.copy(),Y0.copy()
    ns=int(round(spin/dt))
    for _ in range(ns): X,Y,_=rk4(X,Y,dt,p,freezeX)
    n=int(round(T/dt)); Xs=[];Us=[]
    for i in range(n):
        Xo=X; X,Y,U=rk4(X,Y,dt,p,freezeX)
        if i%every==0: Xs.append(Xo.copy()); Us.append(U.copy())
    return np.array(Xs),np.array(Us),X,Y
